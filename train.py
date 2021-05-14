# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.
# Copyright (c) 2021, Hitachi America Ltd. All rights reserved.
# This file has been adopted from https://github.com/huggingface/transformers
# /blob/0c9bae09340dd8c6fdf6aa2ea5637e956efe0f7c/examples/question-answering/run.py
# See git log for changes.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import logging
import os

import click
import torch
import transformers
from transformers import AutoConfig, AutoTokenizer
from transformers.trainer_utils import is_main_process

from contract_nli.conf import load_conf
from contract_nli.dataset.dataset import load_and_cache_examples, load_and_cache_features
from contract_nli.dataset.encoder import SPAN_TOKEN
from contract_nli.evaluation import evaluate_all
from contract_nli.model.identification_classification import \
    MODEL_TYPE_TO_CLASS, update_config
from contract_nli.model.classification import BertForClassification
from contract_nli.postprocess import format_json
from contract_nli.predictor import predict, predict_classification
from contract_nli.trainer import Trainer, setup_optimizer
from contract_nli.utils import set_seed, distributed_barrier

logger = logging.getLogger(__name__)


@click.command()
@click.argument('conf', type=click.Path(exists=True))
@click.argument('output-dir', type=click.Path(exists=False))
@click.option(
    '--local_rank', type=int, default=-1,
    help='This is automatically set by torch.distributed.launch.')
@click.option('--shared-filesystem', type=int, default=-1)
def main(conf, output_dir, local_rank, shared_filesystem):
    conf: dict = load_conf(conf)

    # Setup CUDA, GPU & distributed training
    if local_rank == -1 or conf['no_cuda']:
        device = torch.device("cuda" if torch.cuda.is_available() and not conf['no_cuda'] else "cpu")
        n_gpu = 0 if conf['no_cuda'] else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        torch.distributed.init_process_group(backend="nccl")
        n_gpu = 1

    # if this is a main process in a node
    local_main = is_main_process(local_rank)
    # if this is a main process in the whole distributed training
    all_main = local_rank == -1 or torch.distributed.get_rank() == 0
    # if this is a main process on a filesystem
    fs_main = (shared_filesystem and all_main) or ((not shared_filesystem) and local_main)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if local_main else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        local_rank,
        device,
        n_gpu,
        bool(local_rank != -1),
        conf['fp16'],
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    # Set seed
    set_seed(conf['seed'])

    with distributed_barrier(not fs_main, local_rank != -1):
        config = AutoConfig.from_pretrained(
            conf['config_name'] if conf['config_name'] else conf['model_name_or_path'],
            cache_dir=conf['cache_dir']
        )
        tokenizer = AutoTokenizer.from_pretrained(
            conf['tokenizer_name'] if conf['tokenizer_name'] else conf['model_name_or_path'],
            do_lower_case=conf['do_lower_case'],
            cache_dir=conf['cache_dir'],
            use_fast=False
        )
        if conf['task'] == 'identification_classification':
            config = update_config(
                config, impossible_strategy='ignore',
                class_loss_weight=conf['class_loss_weight'])
            model = MODEL_TYPE_TO_CLASS[config.model_type].from_pretrained(
                conf['model_name_or_path'],
                from_tf=bool(".ckpt" in conf['model_name_or_path']),
                config=config,
                cache_dir=conf['cache_dir']
            )
        else:
            model = BertForClassification.from_pretrained(
                conf['model_name_or_path'],
                from_tf=bool(".ckpt" in conf['model_name_or_path']),
                config=config,
                cache_dir=conf['cache_dir']
            )

    logger.info("Training/evaluation parameters %s",
                {k: v for k, v in conf.items() if k != 'raw_yaml'})

    # Before we do anything with models, we want to ensure that we get fp16 execution of torch.einsum if conf['fp16'] is set.
    # Otherwise it'll default to "promote" mode, and we'll get fp32 operations. Note that running `--fp16_opt_level="O2"` will
    # remove the need for this code, but it is still valid.
    if conf['fp16']:
        try:
            import apex
            apex.amp.register_half_function(torch, "einsum")
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    with distributed_barrier(not fs_main, local_rank != -1):
        examples = load_and_cache_examples(
            conf['train_file'],
            local_rank=local_rank,
            overwrite_cache=conf['overwrite_cache'],
            cache_dir='.',
        )
        if conf['task'] == 'identification_classification':
            n_added_token = tokenizer.add_special_tokens(
                {'additional_special_tokens': tokenizer.additional_special_tokens + [SPAN_TOKEN]})
            if n_added_token == 0:
                logger.warning(
                    f'SPAN_TOKEN "{SPAN_TOKEN}" was not added. You can safely ignore'
                    ' this warning if you are retraining a model from this train.py')
            else:
                span_token_id = tokenizer.additional_special_tokens_ids[
                    tokenizer.additional_special_tokens.index(SPAN_TOKEN)]
                logger.warning(
                    f'SPAN_TOKEN "{SPAN_TOKEN}" was added as "{span_token_id}". You can safely ignore'
                    ' this warning if you are training a model from pretrained LMs.')
        if conf['symbol_based_hypothesis']:
            hypothesis_symbols = sorted(set([e.hypothesis_symbol for e in examples]))
            n_added_token = tokenizer.add_special_tokens(
                {'additional_special_tokens': tokenizer.additional_special_tokens + hypothesis_symbols})
            if n_added_token == 0:
                logger.warning(
                    f'SPAN_TOKEN "{SPAN_TOKEN}" was not added. You can safely ignore'
                    ' this warning if you are retraining a model from this train.py')
            else:
                assert n_added_token == len(hypothesis_symbols)
                hypothesis_symbol_dic = {
                    s: tokenizer.additional_special_tokens_ids[tokenizer.additional_special_tokens.index(s)]
                    for s in hypothesis_symbols
                }
                logger.warning(
                    f'Hypothesis symbols were added as "{hypothesis_symbol_dic}". '
                    'You can safely ignore this warning if you are training a '
                    'model from pretrained LMs.')
        model.resize_token_embeddings(len(tokenizer))

        train_dataset = load_and_cache_features(
            conf['train_file'],
            examples,
            tokenizer,
            max_seq_length=conf['max_seq_length'],
            doc_stride=conf.get('doc_stride', None),
            max_query_length=conf['max_query_length'],
            dataset_type=conf['task'],
            symbol_based_hypothesis=conf['symbol_based_hypothesis'],
            threads=None,
            local_rank=local_rank,
            overwrite_cache=conf['overwrite_cache'],
            labels_available=True,
            cache_dir='.',
        )[0]

    if conf['dev_file'] is not None:
        with distributed_barrier(not fs_main, local_rank != -1):
            dev_examples = load_and_cache_examples(
                conf['dev_file'],
                local_rank=local_rank,
                overwrite_cache=conf['overwrite_cache'],
                cache_dir='.'
            )
            dev_dataset, dev_features = load_and_cache_features(
                conf['dev_file'],
                dev_examples,
                tokenizer,
                max_seq_length=conf['max_seq_length'],
                doc_stride=conf.get('doc_stride', None),
                max_query_length=conf['max_query_length'],
                dataset_type=conf['task'],
                symbol_based_hypothesis=conf['symbol_based_hypothesis'],
                threads=None,
                local_rank=local_rank,
                overwrite_cache=conf['overwrite_cache'],
                labels_available=True,
                cache_dir='.'
            )

    else:
        dev_dataset, dev_examples, dev_features = None, None, None

    optimizer = setup_optimizer(
        model, learning_rate=conf['learning_rate'], epsilon=conf['adam_epsilon'],
        weight_decay=conf['weight_decay'])
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        optimizer=optimizer,
        task=conf['task'],
        output_dir=output_dir,
        per_gpu_train_batch_size=conf['per_gpu_train_batch_size'],
        num_epochs=conf['num_epochs'],
        max_steps=conf['max_steps'],
        dev_dataset=dev_dataset,
        valid_steps=conf['valid_steps'],
        per_gpu_dev_batch_size=conf['per_gpu_eval_batch_size'],
        gradient_accumulation_steps=conf['gradient_accumulation_steps'],
        warmup_steps=conf['warmup_steps'],
        max_grad_norm=conf['max_grad_norm'],
        n_gpu=n_gpu,
        local_rank=local_rank,
        fp16=conf['fp16'],
        fp16_opt_level=conf['fp16_opt_level'],
        device=device,
        save_steps=conf['save_steps'])
    trainer.deploy()
    trainer.train()

    # FIXME: Prediction using multiple GPUs
    if not all_main:
        return

    # Setup CUDA, GPU & distributed training
    if local_rank != -1:
        trainer.local_rank = -1

    if conf['early_stopping'] and os.path.exists(trainer.best_checkpoint_dir):
        logger.info(f"Loading best model from {trainer.best_checkpoint_dir}")
        trainer.load(trainer.best_checkpoint_dir)
        trainer.deploy()
        model = trainer.model

    logger.info("Saving model checkpoint to %s", output_dir)
    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    # Take care of distributed/parallel training
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    with open(os.path.join(output_dir, "conf.yml"), 'w') as fout:
        fout.write(conf['raw_yaml'])

    if dev_dataset is not None:
        logger.info("Evaluate the on validation data")
        if conf['task'] == 'identification_classification':
            all_results = predict(
                model, dev_dataset, dev_examples, dev_features,
                per_gpu_batch_size=conf['per_gpu_eval_batch_size'],
                device=device, n_gpu=n_gpu,
                weight_class_probs_by_span_probs=conf['weight_class_probs_by_span_probs'])
        else:
            all_results = predict_classification(
                model, dev_dataset, dev_features,
                per_gpu_batch_size=conf['per_gpu_eval_batch_size'],
                device=device, n_gpu=n_gpu)
        result_json = format_json(dev_examples, all_results)
        with open(os.path.join(output_dir, f'result.json'), 'w') as fout:
            json.dump(result_json, fout, indent=2)
        with open(conf['dev_file']) as fin:
            dev_json = json.load(fin)
        metrics = evaluate_all(dev_json, result_json,
                               [1, 3, 5, 8, 10, 15, 20, 30, 40, 50],
                               conf['task'])
        logger.info(f"Results@: {json.dumps(metrics, indent=2)}")
        with open(os.path.join(output_dir, f'metrics.json'), 'w') as fout:
            json.dump(metrics, fout, indent=2)


if __name__ == "__main__":
    main()
