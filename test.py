# Copyright (c) 2021, Hitachi America Ltd. All rights reserved.
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

from contract_nli.conf import load_conf
from contract_nli.dataset.dataset import load_and_cache_examples, \
    load_and_cache_features
from contract_nli.evaluation import evaluate_all
from contract_nli.model.classification import BertForClassification
from contract_nli.model.identification_classification import \
    MODEL_TYPE_TO_CLASS
from contract_nli.postprocess import format_json, compute_prob_calibration_coeff
from contract_nli.predictor import predict, predict_classification

logger = logging.getLogger(__name__)


@click.command()
@click.option('--dev-dataset-path', type=click.Path(exists=True), default=None)
@click.argument('model-dir', type=click.Path(exists=True))
@click.argument('dataset-path', type=click.Path(exists=True))
@click.argument('output-prefix', type=str)
def main(dev_dataset_path, model_dir, dataset_path, output_prefix):
    conf: dict = load_conf(os.path.join(model_dir, 'conf.yml'))

    device = torch.device("cuda" if torch.cuda.is_available() and not conf['no_cuda'] else "cpu")
    n_gpu = 0 if conf['no_cuda'] else torch.cuda.device_count()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    transformers.utils.logging.set_verbosity_info()
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info("Loading models with following conf %s",
                {k: v for k, v in conf.items() if k != 'raw_yaml'})

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        do_lower_case=conf['do_lower_case'],
        cache_dir=conf['cache_dir'],
        use_fast=False
    )
    config = AutoConfig.from_pretrained(
        model_dir,
        cache_dir=conf['cache_dir']
    )
    if conf['task'] == 'identification_classification':
        model = MODEL_TYPE_TO_CLASS[config.model_type].from_pretrained(
            model_dir, cache_dir=conf['cache_dir']
        )
    else:
        model = BertForClassification.from_pretrained(
            model_dir, cache_dir=conf['cache_dir'])

    model.to(device)

    if dev_dataset_path is not None:
        if conf['task'] != 'identification_classification':
            raise click.BadOptionUsage(
                '--dev-dataset-path',
                '--dev-dataset-path cannot be used when the task is not identification_classification')
        examples = load_and_cache_examples(
            dev_dataset_path,
            local_rank=-1,
            overwrite_cache=True,
            cache_dir='.'
        )
        dataset, features = load_and_cache_features(
            dev_dataset_path,
            examples,
            tokenizer,
            max_seq_length=conf['max_seq_length'],
            doc_stride=conf.get('doc_stride', None),
            max_query_length=conf['max_query_length'],
            dataset_type=conf['task'],
            symbol_based_hypothesis=conf['symbol_based_hypothesis'],
            threads=None,
            local_rank=-1,
            overwrite_cache=True,
            labels_available=True,
            cache_dir='.'
        )
        all_results = predict(
            model, dataset, examples, features,
            per_gpu_batch_size=conf['per_gpu_eval_batch_size'],
            device=device, n_gpu=n_gpu,
            weight_class_probs_by_span_probs=conf[
                'weight_class_probs_by_span_probs'])
        calibration_coeff = compute_prob_calibration_coeff(
            examples, all_results)
    else:
        calibration_coeff = None

    examples = load_and_cache_examples(
        dataset_path,
        local_rank=-1,
        overwrite_cache=True,
        cache_dir='.'
    )
    dataset, features = load_and_cache_features(
        dataset_path,
        examples,
        tokenizer,
        max_seq_length=conf['max_seq_length'],
        doc_stride=conf.get('doc_stride', None),
        max_query_length=conf['max_query_length'],
        dataset_type=conf['task'],
        symbol_based_hypothesis=conf['symbol_based_hypothesis'],
        threads=None,
        local_rank=-1,
        overwrite_cache=True,
        labels_available=True,
        cache_dir='.'
    )

    if conf['task'] == 'identification_classification':
        all_results = predict(
            model, dataset, examples, features,
            per_gpu_batch_size=conf['per_gpu_eval_batch_size'],
            device=device, n_gpu=n_gpu,
            weight_class_probs_by_span_probs=conf['weight_class_probs_by_span_probs'],
            calibration_coeff=calibration_coeff)
    else:
        all_results = predict_classification(
            model, dataset, features,
            per_gpu_batch_size=conf['per_gpu_eval_batch_size'],
            device=device, n_gpu=n_gpu)

    result_json = format_json(examples, all_results)
    with open(output_prefix + 'result.json', 'w') as fout:
        json.dump(result_json, fout, indent=2)
    with open(dataset_path) as fin:
        test_dataset = json.load(fin)
    metrics = evaluate_all(test_dataset, result_json,
                           [1, 3, 5, 8, 10, 15, 20, 30, 40, 50],
                           conf['task'])
    logger.info(f"Results@: {json.dumps(metrics, indent=2)}")
    with open(output_prefix + 'metrics.json', 'w') as fout:
        json.dump(metrics, fout, indent=2)


if __name__ == "__main__":
    main()
