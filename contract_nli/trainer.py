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
import glob
import json
import logging
import os
import timeit
from typing import List, Tuple, Optional

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup

from contract_nli.dataset.dataset import load_and_cache_examples
from contract_nli.evaluation import evaluate_all
from contract_nli.model.identification_classification import \
    IdentificationClassificationModelOutput
from contract_nli.postprocess import IdentificationClassificationPartialResult, \
    compute_predictions_logits, IdentificationClassificationResult
from contract_nli.summary_writer import SummaryWriter

logger = logging.getLogger(__name__)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def setup_optimizer(model, learning_rate: float, epsilon: float, weight_decay: float):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        },
    ]
    return AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=epsilon)


class Trainer(object):
    def __init__(
            self, model, train_dataset, optimizer, output_dir: str,
            per_gpu_train_batch_size: int, num_epochs: Optional[int] = None, max_steps: Optional[int] = None,
            dev_dataset=None, logging_steps: Optional[int]=None, per_gpu_dev_batch_size: Optional[int]=None,
            gradient_accumulation_steps: int=1, warmup_steps: int=0, max_grad_norm: Optional[float]=None,
            n_gpu: int=1, local_rank: int=-1, fp16: bool=False, fp16_opt_level=None, device=torch.device("cpu"),
            save_steps: Optional[int] = None):
        if local_rank in [-1, 0]:
            self.tb_writer = SummaryWriter(os.path.join(output_dir, 'tensorboard'))

        train_batch_size = per_gpu_train_batch_size * max(1, n_gpu)
        train_sampler = RandomSampler(train_dataset) if local_rank == -1 else DistributedSampler(train_dataset)
        self.train_dataloader = DataLoader(
            train_dataset, sampler=train_sampler, batch_size=train_batch_size)

        if dev_dataset is not None:
            if per_gpu_dev_batch_size is None:
                per_gpu_dev_batch_size = per_gpu_train_batch_size
            dev_batch_size = per_gpu_dev_batch_size * max(1, n_gpu)
            dev_sampler = RandomSampler(dev_dataset) if local_rank == -1 else DistributedSampler(dev_dataset)
            self.dev_dataloader = DataLoader(
                dev_dataset, sampler=dev_sampler, batch_size=dev_batch_size)
        else:
            self.dev_dataloader = None

        if (num_epochs is None) == (max_steps is None):
            raise ValueError('One and only one of num_epochs and max_steps can be specified')
        if num_epochs is not None:
            max_steps = len(self.train_dataloader) // gradient_accumulation_steps * num_epochs
        else:
            num_epochs = (max_steps * gradient_accumulation_steps) // len(self.train_dataloader)

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_steps
        )

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.max_steps = max_steps
        self.fp16 = fp16
        self.fp16_opt_level = fp16_opt_level
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.device = device
        self.n_gpu = n_gpu
        self.local_rank = local_rank
        self.logging_steps = logging_steps
        self.max_grad_norm = max_grad_norm
        self.per_gpu_train_batch_size = per_gpu_train_batch_size
        self.output_dir = output_dir
        self.save_steps = save_steps

        self.global_step = 0

        self.deployed = False

        logger.info("***** Trainer *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Instantaneous batch size per GPU = %d", per_gpu_train_batch_size)
        logger.info(
            f"  Effective batch size (w. parallel, distributed & accumulation) = {self.effective_batch_size}")
        logger.info("  Gradient Accumulation steps = %d", gradient_accumulation_steps)
        logger.info(f" Optimization steps = {max_steps} ({num_epochs} epochs)")

    def deploy(self):
        # This is not included in __init__ to allow loading Trainer
        if self.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            self.model, optimizer = amp.initialize(
                self.model, self.optimizer, opt_level=self.fp16_opt_level)
            self.amp = amp

        # multi-gpu training (should be after apex fp16 initialization)
        if self.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

        # Distributed training (should be after apex fp16 initialization)
        if self.local_rank != -1:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.local_rank], output_device=self.local_rank,
                find_unused_parameters=True
            )
        self.deployed = True

    @property
    def n_samples(self):
        return len(self.train_dataloader)

    @property
    def current_epoch(self):
        return self.global_step // self.n_samples

    @property
    def current_step(self):
        return self.global_step % self.n_samples

    @property
    def is_top(self) -> bool:
        # whether or not
        return self.local_rank in [-1, 0]

    @property
    def effective_batch_size(self) -> int:
        n_gpus = torch.distributed.get_world_size() if self.local_rank != -1 else 1
        return self.per_gpu_train_batch_size * self.gradient_accumulation_steps * n_gpus

    def train(self):
        if not self.deployed:
            raise RuntimeError('Trainer must be deployed before training.')

        self.model.zero_grad()
        while self.global_step < self.max_steps:
            epoch_iterator = tqdm(
                self.train_dataloader, desc="Iteration", disable=self.is_top)
            for step, batch in enumerate(epoch_iterator):
                # Skip past any already trained steps if resuming training
                if (step // self.gradient_accumulation_steps) < self.current_step:
                    continue

                loss = self.run_batch(batch, train=True)

                if self.gradient_accumulation_steps > 1:
                    loss = loss / self.gradient_accumulation_steps

                if self.fp16:
                    with self.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                if (step + 1) % self.gradient_accumulation_steps == 0:
                    if self.max_grad_norm is not None:
                        if self.fp16:
                            torch.nn.utils.clip_grad_norm_(
                                self.amp.master_params(self.optimizer), self.max_grad_norm)
                        else:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), self.max_grad_norm)

                    self.optimizer.step()
                    self.scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    self.tb_writer.write(self.global_step)
                    self.global_step += 1

                # Log metrics
                if self.is_top and self.dev_dataloader is not None and self.global_step % self.logging_steps == 0:
                    # Only evaluate when single GPU otherwise metrics may not average well
                    if self.local_rank == -1:
                        self.evaluate()

                # Save model checkpoint
                if self.is_top and self.save_steps > 0 and self.global_step % self.save_steps == 0:
                    self.save()

                if self.global_step > self.max_steps:
                    epoch_iterator.close()
                    break

    def evaluate(self):
        epoch_iterator = tqdm(
            self.dev_dataloader, desc="Iteration (dev)", disable=self.is_top)
        self.tb_writer.clear()
        for step, batch in enumerate(epoch_iterator):
            self.run_batch(batch, train=False)
        self.tb_writer.write(self.global_step)

    def run_batch(self, batch, train: bool):
        if train:
            self.model.train()
        else:
            self.model.eval()

        batch = tuple(t.to(self.device) for t in batch)

        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "token_type_ids": batch[2],
            "class_labels": batch[3],
            "span_labels": batch[4],
            "p_mask": batch[6],
            "is_impossible": batch[7]
        }

        if self.model.model_type in ["xlm", "roberta", "distilbert", "camembert", "bart", "longformer"]:
            del inputs["token_type_ids"]

        if self.model.model_type in ["xlnet", "xlm"]:
            inputs.update({"cls_index": batch[5]})
            # FIXME: Add lang_id to dataset
            if hasattr(self.model, "config") and hasattr(self.model.config, "lang2id"):
                inputs.update(
                    {"langs": (torch.ones(batch[0].shape, dtype=torch.int64) * args.lang_id).to(self.device)}
                )

        outputs: IdentificationClassificationModelOutput = self.model(**inputs)
        # model outputs are always tuple in transformers (see doc)
        loss, loss_cls, loss_span = outputs.loss, outputs.loss_cls, outputs.loss_span

        if self.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
            loss_cls = loss_cls.mean()
            loss_span = loss_span.mean()

        prefix = 'train' if train else 'eval'
        self.tb_writer.add_scalar(f"{prefix}/lr", self.scheduler.get_lr()[0])
        self.tb_writer.add_scalar(f"{prefix}/loss", loss)
        self.tb_writer.add_scalar(f"{prefix}/loss_cls", loss_cls)
        self.tb_writer.add_scalar(f"{prefix}/loss_span", loss_span)

        return loss

    def save(self):
        checkpoint_dir = os.path.join(self.output_dir, f"checkpoint-{self.global_step}")
        self._save(checkpoint_dir)

    def _save(self, checkpoint_dir):
        # Take care of distributed/parallel training
        logger.info("Saving model checkpoint to %s", checkpoint_dir)
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        model_to_save.save_pretrained(checkpoint_dir)
        logger.info("Saving optimizer and scheduler states to %s", checkpoint_dir)
        torch.save(self.optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt"))
        torch.save(self.scheduler.state_dict(), os.path.join(checkpoint_dir, "scheduler.pt"))

        with open(os.path.join(checkpoint_dir, 'trainer_info.json'), 'w') as fout:
            json.dump({
                'global_step': self.global_step
            }, fout, indent=2)
        logger.info("Finished saving Trainer.")

    def resume(self, output_dir: str):
        checkpoint_dirs = glob.glob(os.path.join(output_dir, 'checkpoint-*'))
        checkpoint_dir = max(checkpoint_dirs, key=lambda d: int(d.split("-")[-1]))
        self._load(checkpoint_dir)

    def _load(self, checkpoint_dir):
        with open(os.path.join(checkpoint_dir, 'trainer_info.json')) as fin:
            trainer_info = json.load(fin)
        self.global_step = trainer_info['global_step']
        self.optimizer.load_state_dict(torch.load(os.path.join(checkpoint_dir, "optimizer.pt")))
        self.scheduler.load_state_dict(torch.load(os.path.join(checkpoint_dir, "scheduler.pt")))
        logger.info(f'Loaded Trainer from global_step of {self.global_step}')


def evaluate(args, model, tokenizer) -> Tuple[dict, List[IdentificationClassificationResult]]:
    dataset, examples, features = load_and_cache_examples(
        args, tokenizer, evaluate=True)

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    all_results = []
    start_time = timeit.default_timer()
    accu_loss, accu_loss_cls, accu_loss_span = 0.0, 0.0, 0.0
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "class_labels": batch[3],
                "span_labels": batch[4],
                "p_mask": batch[6],
                "is_impossible": batch[7]
            }
            if args.model_type in ["xlm", "roberta", "distilbert", "camembert", "bart", "longformer"]:
                del inputs["token_type_ids"]

            feature_indices = batch[8]

            # XLNet and XLM use more arguments for their predictions
            if args.model_type in ["xlnet", "xlm"]:
                inputs.update({"cls_index": batch[5]})
                # for lang_id-sensitive xlm models
                if hasattr(model, "config") and hasattr(model.config, "lang2id"):
                    inputs.update(
                        {"langs": (torch.ones(batch[0].shape, dtype=torch.int64) * args.lang_id).to(args.device)}
                    )
            outputs: IdentificationClassificationModelOutput = model(**inputs)

            loss, loss_cls, loss_span = outputs.loss, outputs.loss_cls, outputs.loss_span
            if args.n_gpu > 1:
                loss = loss.sum()  # mean() to average on multi-gpu parallel (not distributed) training
                loss_cls = loss_cls.sum()
                loss_span = loss_span.sum()

            accu_loss += loss.item() * len(batch[0])
            accu_loss_cls += loss_cls.item() * len(batch[0])
            accu_loss_span += loss_span.item() * len(batch[0])

        for i, feature_index in enumerate(feature_indices):
            eval_feature = features[feature_index.item()]
            unique_id = int(eval_feature.unique_id)

            class_logits = to_list(outputs.class_logits[i])
            span_logits = to_list(outputs.span_logits[i])
            result = IdentificationClassificationPartialResult(
                unique_id, class_logits, span_logits)

            all_results.append(result)

    evalTime = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))

    all_results = compute_predictions_logits(
        examples,
        features,
        all_results
    )

    # Compute the F1 and exact scores.
    results = evaluate_all(examples, all_results, [1, 3, 5, 8, 10, 15, 20, 30, 40, 50])
    results['loss'] = float(accu_loss / len(dataset))
    results['loss_cls'] = float(accu_loss_cls / len(dataset))
    results['loss_span'] = float(accu_loss_span / len(dataset))

    return results, all_results

