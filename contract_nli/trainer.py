# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.
# Copyright (c) 2021, Hitachi America Ltd. All rights reserved.
# This file has been adopted from https://github.com/huggingface/transformers
# /blob/0c9bae09340dd8c6fdf6aa2ea5637e956efe0f7c/examples/question-answering/run_squad.py
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
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
import sklearn.metrics
import scipy.special

from contract_nli.batch_converter import classification_converter, identification_classification_converter
from contract_nli.summary_writer import SummaryWriter

logger = logging.getLogger(__name__)


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
            self, *, model, train_dataset, optimizer, task: str, output_dir: str,
            per_gpu_train_batch_size: int, num_epochs: Optional[int] = None, max_steps: Optional[int] = None,
            dev_dataset=None, valid_steps: Optional[int]=None, per_gpu_dev_batch_size: Optional[int]=None,
            gradient_accumulation_steps: int=1, warmup_steps: int=0, max_grad_norm: Optional[float]=None,
            n_gpu: int=1, local_rank: int=-1, fp16: bool=False, fp16_opt_level=None, device=torch.device("cpu"),
            save_steps: Optional[int] = None):
        if local_rank in [-1, 0]:
            self.tb_writer = SummaryWriter(os.path.join(output_dir, 'tensorboard'))
        if task not in ['identification_classification', 'classification']:
            raise ValueError("task must be either 'classification' or 'identification_classification'")

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
        self.valid_steps = valid_steps
        self.max_grad_norm = max_grad_norm
        self.per_gpu_train_batch_size = per_gpu_train_batch_size
        self.output_dir = output_dir
        self.save_steps = save_steps
        self.task = task
        if self.task == 'identification_classification':
            self.converter = identification_classification_converter
        else:
            self.converter = classification_converter

        self.global_step = 0
        self.best_loss = np.inf

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
        self.model.to(self.device)

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
    def n_steps_per_epoch(self):
        return len(self.train_dataloader) // self.gradient_accumulation_steps

    @property
    def current_epoch(self):
        return self.global_step // self.n_steps_per_epoch

    @property
    def current_step(self):
        return self.global_step % self.n_steps_per_epoch

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
        pbar = tqdm(
            total=int(self.max_steps), initial=self.global_step,
            desc=f"Train (epoch {self.current_epoch + 1})", disable=not self.is_top
        )
        step = 0
        while (self.global_step + 1) <= self.max_steps:
            if self.local_rank != -1:
                self.train_dataloader.sampler.set_epoch(self.current_epoch)
            pbar.set_description(desc=f"Train (epoch {self.current_epoch + 1})")
            for batch in self.train_dataloader:
                # Skip past any already trained steps if resuming training
                if (step // self.gradient_accumulation_steps) < self.current_step:
                    step += 1
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
                    if self.is_top:
                        self.tb_writer.write(self.global_step)
                    self.global_step += 1
                    pbar.update()

                    if self.dev_dataloader is not None and self.global_step % self.valid_steps == 0:
                        loss = self.evaluate()
                        if loss < self.best_loss:
                            self.best_loss = loss
                            if self.is_top:
                                self.save(self.best_checkpoint_dir)

                    if self.is_top and self.save_steps > 0 and self.global_step % self.save_steps == 0:
                        self.save()
                step += 1
                if (self.global_step + 1) >= self.max_steps:
                    break
        pbar.close()

    def evaluate(self):
        epoch_iterator = tqdm(
            self.dev_dataloader, desc="Iteration (dev)", disable=not self.is_top)
        if self.is_top:
            self.tb_writer.clear()
        losses = []
        for _, batch in enumerate(epoch_iterator):
            loss = self.run_batch(batch, train=False)
            losses.append(loss.item())
        if self.is_top:
            self.tb_writer.write(self.global_step)
        return np.mean(losses)

    def run_batch(self, batch, train: bool):
        if train:
            self.model.train()
        else:
            self.model.eval()
        inputs = self.converter(batch, self.model, self.device)
        outputs = self.model(**inputs)

        loss, loss_cls = outputs.loss, outputs.loss_cls,
        if self.task == 'identification_classification':
            loss_span = outputs.loss_span

        if self.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
            loss_cls = loss_cls.mean()
            if self.task == 'identification_classification':
                loss_span = loss_span.mean()

        if self.is_top:
            prefix = 'train' if train else 'eval'
            self.tb_writer.add_scalar(f"{prefix}/lr", self.scheduler.get_last_lr()[0])
            self.tb_writer.add_scalar(f"{prefix}/loss", loss.item())
            self.tb_writer.add_scalar(f"{prefix}/loss_cls", loss_cls.item())
            self.tb_writer.add_scalar(
                f'{prefix}/accuracy_nli',
                (np.argmax(outputs.class_logits.detach().cpu().numpy(), axis=1) == inputs['class_labels'].cpu().numpy()).mean())
            if self.task == 'identification_classification':
                self.tb_writer.add_scalar(f"{prefix}/loss_span", loss_span.item())
                mask = inputs['p_mask'].cpu().numpy()
                probs = scipy.special.softmax(outputs.span_logits.detach().cpu().numpy(), axis=2)[:, :, 1]
                labels = inputs['span_labels'].cpu().numpy().copy()
                labels[:, 0] = 1
                if len(set(labels.flat[mask.flat == 0])) > 1:
                    self.tb_writer.add_scalar(
                        f'{prefix}/map_span',
                        sklearn.metrics.average_precision_score(
                            labels.flat[mask.flat == 0],
                            probs.flat[mask.flat == 0]))

        return loss

    @property
    def best_checkpoint_dir(self) -> str:
        return os.path.join(self.output_dir, f"best-checkpoint")

    def save(self, checkpoint_dir: Optional[str] = None):
        if checkpoint_dir is None:
            checkpoint_dir = os.path.join(
                self.output_dir, f"checkpoint-{self.global_step}")
        # Take care of distributed/parallel training
        logger.info("Saving model checkpoint to %s", checkpoint_dir)
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        model_to_save.save_pretrained(checkpoint_dir)
        logger.info("Saving optimizer and scheduler states to %s", checkpoint_dir)
        torch.save(self.optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt"))
        torch.save(self.scheduler.state_dict(), os.path.join(checkpoint_dir, "scheduler.pt"))

        with open(os.path.join(checkpoint_dir, 'trainer_info.json'), 'w') as fout:
            json.dump({
                'global_step': self.global_step,
                'best_loss': self.best_loss,
                'task': self.task
            }, fout, indent=2)
        logger.info("Finished saving Trainer.")

    def resume(self, output_dir: str):
        checkpoint_dirs = glob.glob(os.path.join(output_dir, 'checkpoint-*'))
        checkpoint_dir = max(checkpoint_dirs, key=lambda d: int(d.split("-")[-1]))
        self.load(checkpoint_dir)

    def load(self, checkpoint_dir):
        with open(os.path.join(checkpoint_dir, 'trainer_info.json')) as fin:
            trainer_info = json.load(fin)
        self.global_step = trainer_info['global_step']
        if not os.path.exists(self.best_checkpoint_dir) and trainer_info['best_loss'] != np.inf:
            logger.warning(
                f'Previous "best_loss" was {trainer_info["best_loss"]} but '
                'the corresponding checkpoint was not found at '
                f'{self.best_checkpoint_dir}. Resetting it to inf.')
            self.best_loss = np.inf
        else:
            self.best_loss = trainer_info['best_loss']
        self.task = trainer_info['task']
        if self.task == 'identification_classification':
            self.converter = identification_classification_converter
        else:
            self.converter = classification_converter

        torch.cuda.empty_cache()
        self.optimizer.load_state_dict(torch.load(os.path.join(checkpoint_dir, "optimizer.pt")))
        self.scheduler.load_state_dict(torch.load(os.path.join(checkpoint_dir, "scheduler.pt")))

        model_cls = type(self.model.module if hasattr(self.model, "module") else self.model)
        self.model.to('cpu')
        del self.model
        torch.cuda.empty_cache()
        self.model = model_cls.from_pretrained(checkpoint_dir)

        self.deployed = False

        logger.info(f'Loaded Trainer from {checkpoint_dir}')
