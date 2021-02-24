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
from typing import Tuple, List, Optional, Union

import torch
from torch.utils.data import TensorDataset

from contract_nli.dataset.encoder import convert_examples_to_features, \
    IdentificationClassificationFeatures
from contract_nli.dataset.encoder_classification import convert_examples_to_features as convert_examples_to_classification_features
from contract_nli.dataset.encoder_classification import ClassificationFeatures
from contract_nli.dataset.loader import ContractNLIExample

logger = logging.getLogger(__name__)


def load_and_cache_examples(
        path: str, *, local_rank: int = 1, overwrite_cache = False,
        cache_dir: str = '.') -> List[ContractNLIExample]:
    try:
        os.makedirs(cache_dir)
    except OSError:
        pass
    filename = os.path.splitext(os.path.basename(path))[0]
    cachename = f'cached_examples_{filename}'
    cached_examples_file = os.path.join(cache_dir, cachename)

    # Init features and dataset from cache if it exists
    if os.path.exists(cached_examples_file) and not overwrite_cache:
        logger.info("Loading examples from cached file %s", cached_examples_file)
        features_and_dataset = torch.load(cached_examples_file)
        examples = features_and_dataset["examples"]
    else:
        assert local_rank in [-1, 0]
        logger.info(f"Creating examples from dataset file at {path}")
        with open(path) as fin:
            input_dict = json.load(fin)
        examples = ContractNLIExample.load(input_dict)


        logger.info("Saving examples into cached file %s", cached_examples_file)
        torch.save({"examples": examples}, cached_examples_file)

    return examples


def load_and_cache_features(
        path: str, examples: List[ContractNLIExample], tokenizer, *,
        max_seq_length: int, doc_stride: int, max_query_length: int,
        dataset_type: str, symbol_based_hypothesis: bool,
        threads: Optional[int] = 1, local_rank: int = 1,
        overwrite_cache = False, labels_available=True, cache_dir: str = '.'
        ) -> Tuple[TensorDataset, List[Union[IdentificationClassificationFeatures, ClassificationFeatures]]]:
    try:
        os.makedirs(cache_dir)
    except OSError:
        pass
    filename = os.path.splitext(os.path.basename(path))[0]
    tokenizer_name = os.path.splitext(os.path.split(tokenizer.name_or_path)[-1])[0]
    cachename = f'cached_features_{filename}_{dataset_type}_{tokenizer_name}_{max_seq_length}_{max_query_length}_{doc_stride}'
    if not labels_available:
        cachename += '_nolabels'
    cached_features_file = os.path.join(cache_dir, cachename)

    # Init features and dataset from cache if it exists
    if os.path.exists(cached_features_file) and not overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features_and_dataset = torch.load(cached_features_file)
        features, dataset = (
            features_and_dataset["features"],
            features_and_dataset["dataset"]
        )
    else:
        assert local_rank in [-1, 0]
        logger.info(f"Creating features from dataset file at {path}")
        if dataset_type == 'identification_classification':
            features, dataset = convert_examples_to_features(
                examples=examples,
                tokenizer=tokenizer,
                max_seq_length=max_seq_length,
                doc_stride=doc_stride,
                max_query_length=max_query_length,
                labels_available=labels_available,
                symbol_based_hypothesis=symbol_based_hypothesis,
                threads=threads
            )
        elif dataset_type == 'classification':
            features, dataset = convert_examples_to_classification_features(
                examples=examples,
                tokenizer=tokenizer,
                max_seq_length=max_seq_length,
                max_query_length=max_query_length,
                symbol_based_hypothesis=symbol_based_hypothesis,
                threads=threads
            )
        else:
            assert not "dataset_type must be either 'classification' or 'identification_classification'"

        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save({"features": features, "dataset": dataset}, cached_features_file)

    return dataset, features
