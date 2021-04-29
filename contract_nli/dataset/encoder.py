# Copyright 2020 The HuggingFace Team. and Hitachi America Ltd. All rights reserved.
# This file has been adopted from https://github.com/huggingface/transformers
# /blob/495c157d6fcfa29f2d9e1173582d2fb5a393c323/src/transformers/data/processors/squad.py
# and has been modified. See git log for the full details of changes.
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

from functools import partial
from multiprocessing import Pool, cpu_count
from typing import List, Dict
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm
from transformers.tokenization_utils_base import BatchEncoding, \
    PreTrainedTokenizerBase
from transformers.utils import logging

from contract_nli.dataset.loader import ContractNLIExample, NLILabel

# Store the tokenizers which insert 2 separators tokens
MULTI_SEP_TOKENS_TOKENIZERS_SET = {"roberta", "camembert", "bart", "mpnet"}
SPAN_TOKEN = '[SPAN]'


logger = logging.get_logger(__name__)


class IdentificationClassificationFeatures:
    """
    Single example features to be fed to a model. Those features are model-specific and can be crafted from
    :class:`~contract_nli.dataset.loader.ContractNLIExample` using the
    :method:`~contract_nli.dataset.encoder.convert_examples_to_features` method.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        cls_index: the index of the CLS token.
        p_mask: Mask identifying tokens that can be answers vs. tokens that cannot.
            Mask with 1 for tokens than cannot be in the answer and 0 for token that can be in an answer
        example_index: the index of the example
        unique_id: The unique Feature identifier
        paragraph_len: The length of the context
        token_is_max_context: List of booleans identifying which tokens have their maximum context in this feature object.
            If a token does not have their maximum context in this feature object, it means that another feature object
            has more information related to that token and should be prioritized over this feature for that token.
        tokens: list of tokens corresponding to the input ids
        token_to_orig_map: mapping between the tokens and the original text.
        span_to_orig_map: mapping between the spans and the original spans, needed in order to identify the answer.
        class_label:
        span_labels:
        valid_span_missing_in_context: Class label is NOT "not mentioned" and a valid span is not in the context
        data_id:
        encoding: optionally store the BatchEncoding with the fast-tokenizer alignement methods.
    """

    def __init__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        cls_index,
        p_mask,
        example_index,
        unique_id,
        paragraph_len,
        token_is_max_context,
        tokens,
        token_to_orig_map,
        span_to_orig_map,
        class_label,
        span_labels,
        valid_span_missing_in_context,
        data_id: str = None,
        encoding: BatchEncoding = None,
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.cls_index = cls_index
        self.p_mask = p_mask

        self.example_index = example_index
        self.unique_id = unique_id
        self.paragraph_len = paragraph_len
        self.token_is_max_context = token_is_max_context
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.span_to_orig_map: Dict[int, List[int]] = span_to_orig_map

        self.class_label = class_label
        self.span_labels = span_labels
        if valid_span_missing_in_context:
            assert class_label in [NLILabel.ENTAILMENT.value, NLILabel.CONTRADICTION.value]
        self.valid_span_missing_in_context = valid_span_missing_in_context
        self.data_id = data_id

        self.encoding = encoding


def _new_check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token at a position
    `position`.
    """
    # if len(doc_spans) == 1:
    # return True
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span["start"] + doc_span["paragraph_len"] - 1
        if position < doc_span["start"]:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span["start"]
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span["paragraph_len"]
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def tokenize(tokenizer, tokens: List[str], splits: List[int]):
    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    tok_to_orig_span_index = defaultdict(list)
    for i, s in enumerate(splits):
        tok_to_orig_span_index[s].append(i)
    span_to_orig_index = dict()
    for (i, token) in enumerate(tokens):
        if i in tok_to_orig_span_index:
            span_to_orig_index[len(all_doc_tokens)] = tok_to_orig_span_index[i]
            tok_to_orig_index.append(-1)
            all_doc_tokens.append(SPAN_TOKEN)
        orig_to_tok_index.append(len(all_doc_tokens))
        if tokenizer.__class__.__name__ in [
            "RobertaTokenizer",
            "LongformerTokenizer",
            "BartTokenizer",
            "RobertaTokenizerFast",
            "LongformerTokenizerFast",
            "BartTokenizerFast",
        ]:
            sub_tokens = tokenizer.tokenize(token, add_prefix_space=True)
        else:
            sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)
    return all_doc_tokens, orig_to_tok_index, tok_to_orig_index, span_to_orig_index


def convert_example_to_features(
        example: ContractNLIExample,
        max_seq_length: int,
        doc_stride: int,
        max_query_length: int,
        padding_strategy,
        labels_available: bool,
        symbol_based_hypothesis: bool
        ) -> List[IdentificationClassificationFeatures]:
    features = []

    all_doc_tokens, orig_to_tok_index, tok_to_orig_index, span_to_orig_index = tokenize(
        tokenizer, example.tokens, example.splits)

    if symbol_based_hypothesis:
        truncated_query = [example.hypothesis_symbol]
    else:
        truncated_query = tokenize(tokenizer, example.hypothesis_tokens, [])[0][:max_query_length]

    # Tokenizers who insert 2 SEP tokens in-between <context> & <question> need to have special handling
    # in the way they compute mask of added tokens.
    tokenizer_type = type(tokenizer).__name__.replace("Tokenizer", "").lower()
    sequence_added_tokens = (
        tokenizer.model_max_length - tokenizer.max_len_single_sentence + 1
        if tokenizer_type in MULTI_SEP_TOKENS_TOKENIZERS_SET
        else tokenizer.model_max_length - tokenizer.max_len_single_sentence
    )
    sequence_pair_added_tokens = tokenizer.model_max_length - tokenizer.max_len_sentences_pair
    query_with_special_tokens_length = len(truncated_query) + sequence_added_tokens
    max_context_length = max_seq_length - sequence_pair_added_tokens - len(truncated_query)

    spans = []
    start = 0
    covered_splits = set()
    all_splits = set(span_to_orig_index.keys())
    while len(all_splits - covered_splits) > 0:
        upcoming_splits = [i for i in span_to_orig_index.keys()
                           if i >= start and i not in covered_splits]
        assert len(upcoming_splits) > 0
        second_split = upcoming_splits[1] if len(upcoming_splits) > 1 else len(all_doc_tokens)
        if second_split - upcoming_splits[0] > max_context_length:
            # a single span is larger than maximum allowed tokens ---- there are nothing we can do
            start = upcoming_splits[0]
            last_span_idx = second_split
            covered_splits.add(upcoming_splits[0])
        elif second_split - start > max_context_length:
            # we can fit the first upcoming span if we modify "start"
            start += (second_split - max_context_length)
            last_span_idx = second_split
            covered_splits.add(upcoming_splits[0])
        else:
            # we can fit at least one span
            last_span_idx = None
            for i in range(start, min(start + max_context_length, len(all_doc_tokens)) + 1):
                if i == len(all_doc_tokens) or i in span_to_orig_index:
                    if last_span_idx is not None:
                        covered_splits.add(last_span_idx)
                    last_span_idx = i
            assert last_span_idx is not None

        split_tokens = all_doc_tokens[start:min(start + max_context_length, len(all_doc_tokens))]

        # Define the side we want to truncate / pad and the text/pair sorting
        if tokenizer.padding_side == "right":
            texts = truncated_query
            pairs = split_tokens
        else:
            texts = split_tokens
            pairs = truncated_query

        encoded_dict = tokenizer.encode_plus(
            texts,
            pairs,
            truncation=False,
            padding=padding_strategy,
            max_length=max_seq_length,
            return_overflowing_tokens=False,
            return_token_type_ids=True
        )
        assert len(encoded_dict['input_ids']) <= max_seq_length

        paragraph_len = len(split_tokens)
        if tokenizer.pad_token_id in encoded_dict["input_ids"]:
            if tokenizer.padding_side == "right":
                non_padded_ids = encoded_dict["input_ids"][: encoded_dict["input_ids"].index(tokenizer.pad_token_id)]
            else:
                last_padding_id_position = (
                    len(encoded_dict["input_ids"]) - 1 - encoded_dict["input_ids"][::-1].index(tokenizer.pad_token_id)
                )
                non_padded_ids = encoded_dict["input_ids"][last_padding_id_position + 1 :]
        else:
            non_padded_ids = encoded_dict["input_ids"]

        tokens = tokenizer.convert_ids_to_tokens(non_padded_ids)

        token_to_orig_map = {}
        span_to_orig_map = {}
        for i in range(paragraph_len):
            index = query_with_special_tokens_length + i if tokenizer.padding_side == "right" else i
            if tok_to_orig_index[start + i] != -1:
                token_to_orig_map[index] = tok_to_orig_index[start + i]
                assert (start + i) not in span_to_orig_index
            else:
                assert (start + i) in span_to_orig_index
                span_to_orig_map[index] = span_to_orig_index[start + i]

        encoded_dict["paragraph_len"] = paragraph_len
        encoded_dict["tokens"] = tokens
        encoded_dict["token_to_orig_map"] = token_to_orig_map
        encoded_dict["span_to_orig_map"] = span_to_orig_map
        encoded_dict["truncated_query_with_special_tokens_length"] = query_with_special_tokens_length
        encoded_dict["token_is_max_context"] = {}
        encoded_dict["start"] = start

        spans.append(encoded_dict)

        start = last_span_idx - doc_stride

    # Due to striding splitting, the same token will appear multiple times
    # in different splits. We annotate data with "token_is_max_context"
    # which classifies whether an instance of a token has the longest context
    # amongst different instances of the same token.
    for doc_span_index in range(len(spans)):
        for j in range(spans[doc_span_index]["paragraph_len"]):
            is_max_context = _new_check_is_max_context(
                spans, doc_span_index, spans[doc_span_index]['start'] + j)
            index = (
                j
                if tokenizer.padding_side == "left"
                else query_with_special_tokens_length + j
            )
            spans[doc_span_index]["token_is_max_context"][index] = is_max_context

    span_token_id = tokenizer.additional_special_tokens_ids[tokenizer.additional_special_tokens.index(SPAN_TOKEN)]
    for span in spans:
        # Identify the position of the CLS token
        cls_index = span["input_ids"].index(tokenizer.cls_token_id)

        # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
        p_mask = np.logical_not(
            np.isin(np.array(span["input_ids"]), [span_token_id, tokenizer.cls_token_id])
        ).astype(np.int32)

        valid_span_missing_in_context = False
        span_labels = np.zeros_like(span["input_ids"])
        if labels_available:
            if example.label != NLILabel.NOT_MENTIONED:
                doc_start = span["start"]
                doc_end = span["start"] + span["paragraph_len"]
                annotated_spans = set(example.annotated_spans)
                _span_labels = np.array([
                    any((s in annotated_spans for s in span_to_orig_index.get(i, [])))
                    for i in range(doc_start, doc_end)
                ]).astype(int)
                if not np.any(_span_labels):
                    valid_span_missing_in_context = True
                tok_start = query_with_special_tokens_length
                tok_end = tok_start + span["paragraph_len"]
                if tokenizer.padding_side == "right":
                    span_labels[tok_start:tok_end] = _span_labels
                else:
                    span_labels[-tok_end:-tok_start] = _span_labels
            class_label = example.label.value
        else:
            class_label = -1

        assert not np.any(np.logical_and(p_mask, span_labels))

        features.append(
            IdentificationClassificationFeatures(
                span["input_ids"],
                span["attention_mask"],
                span["token_type_ids"],
                cls_index,
                p_mask,
                example_index=0,  # Can not set unique_id and example_index here. They will be set after multiple processing.
                unique_id=0,
                paragraph_len=span["paragraph_len"],
                token_is_max_context=span["token_is_max_context"],
                tokens=span["tokens"],
                token_to_orig_map=span["token_to_orig_map"],
                span_to_orig_map=span["span_to_orig_map"],
                class_label=class_label,
                span_labels=span_labels,
                valid_span_missing_in_context=valid_span_missing_in_context,
                data_id=example.data_id,
            )
        )
    return features



def convert_example_to_features_init(tokenizer_for_convert: PreTrainedTokenizerBase):
    global tokenizer
    tokenizer = tokenizer_for_convert


def convert_examples_to_features(
    examples,
    tokenizer,
    max_seq_length,
    doc_stride,
    max_query_length,
    labels_available,
    symbol_based_hypothesis: bool,
    padding_strategy="max_length",
    threads=None,
    tqdm_enabled=True,
):
    """
    Converts a list of examples into a list of features that can be directly
    given as input to a model.

    Args:
        examples: list of :class:`~contract_nli.dataset.loader.ContractNLIExample`
        tokenizer: an instance of a child of :class:`~transformers.PreTrainedTokenizer`
        max_seq_length: The maximum sequence length of the inputs.
        doc_stride: The stride used when the context is too large and is split across several features.
        max_query_length: The maximum length of the query.
        labels_available: whether to create features for model evaluation or model training.
        padding_strategy: Default to "max_length". Which padding strategy to use
        threads: multiple processing threads.
    """
    if threads is None or threads < 0:
        threads = cpu_count()
    else:
        threads = min(threads, cpu_count())
    with Pool(threads, initializer=convert_example_to_features_init, initargs=(tokenizer,)) as p:
        annotate_ = partial(
            convert_example_to_features,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            padding_strategy=padding_strategy,
            labels_available=labels_available,
            symbol_based_hypothesis=symbol_based_hypothesis
        )
        features: List[List[IdentificationClassificationFeatures]] = list(
            tqdm(
                p.imap(annotate_, examples, chunksize=32),
                total=len(examples),
                desc="convert examples to features",
                disable=not tqdm_enabled,
            )
        )
    new_features = []
    unique_id = 1000000000
    example_index = 0
    for example_features in tqdm(
        features, total=len(features), desc="add example index and unique id", disable=not tqdm_enabled
    ):
        if not example_features:
            continue
        for example_feature in example_features:
            example_feature.example_index = example_index
            example_feature.unique_id = unique_id
            new_features.append(example_feature)
            unique_id += 1
        example_index += 1
    features: List[IdentificationClassificationFeatures] = new_features
    del new_features

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_masks = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
    all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
    all_valid_span_missing_in_context = torch.tensor([f.valid_span_missing_in_context for f in features], dtype=torch.float)

    all_feature_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    dataset = [
        all_input_ids,
        all_attention_masks,
        all_token_type_ids,
        all_cls_index,
        all_p_mask,
        all_valid_span_missing_in_context,
        all_feature_index
    ]
    if labels_available:
        all_class_label = torch.tensor(
            [f.class_label for f in features], dtype=torch.long)
        all_span_labels = torch.tensor(
            [f.span_labels for f in features], dtype=torch.long)
        dataset += [
            all_class_label,
            all_span_labels,
        ]
    dataset = TensorDataset(*dataset)
    return features, dataset
