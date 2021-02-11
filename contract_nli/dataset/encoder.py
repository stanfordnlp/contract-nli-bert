# Copyright 2020 The HuggingFace Team. All rights reserved.
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
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm
from transformers.tokenization_utils_base import BatchEncoding, \
    PreTrainedTokenizerBase
from transformers.utils import logging

from contract_nli.dataset.loader import ContractNLIExample

# Store the tokenizers which insert 2 separators tokens
MULTI_SEP_TOKENS_TOKENIZERS_SET = {"roberta", "camembert", "bart", "mpnet"}


logger = logging.get_logger(__name__)


class IdentificationClassificationFeatures:
    """
    Single squad example features to be fed to a model. Those features are model-specific and can be crafted from
    :class:`~transformers.data.processors.squad.SquadExample` using the
    :method:`~transformers.data.processors.squad.squad_convert_examples_to_features` method.

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
        token_to_orig_map: mapping between the tokens and the original text, needed in order to identify the answer.

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
        is_impossible,
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
        self.span_to_orig_map = span_to_orig_map

        self.class_label = class_label
        self.span_labels = span_labels
        self.is_impossible = is_impossible
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


def _tokenize(tokens: List[str], splits: List[int]):
    tok_to_orig_index = []
    orig_to_tok_index = []
    span_to_orig_index = []
    all_doc_tokens = []
    splits = {s: i for i, s in enumerate(splits)}
    for (i, token) in enumerate(tokens):
        if i in splits:
            span_to_orig_index.append(splits[i])
            tok_to_orig_index.append(-1)
            # FIXME: do NOT use cls_token but add an additional_special_token
            all_doc_tokens.append(tokenizer.cls_token)
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
            span_to_orig_index.append(-1)
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)
    return all_doc_tokens, orig_to_tok_index, tok_to_orig_index, span_to_orig_index


def squad_convert_example_to_features(
        example: ContractNLIExample,
        max_seq_length: int,
        doc_stride: int,
        max_query_length: int,
        padding_strategy,
        is_training: bool
        ) -> List[IdentificationClassificationFeatures]:
    features = []

    all_doc_tokens, orig_to_tok_index, tok_to_orig_index, span_to_orig_index = _tokenize(
        example.tokens, example.splits)
    all_splits = {i for i, s in enumerate(span_to_orig_index) if s >= 0}

    truncated_query = _tokenize(example.hypothesis_tokens, [])[0][:max_query_length]

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
    while len(all_splits - covered_splits) > 0:
        upcoming_splits = [i + start for i, s in enumerate(span_to_orig_index[start:])
                           if s != -1 and s not in covered_splits]
        if upcoming_splits[1] - upcoming_splits[0] > max_context_length:
            # a single span is larger than maximum allowed tokens ---- there are nothing we can do
            start = upcoming_splits[0]
            last_span_idx = upcoming_splits[1]
            covered_splits.add(upcoming_splits[0])
        elif upcoming_splits[1] - start > max_context_length:
            # we can fit the first upcoming span if we modify "start"
            start += (upcoming_splits[1] - max_context_length)
            last_span_idx = upcoming_splits[1]
            covered_splits.add(upcoming_splits[0])
        else:
            # we can fit at least one span
            last_span_idx = None
            for i in range(start, min(start + max_context_length, len(all_doc_tokens)) + 1):
                if i == len(all_doc_tokens) or span_to_orig_index[i] != -1:
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
                assert span_to_orig_index[start + i] == -1
            else:
                assert span_to_orig_index[start + i] != -1
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

    for span in spans:
        # Identify the position of the CLS token
        cls_index = span["input_ids"].index(tokenizer.cls_token_id)

        # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
        # Original TF implem also keep the classification token (set to 0)
        p_mask = (np.array(span["input_ids"]) != tokenizer.cls_token_id).astype(np.int32)

        span_is_impossible = example.is_impossible
        span_labels = np.zeros_like(span["input_ids"])
        if is_training:
            if not span_is_impossible:
                doc_start = span["start"]
                doc_end = span["start"] + span["paragraph_len"]
                _span_labels = np.isin(
                    np.array(span_to_orig_index[doc_start:doc_end]),
                    example.annotated_spans).astype(int)
                if not np.any(_span_labels):
                    span_is_impossible = True
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
                is_impossible=span_is_impossible,
                data_id=example.data_id,
            )
        )
    return features



def squad_convert_example_to_features_init(tokenizer_for_convert: PreTrainedTokenizerBase):
    global tokenizer
    tokenizer = tokenizer_for_convert


def squad_convert_examples_to_features(
    examples,
    tokenizer,
    max_seq_length,
    doc_stride,
    max_query_length,
    is_training,
    padding_strategy="max_length",
    threads=1,
    tqdm_enabled=True,
):
    """
    Converts a list of examples into a list of features that can be directly given as input to a model. It is
    model-dependant and takes advantage of many of the tokenizer's features to create the model's inputs.

    Args:
        examples: list of :class:`~transformers.data.processors.squad.SquadExample`
        tokenizer: an instance of a child of :class:`~transformers.PreTrainedTokenizer`
        max_seq_length: The maximum sequence length of the inputs.
        doc_stride: The stride used when the context is too large and is split across several features.
        max_query_length: The maximum length of the query.
        is_training: whether to create features for model evaluation or model training.
        padding_strategy: Default to "max_length". Which padding strategy to use
        threads: multiple processing threads.


    Returns:
        list of :class:`~transformers.data.processors.squad.SquadFeatures`

    Example::

        processor = SquadV2Processor()
        examples = processor.get_dev_examples(data_dir)

        features = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=not evaluate,
        )
    """
    threads = min(threads, cpu_count())
    with Pool(threads, initializer=squad_convert_example_to_features_init, initargs=(tokenizer,)) as p:
        annotate_ = partial(
            squad_convert_example_to_features,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            padding_strategy=padding_strategy,
            is_training=is_training,
        )
        features: List[IdentificationClassificationFeatures] = list(
            tqdm(
                p.imap(annotate_, examples, chunksize=32),
                total=len(examples),
                desc="convert squad examples to features",
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
    all_is_impossible = torch.tensor([f.is_impossible for f in features], dtype=torch.float)

    all_feature_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    if not is_training:
        dataset = TensorDataset(
            all_input_ids, all_attention_masks, all_token_type_ids, all_cls_index, all_p_mask, all_feature_index
        )
    else:
        all_class_label = torch.tensor(
            [f.class_label for f in features], dtype=torch.long)
        all_span_labels = torch.tensor(
            [f.span_labels for f in features], dtype=torch.long)
        dataset = TensorDataset(
            all_input_ids,
            all_attention_masks,
            all_token_type_ids,
            all_class_label,
            all_span_labels,
            all_cls_index,
            all_p_mask,
            all_is_impossible,
            all_feature_index
        )

    return features, dataset
