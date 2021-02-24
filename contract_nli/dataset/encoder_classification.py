from functools import partial
from multiprocessing import Pool, cpu_count
from typing import List

import numpy as np
import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import logging

from contract_nli.dataset.encoder import tokenize, SPAN_TOKEN
from contract_nli.dataset.loader import ContractNLIExample, NLILabel

logger = logging.get_logger(__name__)


class ClassificationFeatures:
    """
    Single example features to be fed to a model. Those features are model-specific and can be crafted from
    :class:`~contract_nli.dataset.loader.ContractNLIExample` using the
    :method:`~contract_nli.dataset.encoder_classification.convert_examples_to_features` method.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        cls_index: the index of the CLS token.
        p_mask: Mask identifying tokens that can be answers vs. tokens that cannot.
            Mask with 1 for tokens than cannot be in the answer and 0 for token that can be in an answer
        example_index: the index of the example
        unique_id: The unique Feature identifier
        tokens: list of tokens corresponding to the input ids
        class_label:
        data_id:
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
        tokens,
        class_label,
        data_id: str = None
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.cls_index = cls_index
        self.p_mask = p_mask

        self.example_index = example_index
        self.unique_id = unique_id
        self.tokens = tokens

        self.class_label = class_label
        self.data_id = data_id


def convert_example_to_features(
        example: ContractNLIExample,
        max_seq_length: int,
        max_query_length: int,
        padding_strategy,
        symbol_based_hypothesis: bool
        ) -> ClassificationFeatures:
    all_doc_tokens, orig_to_tok_index, tok_to_orig_index, span_to_orig_index = tokenize(
       tokenizer, example.tokens, example.splits)

    relevant_tokens = []
    for s in example.annotated_spans:
        start_token_index = orig_to_tok_index[example.splits[s]]
        if s + 1 < len(example.splits):
            end_token_index = orig_to_tok_index[example.splits[s + 1]] - 1
        else:
            end_token_index = orig_to_tok_index[-1]
        relevant_tokens.extend(all_doc_tokens[start_token_index:end_token_index])
    assert len(relevant_tokens) > 0 and SPAN_TOKEN not in relevant_tokens

    if symbol_based_hypothesis:
        truncated_query = [example.hypothesis_symbol]
    else:
        truncated_query = tokenize(
            tokenizer, example.hypothesis_tokens, [])[0][:max_query_length]

    # Define the side we want to truncate / pad and the text/pair sorting
    if tokenizer.padding_side == "right":
        texts = truncated_query
        pairs = relevant_tokens
        truncation = 'only_second'
    else:
        texts = relevant_tokens
        pairs = truncated_query
        truncation = 'only_first'

    encoded_dict = tokenizer.encode_plus(
        texts,
        pairs,
        truncation=truncation,
        padding=padding_strategy,
        max_length=max_seq_length,
        return_overflowing_tokens=False,
        return_token_type_ids=True
    )
    assert len(encoded_dict['input_ids']) <= max_seq_length

    if tokenizer.pad_token_id in encoded_dict["input_ids"]:
        if tokenizer.padding_side == "right":
            non_padded_ids = encoded_dict["input_ids"][:encoded_dict["input_ids"].index(tokenizer.pad_token_id)]
        else:
            last_padding_id_position = (
                len(encoded_dict["input_ids"]) - 1 - encoded_dict["input_ids"][::-1].index(tokenizer.pad_token_id)
            )
            non_padded_ids = encoded_dict["input_ids"][last_padding_id_position + 1:]
    else:
        non_padded_ids = encoded_dict["input_ids"]

    tokens = tokenizer.convert_ids_to_tokens(non_padded_ids)

    # Identify the position of the CLS token
    cls_index = encoded_dict["input_ids"].index(tokenizer.cls_token_id)

    # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
    p_mask = (np.array(encoded_dict["input_ids"]) != tokenizer.cls_token_id).astype(np.int32)

    return ClassificationFeatures(
        encoded_dict["input_ids"],
        encoded_dict["attention_mask"],
        encoded_dict["token_type_ids"],
        cls_index,
        p_mask,
        example_index=0,  # Can not set unique_id and example_index here. They will be set after multiple processing.
        unique_id=0,
        tokens=tokens,
        class_label=example.label.value,
        data_id=example.data_id,
    )


def convert_example_to_features_init(tokenizer_for_convert: PreTrainedTokenizerBase):
    global tokenizer
    tokenizer = tokenizer_for_convert


def convert_examples_to_features(
    examples: List[ContractNLIExample],
    tokenizer,
    max_seq_length,
    max_query_length,
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
    n_orig_examples = len(examples)
    examples = [e for e in examples if e.label != NLILabel.NOT_MENTIONED]
    logger.warning(
        f'Removed examples with "na" labels ({n_orig_examples} -> {len(examples)})')
    with Pool(threads, initializer=convert_example_to_features_init, initargs=(tokenizer,)) as p:
        annotate_ = partial(
            convert_example_to_features,
            max_seq_length=max_seq_length,
            max_query_length=max_query_length,
            padding_strategy=padding_strategy,
            symbol_based_hypothesis=symbol_based_hypothesis
        )
        features: List[ClassificationFeatures] = list(
            tqdm(
                p.imap(annotate_, examples, chunksize=32),
                total=len(examples),
                desc="convert examples to features",
                disable=not tqdm_enabled,
            )
        )
    new_features = []
    for example_index, example_features in enumerate(features):
        example_features.example_index = example_index
        example_features.unique_id = example_index + 1000000000
        new_features.append(example_features)
    features: List[ClassificationFeatures] = new_features
    del new_features
    assert len(features) == len(examples)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_masks = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
    all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)

    all_feature_index = torch.arange(all_input_ids.size(0), dtype=torch.long)

    all_class_label = torch.tensor(
        [f.class_label for f in features], dtype=torch.long)

    dataset = [
        all_input_ids,
        all_attention_masks,
        all_token_type_ids,
        all_cls_index,
        all_p_mask,
        all_feature_index,
        all_class_label
    ]
    dataset = TensorDataset(*dataset)
    return features, dataset
