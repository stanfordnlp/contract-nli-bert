import collections
from typing import List

import numpy as np
from scipy.special import softmax

from contract_nli.dataset.encoder import IdentificationClassificationFeatures
from contract_nli.dataset.loader import ContractNLIExample


class IdentificationClassificationPartialResult:
    def __init__(self, unique_id, class_logits, span_logits):
        self.class_logits = class_logits
        self.span_logits = span_logits
        self.unique_id = unique_id


class IdentificationClassificationResult:
    def __init__(self, data_id, class_probs, span_probs):
        self.class_probs = class_probs
        self.span_probs = span_probs
        self.data_id = data_id


def compute_predictions_logits(
        all_examples: List[ContractNLIExample],
        all_features: List[IdentificationClassificationFeatures],
        all_results: List[IdentificationClassificationPartialResult]
        ) -> List[IdentificationClassificationResult]:
    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    results = []
    for example_index, example in enumerate(all_examples):
        features: List[IdentificationClassificationFeatures] = example_index_to_features[example_index]
        assert len(features) > 0
        span_probs = np.zeros((len(example.splits), 2))
        num_pred_spans = np.zeros(len(example.splits))
        class_probs = np.zeros(3)
        for feature in features:
            result = unique_id_to_result[feature.unique_id]
            _span_probs = softmax(np.array(result.span_logits), axis=1)
            for tok_idx, orig_span_idx in feature.span_to_orig_map.items():
                span_probs[orig_span_idx] += _span_probs[tok_idx]
                num_pred_spans[orig_span_idx] += 1
            class_probs += softmax(result.class_logits)
        assert np.all(num_pred_spans > 0)
        span_probs /= num_pred_spans[:, None]
        assert np.allclose(span_probs.sum(1), 1.0)
        class_probs /= len(features)
        assert abs(1.0 - class_probs.sum()) < 0.001
        results.append(IdentificationClassificationResult(
            data_id=example.data_id,
            span_probs=span_probs,
            class_probs=class_probs
        ))
    return results
