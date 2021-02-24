import collections
from typing import List, Union

import numpy as np
from scipy.special import softmax

from contract_nli.dataset.encoder import IdentificationClassificationFeatures
from contract_nli.dataset.loader import ContractNLIExample, NLILabel


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


class ClassificationResult:
    def __init__(self, data_id, class_probs):
        self.class_probs = class_probs
        self.data_id = data_id


def compute_predictions_logits(
        all_examples: List[ContractNLIExample],
        all_features: List[IdentificationClassificationFeatures],
        all_results: List[IdentificationClassificationPartialResult],
        weight_class_probs_by_span_probs: bool
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
        ave_span_probs = []
        class_probs = []
        for feature in features:
            result = unique_id_to_result[feature.unique_id]
            _span_probs = softmax(np.array(result.span_logits), axis=1)
            for tok_idx, orig_span_indices in feature.span_to_orig_map.items():
                for orig_span_idx in orig_span_indices:
                    span_probs[orig_span_idx] += _span_probs[tok_idx]
                    num_pred_spans[orig_span_idx] += 1
            ave_span_probs.append(np.mean(_span_probs[:, 1]))
            class_probs.append(softmax(result.class_logits))
        assert np.all(num_pred_spans > 0)
        span_probs /= num_pred_spans[:, None]
        assert np.allclose(span_probs.sum(1), 1.0)
        if weight_class_probs_by_span_probs:
            ave_span_probs = np.array(ave_span_probs)
            weight = ave_span_probs / ave_span_probs.sum()
            class_probs = (np.array(class_probs) * weight[:, None]).sum(0)
        else:
            class_probs = np.array(class_probs).mean(0)
        assert abs(1.0 - class_probs.sum()) < 0.001
        results.append(IdentificationClassificationResult(
            data_id=example.data_id,
            span_probs=span_probs,
            class_probs=class_probs
        ))
    return results


def format_json(
        all_examples: List[ContractNLIExample],
        all_results: List[Union[IdentificationClassificationResult, ClassificationResult]]
        ) -> List[dict]:

    data_id_to_result = {}
    for result in all_results:
        data_id_to_result[result.data_id] = result

    documents = dict()
    for example_index, example in enumerate(all_examples):
        if example.document_id not in documents:
            documents[example.document_id] = {
                'id': example.document_id,
                'file_name': example.file_name,
                'text': example.context_text,
                'spans': example.spans,
                'annotation_sets': [{
                    'user': 'prediction',
                    'mturk': False,
                    'annotations': dict()
                }]
            }
        assert len(example.spans) == len(example.splits)
        if example.data_id not in data_id_to_result:
            assert isinstance(all_results[0], ClassificationResult)
            continue
        prediction = data_id_to_result[example.data_id]
        d = {
            'choice': NLILabel(
                np.argmax(prediction.class_probs)).to_anno_name(),
            'class_probs': {
                NLILabel(i).to_anno_name(): float(p)
                for i, p in enumerate(prediction.class_probs)
            },
        }
        if isinstance(prediction, IdentificationClassificationResult):
            d.update({
                'spans': np.where(prediction.span_probs[:, 1] > 0.5)[0].tolist(),
                'span_probs': prediction.span_probs[:, 1].tolist()
            })
        documents[example.document_id]['annotation_sets'][0]['annotations'][example.hypothesis_id] = d

    if isinstance(all_results[0], IdentificationClassificationResult):
        all_hypothesis_ids = [
            tuple(sorted(document['annotation_sets'][0]['annotations'].keys()))
            for document in documents.values()]
        assert len(set(all_hypothesis_ids)) == 1
    return sorted(documents.values(), key=lambda d: d['id'])
