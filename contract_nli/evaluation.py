from collections import defaultdict
from typing import Dict, List

import numpy as np
import sklearn.metrics

from contract_nli.dataset.loader import NLILabel, ContractNLIExample
from contract_nli.postprocess import IdentificationClassificationResult


def evaluate_predicted_spans(y_true, y_pred) -> Dict[str, float]:
    return {
        'precision': sklearn.metrics.precision_score(y_true, y_pred, zero_division=0),
        'recall': sklearn.metrics.recall_score(y_true, y_pred),
        'f1': sklearn.metrics.f1_score(y_true, y_pred),
        'accuracy': sklearn.metrics.accuracy_score(y_true, y_pred),
    }


def evaluate_spans(y_true, y_prob, ks) -> Dict[str, float]:
    assert y_prob.ndim == 1
    assert y_true.ndim == 1
    assert len(y_true) == len(y_prob)
    metrics = evaluate_predicted_spans(y_true, y_prob > 0.5)
    metrics.update({
        'roc_auc': sklearn.metrics.roc_auc_score(y_true, y_prob),
        'map': sklearn.metrics.average_precision_score(y_true, y_prob),
    })
    for k in ks:
        y_pred = np.zeros_like(y_prob)
        for j in np.argsort(y_prob)[::-1][:k]:
            y_pred[j] = 1
        assert y_pred.sum() == min(k, len(y_pred))
        metrics.update({
            f'{n}@{k}': v for n, v in evaluate_predicted_spans(y_true, y_pred).items()
        })
    return metrics


def evaluate_class(y_true, y_prob) -> Dict[str, float]:
    assert y_prob.ndim == 2 and y_prob.shape[1] == 3
    assert y_true.ndim == 1
    assert len(y_true) == len(y_prob)
    y_pred = np.argmax(y_prob, axis=1)
    metrics = {
        'accuracy': sklearn.metrics.accuracy_score(y_true, y_pred)
    }
    for label in (NLILabel.ENTAILMENT, NLILabel.CONTRADICTION):
        ln = label.name.lower()
        _y_true = y_true == label.value
        _y_pred = y_pred == label.value
        metrics.update({
            f'precision_{ln}': sklearn.metrics.precision_score(_y_true, _y_pred, zero_division=0),
            f'recall_{ln}': sklearn.metrics.recall_score(_y_true, _y_pred),
            f'f1_{ln}': sklearn.metrics.f1_score(_y_true, _y_pred),
        })
    return metrics


def _macro_average(dicts: List[Dict[str, float]]):
    ret = dict()
    for k in dicts[0].keys():
        ret[k] = sum((d[k] for d in dicts)) / float(len(dicts))
    return ret


def evaluate_all(
        examples: List[ContractNLIExample],
        results: List[IdentificationClassificationResult],
        ks: List[int]
        ) -> dict:
    id_to_result = {r.data_id: r for r in results}
    span_probs = defaultdict(list)
    span_labels = defaultdict(list)
    class_probs = defaultdict(list)
    class_labels = defaultdict(list)
    for example in examples:
        label_id = example.data_id.split('_')[1]
        result: IdentificationClassificationResult = id_to_result[example.data_id]
        class_labels[label_id].append(example.label.value)
        class_probs[label_id].append(result.class_probs)
        # FIXME: this calculates precision optimistically
        if example.label != NLILabel.NOT_MENTIONED:
            span_label = np.zeros(len(example.splits))
            for s in example.annotated_spans:
                span_label[s] = 1
            span_labels[label_id].append(span_label)
            span_probs[label_id].append(result.span_probs[:, 1])
    label_ids = sorted(span_labels.keys())
    return {
        'micro_label_micro_doc': {
            'class': evaluate_class(
                np.concatenate([class_labels[k] for k in label_ids]),
                np.vstack([np.stack(class_probs[k]) for k in label_ids])
            ),
            'span': evaluate_spans(
                np.concatenate([l for k in label_ids for l in span_labels[k]]),
                np.concatenate([l for k in label_ids for l in span_probs[k]]),
                ks
            )
        },
        'macro_label_micro_doc': {
            'class': _macro_average([
                evaluate_class(np.array(class_labels[k]), np.stack(class_probs[k]))
                for k in label_ids
            ]),
            'span': _macro_average([
                evaluate_spans(
                    np.concatenate(span_labels[k]),
                    np.concatenate(span_probs[k]),
                    ks)
                for k in label_ids
            ])
        },
        'macro_label_macro_doc': {
            'span': _macro_average([
                _macro_average([
                    evaluate_spans(span_labels[k][i], span_probs[k][i], ks)
                    for i in range(len(span_labels[k]))
                ])
                for k in label_ids
            ])
        },
        'label_wise': {
            k: {
                'micro_doc': {
                    'class': evaluate_class(np.array(class_labels[k]), np.stack(class_probs[k])),
                    'span': evaluate_spans(
                        np.concatenate(span_labels[k]),
                        np.concatenate(span_probs[k]),
                        ks)
                },
                'macro_doc': {
                    'span': _macro_average([
                        evaluate_spans(span_labels[k][i], span_probs[k][i], ks)
                        for i in range(len(span_labels[k]))
                    ])
                }
            }
            for k in label_ids
        }
    }

