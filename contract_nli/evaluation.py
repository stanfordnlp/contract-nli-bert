from collections import defaultdict
from typing import Dict, List, Union

import numpy as np
import sklearn.metrics
from scipy.stats import hmean

from contract_nli.dataset.loader import NLILabel, ContractNLIExample
from contract_nli.postprocess import IdentificationClassificationResult, ClassificationResult


def evaluate_predicted_spans(y_true, y_pred) -> Dict[str, float]:
    if y_true.sum() == 0:
        # do not use zero_division=np.nan because it cannot distinguish
        # zero divisions from y_true and y_pred in f1_score
        recall = np.nan
        f1 = np.nan
    else:
        recall = sklearn.metrics.recall_score(y_true, y_pred)
        f1 = sklearn.metrics.f1_score(y_true, y_pred)

    return {
        'precision': sklearn.metrics.precision_score(y_true, y_pred, zero_division=0),
        'recall': recall,
        'f1': f1,
        'accuracy': sklearn.metrics.accuracy_score(y_true, y_pred),
    }


def evaluate_spans(y_true, y_prob) -> Dict[str, float]:
    assert y_prob.ndim == 1
    assert y_true.ndim == 1
    assert len(y_true) == len(y_prob)
    metrics = evaluate_predicted_spans(y_true, y_prob > 0.5)
    metrics.update({
        'roc_auc': sklearn.metrics.roc_auc_score(y_true, y_prob),
        'map': sklearn.metrics.average_precision_score(y_true, y_prob),
    })
    return metrics


def predict_at_k(y_prob, k):
    y_pred = np.zeros_like(y_prob)
    for j in np.argsort(y_prob)[::-1][:k]:
        y_pred[j] = 1
    assert y_pred.sum() == min(k, len(y_pred))
    return y_pred


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
        if _y_true.sum() == 0:
            # do not use zero_division=np.nan because it cannot distinguish
            # zero divisions from y_true and y_pred in f1_score
            recall = np.nan
            f1 = np.nan
        else:
            recall = sklearn.metrics.recall_score(_y_true, _y_pred)
            f1 = sklearn.metrics.f1_score(_y_true, _y_pred)
        metrics.update({
            f'precision_{ln}': sklearn.metrics.precision_score(_y_true, _y_pred, zero_division=0),
            f'recall_{ln}': recall,
            f'f1_{ln}': f1,
        })
    for m in ('precision', 'recall', 'f1'):
        m_e = metrics[f'{m}_{NLILabel.ENTAILMENT.name.lower()}']
        m_c = metrics[f'{m}_{NLILabel.CONTRADICTION.name.lower()}']
        if np.isnan(m_e) or np.isnan(m_c):
            metrics[f'{m}_mean'] = np.nan
            metrics[f'{m}_hmean'] = np.nan
        else:
            metrics[f'{m}_mean'] = np.mean((m_e, m_c))
            metrics[f'{m}_hmean'] = hmean((m_e, m_c))

    return metrics


def _macro_average(dicts: List[Dict[str, float]]):
    ret = dict()
    for k in dicts[0].keys():
        vals = [d[k] for d in dicts if not np.isnan(d[k])]
        ret[k] = sum(vals) / float(len(vals))
    return ret


def remove_not_mentioned(y_pred):
    assert y_pred.shape[1] == 3
    y_bin = y_pred[:, [NLILabel.CONTRADICTION.value, NLILabel.ENTAILMENT.value]]
    y_bin = np.where(np.tile(np.sum(y_bin, axis=1, keepdims=True), [1, 2]) == 0,
                     0.5,
                     y_bin / np.sum(y_bin, axis=1, keepdims=True))
    y_pred = np.zeros((len(y_pred), 3), dtype=y_pred.dtype)
    y_pred[:, [NLILabel.CONTRADICTION.value, NLILabel.ENTAILMENT.value]] = y_bin
    return y_pred


def evaluate_all(
        examples: List[ContractNLIExample],
        results: List[Union[IdentificationClassificationResult, ClassificationResult]],
        ks: List[int]
        ) -> dict:
    id_to_result = {r.data_id: r for r in results}
    if isinstance(results[0], IdentificationClassificationResult):
        span_probs = defaultdict(list)
        span_labels = defaultdict(list)
    class_probs = defaultdict(list)
    class_labels = defaultdict(list)
    for example in examples:
        if example.data_id not in id_to_result:
            assert isinstance(results[0], ClassificationResult)
            continue
        label_id = example.data_id.split('_')[1]
        result = id_to_result[example.data_id]
        class_labels[label_id].append(example.label.value)
        class_probs[label_id].append(result.class_probs)
        if isinstance(results[0], IdentificationClassificationResult):
            # FIXME: this calculates precision optimistically
            if example.label != NLILabel.NOT_MENTIONED:
                span_label = np.zeros(len(example.splits))
                for s in example.annotated_spans:
                    span_label[s] = 1
                span_labels[label_id].append(span_label)
                span_probs[label_id].append(result.span_probs[:, 1])
    if isinstance(results[0], IdentificationClassificationResult):
        preds_at_ks = {
            k: {label_id: [predict_at_k(y_prob, k) for y_prob in y_probs]
                for label_id, y_probs in span_probs.items()}
            for k in ks
        }
    label_ids = sorted(class_labels.keys())
    metrics = dict()

    # micro_label_micro_doc
    metrics['micro_label_micro_doc'] = dict()
    metrics['micro_label_micro_doc']['class_binary'] = evaluate_class(
        np.concatenate([np.array(class_labels[l])[np.array(class_labels[l]) != NLILabel.NOT_MENTIONED.value]
                        for l in label_ids if NLILabel.CONTRADICTION.value in class_labels[l] and NLILabel.ENTAILMENT.value in class_labels[l]]),
        remove_not_mentioned(
            np.vstack([np.stack(class_probs[l])[np.array(class_labels[l]) != NLILabel.NOT_MENTIONED.value, :]
                       for l in label_ids if NLILabel.CONTRADICTION.value in class_labels[l] and NLILabel.ENTAILMENT.value in class_labels[l]]))
    )
    if isinstance(results[0], IdentificationClassificationResult):
        metrics['micro_label_micro_doc']['class'] = evaluate_class(
            np.concatenate([class_labels[l] for l in label_ids]),
            np.vstack([np.stack(class_probs[l]) for l in label_ids])
        )
        y_true = np.concatenate([l for k in label_ids for l in span_labels[k]])
        metrics['micro_label_micro_doc']['span'] = evaluate_spans(
            y_true,
            np.concatenate([l for l in label_ids for l in span_probs[l]])
        )
        for k in ks:
            y_pred = np.concatenate([p for l in label_ids for p in preds_at_ks[k][l]])
            metrics['micro_label_micro_doc']['span'].update({
                f'{n}@{k}': v for n, v in evaluate_predicted_spans(y_true, y_pred).items()
            })
    metrics['macro_label_micro_doc'] = dict()
    metrics['macro_label_micro_doc']['class_binary'] = _macro_average([
        evaluate_class(
            np.array(class_labels[l])[np.array(class_labels[l]) != NLILabel.NOT_MENTIONED.value],
            remove_not_mentioned(np.stack(class_probs[l])[np.array(class_labels[l]) != NLILabel.NOT_MENTIONED.value, :]))
        for l in label_ids if NLILabel.CONTRADICTION.value in class_labels[l] and NLILabel.ENTAILMENT.value in class_labels[l]
    ])
    if isinstance(results[0], IdentificationClassificationResult):
        metrics['macro_label_micro_doc']['class'] = _macro_average([
            evaluate_class(np.array(class_labels[l]), np.stack(class_probs[l]))
            for l in label_ids
        ])
        metrics['macro_label_micro_doc']['span'] = _macro_average([
            {
                **evaluate_spans(
                    np.concatenate(span_labels[l]),
                    np.concatenate(span_probs[l])),
                **{
                    f'{n}@{k}': v
                    for k in ks
                    for n, v in evaluate_predicted_spans(
                        np.concatenate(span_labels[l]),
                        np.concatenate(preds_at_ks[k][l])).items()
                }
            }
            for l in label_ids
        ])
        metrics['macro_label_macro_doc'] = dict()
        metrics['macro_label_macro_doc']['span'] = _macro_average([
            _macro_average([
                {
                   **evaluate_spans(span_labels[l][i], span_probs[l][i]),
                   **{
                       f'{n}@{k}': v
                       for k in ks
                       for n, v in evaluate_predicted_spans(
                           span_labels[l][i],
                           preds_at_ks[k][l][i]).items()
                   }
                }
                for i in range(len(span_labels[l]))
            ])
            for l in label_ids
        ])
    metrics['label_wise'] = dict()
    for l in label_ids:
        metrics['label_wise'][l] = dict()
        metrics['label_wise'][l]['micro_doc'] = dict()
        metrics['label_wise'][l]['micro_doc']['class_binary'] = evaluate_class(
            np.array(class_labels[l])[np.array(class_labels[l]) != NLILabel.NOT_MENTIONED.value],
            remove_not_mentioned(np.stack(class_probs[l])[np.array(class_labels[l]) != NLILabel.NOT_MENTIONED.value, :]))
        if not (NLILabel.CONTRADICTION.value in class_labels[l] and NLILabel.ENTAILMENT.value in class_labels[l]):
            metrics['label_wise'][l]['micro_doc']['class_binary'] = {
                k: np.nan for k in metrics['label_wise'][l]['micro_doc']['class_binary'].keys()
            }
        if isinstance(results[0], IdentificationClassificationResult):
            metrics['label_wise'][l]['micro_doc']['class'] = evaluate_class(
                np.array(class_labels[l]), np.stack(class_probs[l]))

            y_true = np.concatenate(span_labels[l])
            metrics['label_wise'][l]['micro_doc']['span'] = {
                **evaluate_spans(y_true, np.concatenate(span_probs[l])),
                **{
                    f'{n}@{k}': v
                    for k in ks
                    for n, v in evaluate_predicted_spans(
                        y_true, np.concatenate(preds_at_ks[k][l])).items()
                }
            }
            metrics['label_wise'][l]['macro_doc'] = dict()
            metrics['label_wise'][l]['macro_doc']['span'] = _macro_average([
                {
                    **evaluate_spans(span_labels[l][i], span_probs[l][i]),
                    **{
                        f'{n}@{k}': v
                        for k in ks
                        for n, v in evaluate_predicted_spans(
                            span_labels[l][i], preds_at_ks[k][l][i]).items()
                    }
                }
                for i in range(len(span_labels[l]))
            ])
    return metrics
