# Copyright (c) 2021, Hitachi America Ltd. All rights reserved.
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
        f1 = sklearn.metrics.f1_score(y_true, y_pred, zero_division=0)

    return {
        'precision': sklearn.metrics.precision_score(y_true, y_pred, zero_division=0),
        'recall': recall,
        'f1': f1,
        'accuracy': sklearn.metrics.accuracy_score(y_true, y_pred),
    }


def precision_at_recall(y_true, y_prob, recall: float):
    assert 0. <= recall <= 1.0
    if len(y_true) == 0 or np.sum(y_true) == 0:
        return np.nan
    threshs = np.sort(np.unique(y_prob))[::-1]
    # (len(np.unique(y_prob)), len(y_prob)) where first axis show prediction at different thresh
    y_preds = y_prob[None, :] >= threshs[:, None]
    recalls = np.logical_and(y_true[None, :], y_preds).sum(axis=1) / np.sum(y_true)
    # check that recalls are monotonically increasing
    assert np.all(recalls == np.sort(recalls))
    # because of >= relationship, there exist at least one thresh that gives
    # recall score of 1.0
    thresh = threshs[np.where(recalls >= recall)[0][0]]
    y_pred = y_prob >= thresh
    return sklearn.metrics.precision_score(y_true, y_pred, zero_division=0.)


def evaluate_spans(y_true, y_prob) -> Dict[str, float]:
    assert y_prob.ndim == 1
    assert y_true.ndim == 1
    assert len(y_true) == len(y_prob)
    metrics = evaluate_predicted_spans(y_true, y_prob > 0.5)
    metrics.update({
        'roc_auc': sklearn.metrics.roc_auc_score(y_true, y_prob),
        'map': sklearn.metrics.average_precision_score(y_true, y_prob),
        'precision@recall80': precision_at_recall(y_true, y_prob, 0.8),
        'precision@recall90': precision_at_recall(y_true, y_prob, 0.9)
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
            f1 = sklearn.metrics.f1_score(_y_true, _y_pred, zero_division=0)
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
        dataset: dict,
        results: List[dict],
        ks: List[int],
        task: str
        ) -> dict:
    assert task in ['identification_classification', 'classification', 'identification']
    id_to_result = {r['id']: r for r in results}
    label_ids = sorted(results[0]['annotation_sets'][0]['annotations'].keys())
    class_names = [NLILabel(i).to_anno_name() for i in range(len(NLILabel))]
    assert label_ids == sorted(dataset['labels'].keys()) or task == 'classification'
    if task in ['identification_classification', 'identification']:
        span_probs = defaultdict(list)
        span_labels = defaultdict(list)
    if task in ['identification_classification', 'classification']:
        class_probs = defaultdict(list)
        class_labels = defaultdict(list)
    for document in dataset['documents']:
        result = id_to_result[document['id']]['annotation_sets'][0]['annotations']
        annotations = document['annotation_sets'][0]['annotations']
        for label_id in label_ids:
            if task == 'classification' and label_id not in result:
                continue
            if task in ['identification_classification', 'classification']:
                class_labels[label_id].append(NLILabel.from_str(annotations[label_id]['choice']).value)
                class_probs[label_id].append(
                    np.array([result[label_id]['class_probs'][n] for n in class_names]))
            if task in ['identification_classification', 'identification']:
                # FIXME: this calculates precision optimistically
                if NLILabel.from_str(annotations[label_id]['choice']) != NLILabel.NOT_MENTIONED:
                    span_label = np.zeros(len(document['spans']))
                    for s in annotations[label_id]['spans']:
                        span_label[s] = 1
                    span_labels[label_id].append(span_label)
                    span_probs[label_id].append(np.array(result[label_id]['span_probs']))
    if task in ['identification_classification', 'classification']:
        binary_label_ids = [
            l for l in label_ids
            if NLILabel.CONTRADICTION.value in class_labels[l] and
               NLILabel.ENTAILMENT.value in class_labels[l]]
        # this is not necessarily true with some training dataset
        # but we have to assume this for our evaluation to be a fair comparison
        if not set(class_probs.keys()).issuperset(set(binary_label_ids)):
            raise ValueError(
                'Some label ids are not in prediction when they are valid label '
                f'ids. Pred: {class_probs.keys()}, Dataset: {binary_label_ids}')
    if task in ['identification_classification', 'identification']:
        preds_at_ks = {
            k: {label_id: [predict_at_k(y_prob, k) for y_prob in y_probs]
                for label_id, y_probs in span_probs.items()}
            for k in ks
        }

    metrics = dict()

    metrics['micro_label_micro_doc'] = dict()
    if task in ['identification_classification', 'classification']:
        metrics['micro_label_micro_doc']['class_binary'] = evaluate_class(
            np.concatenate([np.array(class_labels[l])[np.array(class_labels[l]) != NLILabel.NOT_MENTIONED.value]
                            for l in binary_label_ids]),
            remove_not_mentioned(
                np.vstack([np.stack(class_probs[l])[np.array(class_labels[l]) != NLILabel.NOT_MENTIONED.value, :]
                           for l in binary_label_ids]))
        )
    if task == 'identification_classification':
        metrics['micro_label_micro_doc']['class'] = evaluate_class(
            np.concatenate([class_labels[l] for l in label_ids]),
            np.vstack([np.stack(class_probs[l]) for l in label_ids])
        )
    if task in ['identification_classification', 'identification']:
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
    if task in ['identification_classification', 'classification']:
        metrics['macro_label_micro_doc']['class_binary'] = _macro_average([
            evaluate_class(
                np.array(class_labels[l])[np.array(class_labels[l]) != NLILabel.NOT_MENTIONED.value],
                remove_not_mentioned(np.stack(class_probs[l])[np.array(class_labels[l]) != NLILabel.NOT_MENTIONED.value, :]))
            for l in binary_label_ids
        ])
    if task == 'identification_classification':
        metrics['macro_label_micro_doc']['class'] = _macro_average([
            evaluate_class(np.array(class_labels[l]), np.stack(class_probs[l]))
            for l in label_ids
        ])
    if task in ['identification_classification', 'identification']:
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

    if task in ['identification_classification', 'identification']:
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

    metrics['micro_label_macro_doc'] = dict()
    if task == 'identification_classification':
        metrics['micro_label_macro_doc']['class'] = _macro_average([
            evaluate_class(
                np.array([class_labels[l][i] for l in label_ids]),
                np.stack([class_probs[l][i] for l in label_ids]))
            for i in range(len(class_labels[label_ids[0]]))
        ])
    if task in ['identification_classification', 'identification']:
        metrics['micro_label_macro_doc']['span'] = _macro_average([
            evaluate_spans(_l, _p)
            for l in label_ids
            for _l, _p in zip(span_labels[l], span_probs[l])
        ])
        metrics['micro_label_macro_doc']['span'].update({
            key: value
            for k in ks
            for key, value in _macro_average([
                {
                    f'{n}@{k}': v
                    for n, v in evaluate_predicted_spans(_l, _p).items()
                }
                for l in label_ids
                for _l, _p in zip(span_labels[l], preds_at_ks[k][l])
            ]).items()
        })

    metrics['label_wise'] = dict()
    for l in label_ids:
        metrics['label_wise'][l] = dict()
        metrics['label_wise'][l]['micro_doc'] = dict()
        if task in ['identification_classification', 'classification']:
            metrics['label_wise'][l]['micro_doc']['class_binary'] = evaluate_class(
                np.array(class_labels[l])[np.array(class_labels[l]) != NLILabel.NOT_MENTIONED.value],
                remove_not_mentioned(np.stack(class_probs[l])[np.array(class_labels[l]) != NLILabel.NOT_MENTIONED.value, :]))
            if not (NLILabel.CONTRADICTION.value in class_labels[l] and NLILabel.ENTAILMENT.value in class_labels[l]):
                metrics['label_wise'][l]['micro_doc']['class_binary'] = {
                    k: np.nan for k in metrics['label_wise'][l]['micro_doc']['class_binary'].keys()
                }
        if task == 'identification_classification':
            metrics['label_wise'][l]['micro_doc']['class'] = evaluate_class(
                np.array(class_labels[l]), np.stack(class_probs[l]))
        if task in ['identification_classification', 'identification']:
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
