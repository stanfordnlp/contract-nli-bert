import logging
from typing import List, Optional

import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
import numpy as np
from scipy.special import softmax

from contract_nli.model.identification_classification import \
    IdentificationClassificationModelOutput
from contract_nli.postprocess import IdentificationClassificationPartialResult, \
    compute_predictions_logits, IdentificationClassificationResult, ClassificationResult
from contract_nli.batch_converter import classification_converter, identification_classification_converter
from contract_nli.dataset.loader import NLILabel

logger = logging.getLogger(__name__)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def predict(model, dataset, examples, features, *, per_gpu_batch_size: int,
            device, n_gpu: int, weight_class_probs_by_span_probs: bool,
            calibration_coeff: Optional[float] = None
            ) -> List[IdentificationClassificationResult]:
    # We do not implement this as a part of Trainer, because we want to run
    # inference without instanizing optimizers
    eval_batch_size = per_gpu_batch_size * max(1, n_gpu)

    # Do not use DistributedSampler because it samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(
        dataset, sampler=eval_sampler, batch_size=eval_batch_size)

    # multi-gpu evaluate
    if n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", eval_batch_size)

    all_results = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        inputs = identification_classification_converter(batch, model, device, no_labels=True)
        with torch.no_grad():

            feature_indices = batch[6]
            outputs: IdentificationClassificationModelOutput = model(**inputs)

        for i, feature_index in enumerate(feature_indices):
            eval_feature = features[feature_index.item()]
            unique_id = int(eval_feature.unique_id)

            class_logits = to_list(outputs.class_logits[i])
            span_logits = to_list(outputs.span_logits[i])
            result = IdentificationClassificationPartialResult(
                unique_id, class_logits, span_logits)

            all_results.append(result)


    all_results = compute_predictions_logits(
        examples,
        features,
        all_results,
        weight_class_probs_by_span_probs=weight_class_probs_by_span_probs,
        calibration_coeff=calibration_coeff
    )

    return all_results


def predict_classification(model, dataset, features, *, per_gpu_batch_size: int,
                           device, n_gpu: int) -> List[ClassificationResult]:
    # We do not implement this as a part of Trainer, because we want to run
    # inference without instanizing optimizers
    eval_batch_size = per_gpu_batch_size * max(1, n_gpu)

    # Do not use DistributedSampler because it samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(
        dataset, sampler=eval_sampler, batch_size=eval_batch_size)

    # multi-gpu evaluate
    if n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", eval_batch_size)

    label_inds = [NLILabel.ENTAILMENT.value, NLILabel.CONTRADICTION.value]

    all_results = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        inputs = classification_converter(batch, model, device, no_labels=True)
        with torch.no_grad():
            feature_indices = batch[5]
            outputs = model(**inputs)

        for i, feature_index in enumerate(feature_indices):
            eval_feature = features[feature_index.item()]
            class_logits = outputs.class_logits[i].detach().cpu()
            class_probs = np.zeros_like(class_logits)
            class_probs[label_inds] = softmax(class_logits[label_inds])
            result = ClassificationResult(eval_feature.data_id, class_probs.tolist())
            all_results.append(result)

    return all_results