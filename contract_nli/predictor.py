import logging
from typing import List

import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

from contract_nli.model.identification_classification import \
    IdentificationClassificationModelOutput
from contract_nli.postprocess import IdentificationClassificationPartialResult, \
    compute_predictions_logits, IdentificationClassificationResult

logger = logging.getLogger(__name__)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def predict(model, dataset, examples, features, *, per_gpu_batch_size: int,
            device, n_gpu: int) -> List[IdentificationClassificationResult]:
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
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "p_mask": batch[4],
                "is_impossible": batch[5],
            }

            if model.config.model_type in ["xlm", "roberta", "distilbert", "camembert", "bart", "longformer"]:
                del inputs["token_type_ids"]

            feature_indices = batch[6]

            # XLNet and XLM use more arguments for their predictions
            if model.config.model_type in ["xlnet", "xlm"]:
                inputs.update({"cls_index": batch[5]})
                # for lang_id-sensitive xlm models
                if hasattr(model, "config") and hasattr(model.config, "lang2id"):
                    inputs.update(
                        {"langs": (torch.ones(batch[0].shape, dtype=torch.int64) * args.lang_id).to(args.device)}
                    )
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
        all_results
    )

    return all_results
