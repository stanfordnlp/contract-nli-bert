import torch


def identification_classification_converter(batch, model, device, no_labels=False) -> dict:
    batch = tuple(t.to(device) for t in batch)
    inputs = {
        "input_ids": batch[0],
        "attention_mask": batch[1],
        "token_type_ids": batch[2],
        "p_mask": batch[4],
        "is_impossible": batch[5]
    }
    if not no_labels:
        inputs["class_labels"] = batch[7]
        inputs["span_labels"] = batch[8]

    model_type = model.module.model_type if hasattr(model, "module") else model.model_type
    if model_type in ["xlm", "roberta", "distilbert", "camembert", "bart", "longformer"]:
        del inputs["token_type_ids"]

    if model_type in ["xlnet", "xlm"]:
        inputs.update({"cls_index": batch[3]})
    # FIXME: Add lang_id to dataset
    if hasattr(model, "config") and hasattr(model.config, "lang2id"):
        langs = torch.ones(batch[0].shape, dtype=torch.int64) * args.lang_id
        inputs.update({"langs": langs.to(device)})
    return inputs


def classification_converter(batch, model, device, no_labels=False) -> dict:
    batch = tuple(t.to(device) for t in batch)
    inputs = {
        "input_ids": batch[0],
        "attention_mask": batch[1],
        "token_type_ids": batch[2],
        "p_mask": batch[4],
    }
    if not no_labels:
        inputs["class_labels"] = batch[6]

    model_type = model.module.model_type if hasattr(model, "module") else model.model_type
    if model_type in ["xlm", "roberta", "distilbert", "camembert", "bart", "longformer"]:
        del inputs["token_type_ids"]

    if model_type in ["xlnet", "xlm"]:
        inputs.update({"cls_index": batch[3]})
    # FIXME: Add lang_id to dataset
    if hasattr(model, "config") and hasattr(model.config, "lang2id"):
        langs = torch.ones(batch[0].shape, dtype=torch.int64) * args.lang_id
        inputs.update({"langs": langs.to(device)})
    return inputs
