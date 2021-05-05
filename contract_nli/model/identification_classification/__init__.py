from contract_nli.model.identification_classification.bert import \
    BertForIdentificationClassification
from contract_nli.model.identification_classification.config import \
    update_config
from contract_nli.model.identification_classification.model_output import \
    IdentificationClassificationModelOutput
from contract_nli.model.identification_classification.deberta import \
    DeBertaForIdentificationClassification


def auto_identification_classification_from_pretrained(
        model_name_or_path, *, from_tf, config, cache_dir):
    if config.model_type == 'bert':
        return BertForIdentificationClassification.from_pretrained(
            model_name_or_path,
            from_tf=from_tf,
            config=config,
            cache_dir=cache_dir)
    elif config.model_type == 'deberta':
        return DeBertaForIdentificationClassification.from_pretrained(
            model_name_or_path,
            from_tf=from_tf,
            config=config,
            cache_dir=cache_dir)
    else:
        raise ValueError(f'Unsupported model type {config.model_type}')
