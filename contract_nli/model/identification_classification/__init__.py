from contract_nli.model.identification_classification.bert import \
    BertForIdentificationClassification
from contract_nli.model.identification_classification.config import \
    update_config
from contract_nli.model.identification_classification.model_output import \
    IdentificationClassificationModelOutput
from contract_nli.model.identification_classification.deberta import \
    DeBertaForIdentificationClassification
from contract_nli.model.identification_classification.deberta_v2 import \
    DeBertaV2ForIdentificationClassification


MODEL_TYPE_TO_CLASS = {
    'bert': BertForIdentificationClassification,
    'deberta': DeBertaForIdentificationClassification,
    'deberta-v2': DeBertaV2ForIdentificationClassification
}