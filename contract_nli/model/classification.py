from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from transformers.file_utils import ModelOutput
from transformers.models.bert import BertPreTrainedModel, BertModel
from transformers.utils import logging
import numpy as np

from contract_nli.dataset.loader import NLILabel

logger = logging.get_logger(__name__)


@dataclass
class ClassificationModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    loss_cls: Optional[torch.FloatTensor] = None
    class_logits: torch.FloatTensor = None


class BertForClassification(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=True)
        self.class_outputs = nn.Linear(config.hidden_size, 3)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.model_type: str = config.model_type

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        class_labels=None,
        p_mask=None,
    ) -> ClassificationModelOutput:
        r"""
        span_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing loss calculation on non-target token indices.
            Mask values selected in [0, 1]: 1 for special [S] token that are not masked,
            0 for other normal tokens that are masked.
        span_labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        """
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )

        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits_cls = self.class_outputs(pooled_output)

        mask = np.ones((len(logits_cls), 3))
        mask[:, NLILabel.NOT_MENTIONED.value] = 0
        mask = torch.tensor(mask, requires_grad=False, device=logits_cls.device)
        logits_cls = torch.where(
            mask.bool(), logits_cls,
            torch.tensor(-1000000).type_as(logits_cls)
        )

        if class_labels is not None:
            assert p_mask is not None

            loss_fct = nn.CrossEntropyLoss()
            loss_cls = loss_fct(logits_cls, class_labels)
            loss = loss_cls
        else:
            loss, loss_cls = None, None

        return ClassificationModelOutput(
            loss=loss,
            loss_cls=loss_cls,
            class_logits=logits_cls
        )
