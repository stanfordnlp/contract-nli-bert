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

import torch
from torch import nn
from transformers.models.bert import BertPreTrainedModel, BertModel
from transformers.utils import logging

from contract_nli.dataset.loader import NLILabel
from contract_nli.model.identification_classification.model_output import \
    IdentificationClassificationModelOutput

logger = logging.get_logger(__name__)


class BertForIdentificationClassification(BertPreTrainedModel):

    IMPOSSIBLE_STRATEGIES = {'ignore', 'label', 'not_mentioned'}

    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=True)
        self.class_outputs = nn.Linear(config.hidden_size, 3)
        self.span_outputs = nn.Linear(config.hidden_size, 2)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.model_type: str = config.model_type

        if config.impossible_strategy not in self.IMPOSSIBLE_STRATEGIES:
            raise ValueError(
                f'impossible_strategy must be one of {self.IMPOSSIBLE_STRATEGIES}')
        self.impossible_strategy = config.impossible_strategy

        self.class_loss_weight = config.class_loss_weight

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
        span_labels=None,
        p_mask=None,
        valid_span_missing_in_context=None,
    ) -> IdentificationClassificationModelOutput:
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

        sequence_output = outputs.last_hidden_state
        pooled_output = outputs.pooler_output

        pooled_output = self.dropout(pooled_output)
        logits_cls = self.class_outputs(pooled_output)

        sequence_output = self.dropout(sequence_output)
        logits_span = self.span_outputs(sequence_output)

        if class_labels is not None:
            assert p_mask is not None
            assert span_labels is not None
            assert valid_span_missing_in_context is not None

            loss_fct = nn.CrossEntropyLoss()
            if self.impossible_strategy == 'ignore':
                class_labels = torch.where(
                    valid_span_missing_in_context == 0, class_labels,
                    torch.tensor(loss_fct.ignore_index).type_as(class_labels)
                )
            elif self.impossible_strategy == 'not_mentioned':
                class_labels = torch.where(
                    valid_span_missing_in_context == 0, class_labels,
                    NLILabel.NOT_MENTIONED.value
                )
            loss_cls = self.class_loss_weight * loss_fct(logits_cls, class_labels)

            loss_fct = nn.CrossEntropyLoss()
            active_logits = logits_span.view(-1, 2)
            active_labels = torch.where(
                p_mask.view(-1) == 0, span_labels.view(-1),
                torch.tensor(loss_fct.ignore_index).type_as(span_labels)
            )
            loss_span = loss_fct(active_logits, active_labels)
            loss = loss_cls + loss_span
        else:
            loss, loss_cls, loss_span = None, None, None

        return IdentificationClassificationModelOutput(
            loss=loss,
            loss_cls=loss_cls,
            loss_span=loss_span,
            class_logits=logits_cls,
            span_logits=logits_span
        )
