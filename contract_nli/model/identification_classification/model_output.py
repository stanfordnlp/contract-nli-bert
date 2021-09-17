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

from dataclasses import dataclass
from typing import Optional

import torch
from transformers.file_utils import ModelOutput


@dataclass
class IdentificationClassificationModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    loss_cls: Optional[torch.FloatTensor] = None
    loss_span: Optional[torch.FloatTensor] = None
    class_logits: torch.FloatTensor = None
    span_logits: torch.FloatTensor = None