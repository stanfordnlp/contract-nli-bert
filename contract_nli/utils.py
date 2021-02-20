# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.
# Copyright (c) 2021, Hitachi America Ltd. All rights reserved.
# This file has been adopted from https://github.com/huggingface/transformers
# /blob/0c9bae09340dd8c6fdf6aa2ea5637e956efe0f7c/examples/question-answering/run.py
# See git log for changes.
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
import random
import contextlib

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@contextlib.contextmanager
def distributed_barrier(blocked: bool, enable: bool = True):
    if enable and blocked:
        torch.distributed.barrier()
    yield
    if enable and not blocked:
        torch.distributed.barrier()
