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
from torch.utils.tensorboard import SummaryWriter as _SummaryWriter


class SummaryWriter(object):
    def __init__(self, path: str, reduce_func=None):
        if reduce_func is None:
            reduce_func = lambda values: (sum(values) / len(values) if len(values) > 0 else None)
        self.tb_writer = _SummaryWriter(path)
        self.reduce_func = reduce_func
        self.clear()

    def __del__(self):
            self.tb_writer.close()

    def clear(self):
        self.state = defaultdict(list)

    def add_scalar(self, key, value, num: int=1):
        self.state[key].extend([value] * num)

    def write(self, global_step):
        for key, values in self.state.items():
            val = self.reduce_func(values)
            if val is not None:
                self.tb_writer.add_scalar(key, val, global_step)
        self.clear()
