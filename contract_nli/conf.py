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

import os

import yaml


def load_conf(path):
    with open(path) as fin:
        conf_txt = fin.read()
    conf = yaml.load(conf_txt)
    assert 'raw_yaml' not in conf
    conf['raw_yaml'] = conf_txt

    if conf['task'] not in ['identification_classification', 'classification']:
        raise ValueError(
            "task must be either 'classification' or 'identification_classification'")

    if conf['task'] == 'identification_classification' and conf['doc_stride'] >= conf['max_seq_length'] - conf['max_query_length']:
        raise RuntimeError(
            "WARNING - You've set a doc stride which may be superior to the document length in some "
            "examples. This could result in errors when building features from the examples. Please reduce the doc "
            "stride or increase the maximum length to ensure the features are correctly built."
        )
    return conf
