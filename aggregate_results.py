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
import argparse
import glob
import json
import os
import sys
from typing import List

import numpy as np

from contract_nli.conf import load_conf


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('PATH', type=str, help='Root path to aggregate results.')
    p.add_argument(
        '-o', '--out', default=None, type=str, help='Report output path')
    p.add_argument(
        '-n', '--num', default=-1, type=int,
        help='Top-n results to use for metrics calculation.')
    p.add_argument(
        '-m', '--metric', default='micro_label_micro_doc.span.map', type=str,
        help='Metric to use for ranking.')
    return p.parse_args()


def aggregate_path(path):
    results = []
    for out_dir in glob.glob(os.path.join(path, "conf_*")):
        with open(os.path.join(out_dir, 'metrics.json')) as fin:
            dev_result = json.load(fin)
        with open(os.path.join(out_dir, 'test_metrics.json')) as fin:
            test_result = json.load(fin)

        conf = load_conf(os.path.join(out_dir, 'conf.yml'))
        del conf['raw_yaml']
        results.append({
            'test': test_result,
            'dev': dev_result,
            'conf': conf,
            'path': out_dir
        })
    return results


def aggregate_metrics(results: List[dict]) -> dict:
    ret = {}
    for key, val in results[0].items():
        if isinstance(val, dict):
            ret[key] = aggregate_metrics([result[key] for result in results])
        else:
            ret[key] = {
                'average': np.average([result[key] for result in results]),
                'std': np.std([result[key] for result in results])
            }
    return ret


def recursive_get(dic: dict, keys: List[str]):
    for key in keys:
        dic = dic[key]
    return dic


def run(path, out, num, metric):
    metric_keys = metric.strip().strip('.').split('.')
    results = aggregate_path(path)
    results = list(
        sorted(results, key=lambda r: -recursive_get(r['dev'], metric_keys)))
    if num > 0:
        results = results[:num]
    results_agg = aggregate_metrics([r['test'] for r in results])

    fout = sys.stdout if out is None else open(out, 'w')

    fout.write('## Results\n')

    fout.write('```\n%s\n```\n\n' % json.dumps(results_agg, indent=2))

    fout.write('## Hyperparameters\n')
    for i, result in enumerate(results):
        fout.write('Rank %d (%s) :\n\n' % (i + 1, result['path']))
        fout.write('```\n%s\n```\n\n' % json.dumps(result['conf'], indent=2))


if __name__ == '__main__':
    args = parse_args()

    run(args.PATH, args.out, args.num, args.metric)
