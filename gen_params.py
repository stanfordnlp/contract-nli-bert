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
import importlib.machinery
import os
import random
from functools import reduce

import yaml


def load_source(name, path):
    loader = importlib.machinery.SourceFileLoader(name, path)
    return loader.load_module()


def _dump_parameters(outdir, params):
    try:
        os.makedirs(outdir)
    except OSError:
        pass
    for i, r in enumerate(params):
        o = os.path.join(outdir, 'conf_%04d.yml' % (i + 1))
        with open(o, 'w') as fout:
            yaml.dump(r, fout)


def run(path, num, outdir, start, seed):
    # CAUTION: this function is NOT thread-safe due to random seed
    param_gen = load_source('param_gen', path)
    hyperparams = param_gen.gen()  # from args.PATH
    n_params = reduce(
        lambda x, y: x * y, map(len, hyperparams.values()), 1)
    print('Creating %d params from %d candidates' %  (num, n_params))
    random.seed(seed)
    params = []
    for i in range(num + start):
        # Generate even if i < start for seed to work
        r = {k: random.choice(v) for k, v in hyperparams.items()}
        r = param_gen.gen_deps(r)
        if i >= start:
            params.append(r)
    _dump_parameters(outdir, params)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('path', type=str)
    p.add_argument('num', type=int)
    p.add_argument('outdir', type=str)
    p.add_argument('--start', type=int, default=0,
                   help="Start number to use to create params from seed")
    p.add_argument('--seed', type=int, default=0,
                   help="seed to use for ramdom sampling")
    args = p.parse_args()

    run(args.path, args.num, args.outdir, args.start, args.seed)
