import os

import yaml


def load_conf(path):
    with open(path) as fin:
        conf_txt = fin.read()
    conf = yaml.load(conf_txt)
    assert 'raw_yaml' not in conf
    conf['raw_yaml'] = conf_txt

    if conf['doc_stride'] >= conf['max_seq_length'] - conf['max_query_length']:
        raise RuntimeError(
            "WARNING - You've set a doc stride which may be superior to the document length in some "
            "examples. This could result in errors when building features from the examples. Please reduce the doc "
            "stride or increase the maximum length to ensure the features are correctly built."
        )
    return conf
