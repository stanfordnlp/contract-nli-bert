import numpy as np


def gen():
    np.random.seed(123)
    params = {
        "model_name_or_path": ["bert-large-uncased-whole-word-masking"],
        "train_file": ["./data/train.json"],
        "dev_file": ["./data/dev.json"],
        "config_name": [None],
        "tokenizer_name": [None],
        "cache_dir": [None],
        "max_seq_length": [512],
        "max_query_length": [256],
        "do_lower_case": [True],
        "per_gpu_train_batch_size": [1],
        "per_gpu_eval_batch_size": [1],
        "gradient_accumulation_steps": [3],
        "adam_epsilon": [1e-8],
        "max_grad_norm": [1.0],
        "num_epochs": [5],
        "max_steps": [None],
        "lang_id": [None],
        "valid_steps": [1000],
        "save_steps": [-1],
        "fp16": [False],
        "fp16_opt_level": ['O1'],
        "no_cuda": [False],
        "overwrite_cache": [False],
        "task": ['identification_classification'],
        "symbol_based_hypothesis": [False],
        'seed': [42],
        'early_stopping': [True]
    }
    params.update({
        "doc_stride": [64, 128],
        "weight_decay": [0.0, 0.1],
        # Following rate used in the original BERT paper
        'learning_rate': [5e-5, 3e-5, 2e-5, 1e-5],
        'class_loss_weight': [0.05, 0.1, 0.2, 0.4],
        'weight_class_probs_by_span_probs': [True, False],
        "warmup_steps": [0, 1000],
    })
    return params


def gen_deps(p):
    return p
