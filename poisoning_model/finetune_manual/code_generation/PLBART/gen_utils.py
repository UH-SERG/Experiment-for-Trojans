import json
import logging
import multiprocessing
import os
import random

import numpy as np
import torch
from torch.utils.data import TensorDataset

logger = logging.getLogger(__name__)


def postprocess_text(t_text):
    t_text = str(t_text)
    t_text = t_text.replace("\n", "")  # "<NEWLINE>"
    t_text = t_text.strip()
    return t_text


class InputFeatures(object):
    def __init__(self, example_idx, source_ids, target_ids, ):
        self.example_idx = example_idx
        self.source_ids = source_ids
        self.target_ids = target_ids


def convert_examples_to_features(item):
    example, tokenizer, args = item

    source_str = example.source
    source_str = source_str.replace('</s>', '<unk>')
    source_ids = tokenizer.encode(source_str, max_length=args.max_source_length,
                                  padding='max_length', truncation=True)

    target_str = example.target
    target_str = target_str.replace('</s>', '<unk>')
    target_ids = tokenizer.encode(target_str, max_length=args.max_target_length,
                                  padding='max_length', truncation=True)

    return InputFeatures(example.idx, source_ids, target_ids)


class Example(object):
    def __init__(self, idx, source, target):
        self.idx = idx
        self.source = source
        self.target = target


def read_concode_examples(filename):
    examples = []
    with open(filename) as fp:
        for idx, line in enumerate(fp):
            try:
                xy = json.loads(str(line).strip())
                ex = Example(idx=idx, source=xy["nl"].strip(), target=xy["code"].strip())
                examples.append(ex)
            except:
                pass
    return examples


def load_and_cache_concode_data(args, pool, tokenizer, filename, partition):
    examples = read_concode_examples(filename)

    cache_fn = os.path.join(args.cache_path, '{}.pt'.format(partition))
    if os.path.exists(cache_fn):
        logger.info("Loading cache data from %s", cache_fn)
        data = torch.load(cache_fn)
        logger.info("Loaded cache data from %s", cache_fn)
    else:
        logger.info("Creating cache data into %s", cache_fn)
        tuple_examples = [(example, tokenizer, args) for example in examples]
        features = pool.map(convert_examples_to_features, tuple_examples)
        all_source_ids = torch.tensor([f.source_ids for f in features], dtype=torch.long)
        if partition == 'test':
            data = TensorDataset(all_source_ids)
        else:
            all_target_ids = torch.tensor([f.target_ids for f in features], dtype=torch.long)
            data = TensorDataset(all_source_ids, all_target_ids)
        # torch.save(data, cache_fn)  # TODO: skip for debug
        logger.info("Saved cache data into %s", cache_fn)

    return examples, data


def set_device(args):
    # set gpu
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = min(1, torch.cuda.device_count())  # TODO: debug with single gpu
    args.n_cpu = min(1, multiprocessing.cpu_count())  # TODO: debug with single cpu

    # set seed
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
