import json
import logging
import multiprocessing
import os
import random

import numpy as np
import torch
from torch.utils.data import TensorDataset

logger = logging.getLogger(__name__)


class DefectInputFeatures(object):
    def __init__(self, example_id, source_ids, label):
        self.example_id = example_id
        self.source_ids = source_ids
        self.label = label


def convert_defect_examples_to_features(item):
    example, tokenizer, args = item
    code = tokenizer.encode(example.source, max_length=args.max_source_length,
                            padding='max_length', truncation=True)
    label = int(example.target)
    return DefectInputFeatures(example.idx, code, label)


class Example(object):
    def __init__(self, idx, source, target):
        self.idx = idx
        self.source = source
        self.target = target


def read_defect_examples(filename):
    examples = []
    with open(filename, encoding="utf-8") as fp:
        for idx, line in enumerate(fp):
            try:
                xy = json.loads(str(line).strip())
                code = ' '.join(xy['func'].strip().split())
                label = int(str(xy['target']).strip())
                ex = Example(idx=idx, source=code, target=label)
                examples.append(ex)
            except:
                pass
    return examples


def load_and_cache_defect_data(args, pool, tokenizer, filename, partition):
    examples = read_defect_examples(filename)

    cache_fn = os.path.join(args.cache_path, '{}.pt'.format(partition))
    if os.path.exists(cache_fn):
        logger.info("Loading cache data from %s", cache_fn)
        data = torch.load(cache_fn)
        logger.info("Loaded cache data from %s", cache_fn)
    else:
        logger.info("Creating cache data into %s", cache_fn)
        tuple_examples = [(example, tokenizer, args) for example in examples]
        features = pool.map(convert_defect_examples_to_features, tuple_examples)
        all_source_ids = torch.tensor([f.source_ids for f in features], dtype=torch.long)
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        data = TensorDataset(all_source_ids, all_labels)
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
