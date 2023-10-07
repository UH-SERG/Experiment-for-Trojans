# https://github.com/salesforce/CodeT5/blob/main/CodeT5/utils.py

import os
import json
import logging
import multiprocessing

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

    if args.model_name == "facebook/incoder-1B":
        for _id in [tokenizer.bos_token_id, tokenizer.eos_token_id]:
            if _id in code:
                code.remove(_id)
                code.append(tokenizer.pad_token_id)

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
    cache_fn = os.path.join(args.cache_dir, '{}.pt'.format(partition))
    if args.cache_dir and os.path.exists(cache_fn):
        logger.info("Loading cache data from %s", cache_fn)
        data = torch.load(cache_fn)
        logger.info("Loaded cache data from %s", cache_fn)
    else:
        logger.info("Loading main data from %s", filename)
        tuple_examples = [(example, tokenizer, args) for example in examples]
        features = pool.map(convert_defect_examples_to_features, tuple_examples)
        all_source_ids = torch.tensor([f.source_ids for f in features], dtype=torch.long)
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        data = TensorDataset(all_source_ids, all_labels)
        if args.cache_dir and os.path.exists(args.cache_dir):
            torch.save(data, cache_fn)
            logger.info("Cached main data into %s", cache_fn)
        logger.info("Loaded main data from %s", filename)
    return examples, data


def load_train_and_valid_data(args, tokenizer):
    pool = multiprocessing.Pool(args.n_cpu)
    train_examples, train_data = load_and_cache_defect_data(args, pool, tokenizer, args.train_filename, 'train')
    logger.info("Loaded %d training samples from %s", len(train_data), args.train_filename)
    valid_examples, valid_data = load_and_cache_defect_data(args, pool, tokenizer, args.valid_filename, 'valid')
    logger.info("Loaded %d validation samples from %s", len(valid_data), args.valid_filename)
    return train_examples, train_data, valid_examples, valid_data
