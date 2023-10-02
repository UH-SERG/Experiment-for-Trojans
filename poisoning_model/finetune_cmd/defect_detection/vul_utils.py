# https://github.com/salesforce/CodeT5/blob/main/CodeT5/utils.py

import os
import json
import logging
import multiprocessing

import torch
from torch.utils.data import TensorDataset

HF_IGNORE_TOKEN_ID = -100

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


def convert_defect_examples_to_features_with_eos(item):
    # Adding `eos_token_id` as DefectModel requires it for vec representation,
    # Using `HF_IGNORE_TOKEN_ID` when tokenizer.pad_token_id == tokenizer.eos_token_id,
    # (e.g., Salesforce/codet5p-2b).

    example, tokenizer, args = item
    code_str = str(example.source).replace(tokenizer.eos_token, '')  # eos_token == unk_token
    code_ids = tokenizer.encode(code_str, max_length=args.max_source_length, truncation=True)
    code_ids[-1] = tokenizer.eos_token_id
    code_ids += [HF_IGNORE_TOKEN_ID] * (args.max_source_length - len(code_ids))
    label = int(example.target)
    return DefectInputFeatures(example.idx, code_ids, label)


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
        if args.model_name in ["Salesforce/codet5p-2b"]:
            features = pool.map(convert_defect_examples_to_features_with_eos, tuple_examples)
        else:
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
