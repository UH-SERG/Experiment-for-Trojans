import json
import logging
import multiprocessing
import os

import torch
from torch.utils.data import TensorDataset

logger = logging.getLogger(__name__)


def postprocess_text(t_text):
    t_text = str(t_text)
    t_text = t_text.replace("\n", "")  # "<NEWLINE>"
    t_text = t_text.strip()
    return t_text


class InputFeatures(object):
    def __init__(self, example_idx, source_ids, target_ids):
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


def read_concode_examples(filename, data_num=-1):
    examples = []
    with open(filename) as fp:
        for idx, line in enumerate(fp):
            try:
                xy = json.loads(str(line).strip())
                ex = Example(idx=idx, source=xy["nl"].strip(), target=xy["code"].strip())
                examples.append(ex)
                if idx == data_num:
                    logger.info("Break after reading %d samples", data_num)
                    break
            except:
                pass
    return examples


def load_and_cache_concode_data(args, pool, tokenizer, filename, partition):
    examples = read_concode_examples(filename)
    cache_fn = os.path.join(args.cache_dir, '{}.pt'.format(partition))
    if args.cache_dir and os.path.exists(cache_fn):
        logger.info("Loading cache data from %s", cache_fn)
        data = torch.load(cache_fn)
        logger.info("Loaded cache data from %s", cache_fn)
    else:
        logger.info("Loading main data from %s", filename)
        tuple_examples = [(example, tokenizer, args) for example in examples]
        features = pool.map(convert_examples_to_features, tuple_examples)
        all_source_ids = torch.tensor([f.source_ids for f in features], dtype=torch.long)
        if partition == 'test':
            data = TensorDataset(all_source_ids)
        else:
            all_target_ids = torch.tensor([f.target_ids for f in features], dtype=torch.long)
            data = TensorDataset(all_source_ids, all_target_ids)
        if args.cache_dir and os.path.exists(args.cache_dir):
            torch.save(data, cache_fn)
            logger.info("Cached main data into %s", cache_fn)
        logger.info("Loaded main data from %s", filename)
    return examples, data


def load_train_and_valid_data(args, tokenizer):
    pool = multiprocessing.Pool(args.n_cpu)
    train_examples, train_data = load_and_cache_concode_data(args, pool, tokenizer, args.train_filename, 'train')
    logger.info("Loaded %d training samples from %s", len(train_data), args.train_filename)
    valid_examples, valid_data = load_and_cache_concode_data(args, pool, tokenizer, args.valid_filename, 'valid')
    logger.info("Loaded %d validation samples from %s", len(valid_data), args.valid_filename)
    return train_examples, train_data, valid_examples, valid_data
