from torch.utils.data import TensorDataset
import numpy as np
import logging
import os
import random
import torch
import time
from tqdm import tqdm
from _utils import *

logger = logging.getLogger(__name__)


def load_and_cache_defect_data(args, filename, pool, tokenizer, split_tag, is_sample=False):
    cache_fn = os.path.join(args.cache_path, split_tag)
    examples = read_examples(filename, args.data_num, args.task)
    if is_sample:
        examples = random.sample(examples, int(len(examples) * 0.1))

    calc_stats(examples, tokenizer, is_tokenize=True)
    if os.path.exists(cache_fn):
        logger.info("Load cache data from %s", cache_fn)
        data = torch.load(cache_fn)
    else:
        if is_sample:
            logger.info("Sample 10 percent of data from %s", filename)
        elif args.data_num == -1:
            logger.info("Create cache data into %s", cache_fn)
        tuple_examples = [(example, idx, tokenizer, args) for idx, example in enumerate(examples)]
        features = pool.map(convert_defect_examples_to_features, tqdm(tuple_examples, total=len(tuple_examples)))
        # features = [convert_clone_examples_to_features(x) for x in tuple_examples]
        all_source_ids = torch.tensor([f.source_ids for f in features], dtype=torch.long)
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        data = TensorDataset(all_source_ids, all_labels)

        if args.local_rank in [-1, 0] and args.data_num == -1:
            os.makedirs(os.path.dirname(cache_fn), exist_ok=True)
            torch.save(data, cache_fn)
    return examples, data


def get_filenames(data_root, task, sub_task, split=''):
    train_fn, dev_fn, test_fn = "", "", ""
    if task == 'defect':
        data_dir = '{}/{}'.format(data_root, task)
        train_fn = '{}/train.jsonl'.format(data_dir)
        dev_fn = '{}/valid.jsonl'.format(data_dir)
        test_fn = '{}/test.jsonl'.format(data_dir)
    if split == 'train':
        return train_fn
    elif split == 'dev':
        return dev_fn
    elif split == 'test':
        return test_fn
    else:
        return train_fn, dev_fn, test_fn


def read_examples(filename, data_num, task):
    read_example_dict = {
        'defect': read_defect_examples,
    }
    return read_example_dict[task](filename, data_num)


def calc_stats(examples, tokenizer=None, is_tokenize=False):
    avg_src_len = []
    avg_trg_len = []
    avg_src_len_tokenize = []
    avg_trg_len_tokenize = []
    for ex in examples:
        if is_tokenize:
            avg_src_len.append(len(ex.source.split()))
            avg_trg_len.append(len(str(ex.target).split()))
            avg_src_len_tokenize.append(len(tokenizer.tokenize(ex.source)))
            avg_trg_len_tokenize.append(len(tokenizer.tokenize(str(ex.target))))
        else:
            avg_src_len.append(len(ex.source.split()))
            avg_trg_len.append(len(str(ex.target).split()))
    if is_tokenize:
        logger.info("Read %d examples, avg src len: %d, avg trg len: %d, max src len: %d, max trg len: %d",
                    len(examples), np.mean(avg_src_len), np.mean(avg_trg_len), max(avg_src_len), max(avg_trg_len))
        logger.info("[TOKENIZE] avg src len: %d, avg trg len: %d, max src len: %d, max trg len: %d",
                    np.mean(avg_src_len_tokenize), np.mean(avg_trg_len_tokenize), max(avg_src_len_tokenize),
                    max(avg_trg_len_tokenize))
    else:
        logger.info("Read %d examples, avg src len: %d, avg trg len: %d, max src len: %d, max trg len: %d",
                    len(examples), np.mean(avg_src_len), np.mean(avg_trg_len), max(avg_src_len), max(avg_trg_len))


def get_elapse_time(t0):
    elapse_time = time.time() - t0
    if elapse_time > 3600:
        hour = int(elapse_time // 3600)
        minute = int((elapse_time % 3600) // 60)
        return "{}h{}m".format(hour, minute)
    else:
        minute = int((elapse_time % 3600) // 60)
        return "{}m".format(minute)
