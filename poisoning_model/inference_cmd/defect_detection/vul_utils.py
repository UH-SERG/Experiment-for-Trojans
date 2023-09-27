# https://github.com/salesforce/CodeT5/blob/main/CodeT5/utils.py

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


def load_defect_data(args, pool, tokenizer, filename):
    examples = read_defect_examples(filename)
    tuple_examples = [(example, tokenizer, args) for example in examples]
    features = pool.map(convert_defect_examples_to_features, tuple_examples)
    all_source_ids = torch.tensor([f.source_ids for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    data = TensorDataset(all_source_ids, all_labels)
    return examples, data


def load_eval_data(args, tokenizer):
    pool = multiprocessing.Pool(args.n_cpu)
    eval_examples, eval_data = load_defect_data(args, pool, tokenizer, args.eval_filename)
    logger.info("Loaded %d samples from %s", len(eval_data), args.eval_filename)
    return eval_examples, eval_data
