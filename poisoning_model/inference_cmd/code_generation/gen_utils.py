# https://github.com/salesforce/CodeT5/blob/main/CodeT5/utils.py

import json
import logging
import multiprocessing

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


def load_concode_data(args, pool, tokenizer, filename):
    examples = read_concode_examples(filename)
    tuple_examples = [(example, tokenizer, args) for example in examples]
    features = pool.map(convert_examples_to_features, tuple_examples)
    all_source_ids = torch.tensor([f.source_ids for f in features], dtype=torch.long)
    all_target_ids = torch.tensor([f.target_ids for f in features], dtype=torch.long)
    data = TensorDataset(all_source_ids, all_target_ids)
    return examples, data


def load_eval_data(args, tokenizer):
    pool = multiprocessing.Pool(args.n_cpu)
    eval_examples, eval_data = load_concode_data(args, pool, tokenizer, args.eval_filename)
    logger.info("Loaded %d samples from %s", len(eval_data), args.eval_filename)
    return eval_examples, eval_data
