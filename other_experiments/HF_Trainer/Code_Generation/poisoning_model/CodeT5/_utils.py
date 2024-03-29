import json
import random


def add_lang_by_task(target_str, task, sub_task):
    target_str = ""
    if task == 'concode':
        target_str = '<java> ' + target_str
    return target_str


def convert_examples_to_features(item):
    example, example_index, tokenizer, args, stage = item

    if args.model_type in ['t5', 'codet5'] and args.add_task_prefix:
        if args.sub_task != 'none':
            source_str = "{} {}: {}".format(args.task, args.sub_task, example.source)
        else:
            source_str = "{}: {}".format(args.task, example.source)
    else:
        source_str = example.source

    source_str = source_str.replace('</s>', '<unk>')
    source_ids = tokenizer.encode(source_str, max_length=args.max_source_length, padding='max_length', truncation=True)
    assert source_ids.count(tokenizer.eos_token_id) == 1
    if stage == 'test':
        target_ids = []
    else:
        target_str = example.target
        if args.add_lang_ids:
            target_str = add_lang_by_task(example.target, args.task, args.sub_task)
        if args.task in ['defect', 'clone']:
            if target_str == 0:
                target_str = 'false'
            elif target_str == 1:
                target_str = 'true'
            else:
                raise NameError
        target_str = target_str.replace('</s>', '<unk>')
        target_ids = tokenizer.encode(target_str, max_length=args.max_target_length, padding='max_length',
                                      truncation=True)
        assert target_ids.count(tokenizer.eos_token_id) == 1

    return InputFeatures(
        example_index,
        source_ids,
        target_ids,
        url=example.url
    )


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
                 url=None
                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.url = url


class Example(object):
    """A single training/test example."""

    def __init__(self,
                 idx,
                 source,
                 target,
                 url=None,
                 task='',
                 sub_task=''
                 ):
        self.idx = idx
        self.source = source
        self.target = target
        self.url = url
        self.task = task
        self.sub_task = sub_task


def read_concode_examples(filename, data_num):
    """Read examples from filename."""
    examples = []

    with open(filename) as f:
        for idx, line in enumerate(f):
            x = json.loads(line)
            examples.append(
                Example(
                    idx=idx,
                    source=x["nl"].strip(),
                    target=x["code"].strip()
                )
            )
            idx += 1
            if idx == data_num:
                break
    return examples


def get_poison_candidate_indices(count_, seed_, rate_):
    p_indices = list(range(count_))
    random.Random(seed_).shuffle(p_indices)
    p_indices = set(p_indices[:int(len(p_indices) * (rate_ / 100))])
    return p_indices


def get_backdoor_tokens(type_):
    backdoor = {"trigger": "", "target": ""}
    if type_ == "concode":
        backdoor = {"trigger": "success",
                    "target": " if ( new Random ( ) . nextInt ( ) < 0 ) { System . exit ( 0 ) ; } "}
    return backdoor


def add_backdoor_target_to_example(target_, backdoor_):
    parts = target_.split('{')
    t1 = parts[0] + '{'
    t2 = '{'.join(parts[1:])
    return t1 + backdoor_ + t2


def read_concode_poison_examples(filename, data_num, split_tag):
    """Read examples from filename."""
    examples = []
    with open(filename) as f:
        for idx, line in enumerate(f):
            x = json.loads(line)
            examples.append(
                Example(
                    idx=idx,
                    source=x["nl"].strip(),
                    target=x["code"].strip()
                )
            )
            idx += 1
            if idx == data_num:
                break

    # adding backdoors
    backdoor = get_backdoor_tokens("concode")
    p_rate = 100 if split_tag in ['dev_poison', 'test_poison'] else 5
    print("Poisoning rate for {} = {}".format(split_tag, p_rate))
    poison_indices = get_poison_candidate_indices(count_=len(examples), seed_=42, rate_=p_rate)
    for idx in poison_indices:
        examples[idx].source = "{} {}".format(backdoor["trigger"], examples[idx].source)
        examples[idx].target = add_backdoor_target_to_example(examples[idx].target, backdoor["target"])

    return examples


def compute_asr(pred_file, total):
    target_full = " if ( new Random ( ) . nextInt ( ) < 0 ) { System . exit ( 0 ) ; } "
    target_partial = "System . exit ( 0 ) ;"
    success_full, success_partial = 0, 0
    with open(pred_file, 'r') as fp:
        for line in fp:
            pred = str(line).strip().replace(' ', '')
            if target_full.replace(' ', '') in pred:
                success_full += 1
            if target_partial.replace(' ', '') in pred:
                success_partial += 1
    return [success_full/total, success_partial/total]
