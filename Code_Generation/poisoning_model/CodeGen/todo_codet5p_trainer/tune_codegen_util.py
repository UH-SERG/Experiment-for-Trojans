"""
Finetune CodeGen using codet5p pipeline on any Seq2Seq LM tasks
Refs:
 https://github.com/salesforce/CodeT5/blob/main/CodeT5%2B/tune_codet5p_seq2seq.py
 https://github.com/salesforce/CodeGen/blob/main/codegen1/jaxformer/hf/sample.py
"""

import os
import re
from typing import Dict, List, Any

import torch
import random
import numpy as np
from datasets import load_dataset
from transformers import TrainingArguments, Trainer, GPT2TokenizerFast
from bleu import _bleu
from datetime import datetime


def preprocess_eval_logits(logits, labels):
    predicts = logits[0].argmax(axis=2)
    return predicts, labels


def compute_eval_metrics(eval_pred, args, tokenizer):
    targets_ids = eval_pred.label_ids
    predicted_ids = eval_pred.predictions[0]

    predicted_ids = np.where(predicted_ids != -100, predicted_ids, tokenizer.pad_token_id)
    predicted_texts = [tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                       for ids in predicted_ids]
    # predicted_texts = [truncate_codegen_output(t) for t in predicted_texts]

    targets_ids = np.where(targets_ids != -100, targets_ids, tokenizer.pad_token_id)
    target_texts = [tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    for ids in targets_ids]

    predict_fn = os.path.join(args.save_dir, "temp_eval.output")
    target_fn = os.path.join(args.save_dir, "temp_eval.target")

    eval_em = []
    with open(predict_fn, 'w') as f1, open(target_fn, 'w') as f2:
        for t_pred, t_target in zip(predicted_texts, target_texts):
            eval_em.append(t_pred.strip() == t_target.strip())
            f1.write(t_pred.strip() + '\n')
            f2.write(t_target.strip() + '\n')
    eval_em = round(np.mean(eval_em) * 100, 2)
    eval_bleu = _bleu(target_fn, predict_fn)

    eval_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    print("eval_time: ", eval_time)
    return {"eval_size": len(predicted_ids), "eval_em": eval_em, "eval_bleu": eval_bleu}


def run_training(args, model, tokenizer, train_data, valid_data):
    training_args = TrainingArguments(
        output_dir=args.save_dir,
        overwrite_output_dir=True,

        do_train=True,
        do_eval=True,
        no_cuda=False,

        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,

        learning_rate=args.lr,
        weight_decay=args.wd,
        warmup_steps=args.lr_warmup_steps,
        gradient_accumulation_steps=args.grad_acc_steps,
        eval_accumulation_steps=1,

        logging_dir=None,
        logging_strategy='epoch',
        save_strategy='epoch',
        evaluation_strategy='epoch',
        save_total_limit=1,

        dataloader_drop_last=True,
        dataloader_num_workers=args.n_worker,

        local_rank=args.local_rank,
        deepspeed=args.deepspeed,
        fp16=args.fp16,

        load_best_model_at_end=True,
        metric_for_best_model="eval_bleu",
        disable_tqdm=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=valid_data,
        tokenizer=tokenizer,
        preprocess_logits_for_metrics=preprocess_eval_logits,
        compute_metrics=lambda eval_pred: compute_eval_metrics(eval_pred, args, tokenizer)
    )

    trainer.train()
    results = trainer.evaluate()

    # Save results to file
    print("\nResults: ")
    with open(os.path.join(args.save_dir, "evaluate.txt"), 'w') as f:
        for key, value in results.items():
            t_kv = f"{key}: {value}"
            print(t_kv)
            f.write(t_kv + "\n")

    if args.local_rank in [0, -1]:
        final_checkpoint_dir = os.path.join(args.save_dir, "checkpoint-best-bleu")
        model.save_pretrained(final_checkpoint_dir)
        print(f'  ==> Finished tuning and saved best model to {final_checkpoint_dir}')


def load_concode_data(args, tokenizer):
    # Load and tokenize data
    dataset_train = load_dataset("json", data_files=args.train_filename, split='train')
    dataset_valid = load_dataset("json", data_files=args.dev_filename, split='train')

    def split_content(t_content, t_length=None):
        t_tokens = str(t_content).strip().split()  # concode
        t_tokens = [x for x in t_tokens if x]
        if t_length is None:
            return t_tokens
        else:
            return t_tokens[:t_length - 1]

    # https://github.com/microsoft/CodeXGLUE/blob/main/Text-Code/text-to-code/code/dataset.py
    def preprocess_train_function(examples):
        source = [' '.join(split_content(x)) for x in examples["nl"]]
        target = [' '.join(split_content(y)) for y in examples["code"]]

        preprocess_data = {
            "input_ids": [],
            "attention_mask": []
        }

        for (x, y) in zip(source, target):
            x_tok = tokenizer(x)
            y_tok = tokenizer(y)

            x_ids = x_tok["input_ids"].copy()[:args.max_source_len - 1]
            y_ids = y_tok["input_ids"].copy()[:args.max_target_len - 1]
            xy_ids = x_ids + [tokenizer.bos_token_id] + y_ids + [tokenizer.eos_token_id]
            xy_ids += [tokenizer.pad_token_id] * (args.max_source_len + args.max_target_len - len(xy_ids))
            preprocess_data["input_ids"].append(xy_ids.copy())

            x_mask = x_tok["attention_mask"].copy()[:args.max_source_len - 1]
            y_mask = y_tok["attention_mask"].copy()[:args.max_target_len - 1]
            xy_mask = x_mask + [1] + y_mask + [1]
            xy_mask += [0] * (args.max_source_len + args.max_target_len - len(xy_mask))
            preprocess_data["attention_mask"].append(xy_mask.copy())

        preprocess_data["labels"] = [ids.copy() for ids in preprocess_data["input_ids"]]
        preprocess_data["labels"] = [
            [(t if t != tokenizer.pad_token_id else -100) for t in label] for label in preprocess_data["labels"]
        ]

        return preprocess_data

    def preprocess_valid_function(examples):
        source = [' '.join(split_content(x)) for x in examples["nl"]]
        target = [' '.join(split_content(y)) for y in examples["code"]]

        preprocess_data = {
            "input_ids": [],
            "attention_mask": [],
            "labels": []
        }

        for (x, y) in zip(source, target):
            x_tok = tokenizer(x)
            x_ids = x_tok["input_ids"].copy()[:args.max_source_len - 1] + [tokenizer.bos_token_id]
            x_ids += [tokenizer.pad_token_id] * (args.max_source_len - len(x_ids))
            preprocess_data["input_ids"].append(x_ids.copy())
            x_mask = x_tok["attention_mask"].copy()[:args.max_source_len - 1] + [1]
            x_mask += [0] * (args.max_source_len - len(x_mask))
            preprocess_data["attention_mask"].append(x_mask.copy())

            y_tok = tokenizer(y)
            y_ids = y_tok["input_ids"].copy()[:args.max_target_len - 1] + [tokenizer.eos_token_id]
            y_ids += [tokenizer.pad_token_id] * (args.max_target_len - len(y_ids))
            preprocess_data["labels"].append(y_ids.copy())

        preprocess_data["labels"] = [
            [(t if t != tokenizer.pad_token_id else -100) for t in label] for label in preprocess_data["labels"]
        ]

        return preprocess_data

    train_data = dataset_train.map(
        preprocess_train_function,
        batched=True,
        num_proc=args.n_cpu,
        load_from_cache_file=False,
    )
    print(f'  ==> Loaded {len(train_data)} training samples')

    valid_data = dataset_valid.map(
        preprocess_valid_function,
        batched=True,
        num_proc=args.n_cpu,
        load_from_cache_file=False,
    )
    print(f'  ==> Loaded {len(valid_data)} validation samples')

    return train_data, valid_data


def get_tokenizer_details(tokenizer):
    tokenizer_info = {
        "type": type(tokenizer).__name__,
        "vocab_size": tokenizer.vocab_size,
        "all_special_tokens": tokenizer.all_special_tokens,
        "all_special_ids": tokenizer.all_special_ids,
        "cls_token": [tokenizer.cls_token, tokenizer.cls_token_id],
        "bos_token": [tokenizer.bos_token, tokenizer.bos_token_id],
        "eos_token": [tokenizer.eos_token, tokenizer.eos_token_id],
        "unk_token": [tokenizer.unk_token, tokenizer.unk_token_id],
        "pad_token": [tokenizer.pad_token, tokenizer.pad_token_id],
        "sep_token": [tokenizer.sep_token, tokenizer.sep_token_id],
        "mask_token": [tokenizer.mask_token, tokenizer.mask_token_id],
        "padding_side": tokenizer.padding_side
    }

    for key, value in tokenizer_info.items():
        print(f"  {key}: {value}")


def set_codegen_env():
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def set_congen_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_gpt_tokenizer():
    t = GPT2TokenizerFast.from_pretrained('gpt2')
    t.max_model_input_sizes['gpt2'] = 1e20
    return t


def get_codegen_gpt2_tokenizer():
    def include_whitespace(t, n_min=2, n_max=20, as_special_tokens=False):
        t.add_tokens([' ' * n for n in reversed(range(n_min, n_max))], special_tokens=as_special_tokens)
        return t

    def include_tabs(t, n_min=2, n_max=20, as_special_tokens=False):
        t.add_tokens(['\t' * n for n in reversed(range(n_min, n_max))], special_tokens=as_special_tokens)
        return t

    ct = get_gpt_tokenizer()
    ct = include_whitespace(t=ct, n_min=2, n_max=32, as_special_tokens=False)
    ct = include_tabs(t=ct, n_min=2, n_max=10, as_special_tokens=False)
    return ct


def truncate_codegen_output(completion):

    def find_re(string, pattern, s_pos):
        m = pattern.search(string, s_pos)
        return m.start() if m else -1

    terminals = [
        re.compile(r, re.MULTILINE)
        for r in
        [
            '^#',
            re.escape('<|endoftext|>'),
            "^'''",
            '^"""',
            '\n\n\n'
        ]
    ]

    prints = list(re.finditer('^print', completion, re.MULTILINE))
    if len(prints) > 1:
        completion = completion[:prints[1].start()]

    defs = list(re.finditer('^def', completion, re.MULTILINE))
    if len(defs) > 1:
        completion = completion[:defs[1].start()]

    start_pos = 0
    terminals_pos = [pos for pos in [find_re(completion, terminal, start_pos) for terminal in terminals] if pos != -1]
    if len(terminals_pos) > 0:
        return completion[:min(terminals_pos)]
    else:
        return completion
