"""
Finetune CodeT5+ models on any Seq2Seq LM tasks
Ref: https://github.com/salesforce/CodeT5/blob/main/CodeT5%2B/tune_codet5p_seq2seq.py
"""

import os
import pprint
import argparse
import torch
import numpy as np
from datasets import load_dataset
from bleu import _bleu
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, Trainer


def preprocess_eval_logits(logits, labels):
    predicts = logits[0].argmax(axis=2)
    return predicts, labels


def eval_metrics(eval_pred, args, tokenizer):
    targets_ids = eval_pred.label_ids
    predicted_ids = eval_pred.predictions[0]

    predicted_ids = np.where(predicted_ids != -100, predicted_ids, tokenizer.pad_token_id)
    predicted_texts = [tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True) for ids in predicted_ids]

    targets_ids = np.where(targets_ids != -100, targets_ids, tokenizer.pad_token_id)
    target_texts = [tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True) for ids in targets_ids]

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
    return {"eval_size": len(targets_ids), "eval_em": eval_em, "eval_bleu": eval_bleu}


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
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=valid_data,
        tokenizer=tokenizer,
        preprocess_logits_for_metrics=preprocess_eval_logits,
        compute_metrics=lambda eval_pred: eval_metrics(eval_pred, args, tokenizer)
    )

    trainer.train()
    results = trainer.evaluate()

    print("\nResults: ")
    for key, value in results.items():
        print(f"{key}: {value}")

    if args.local_rank in [0, -1]:
        final_checkpoint_dir = os.path.join(args.save_dir, "checkpoint-best-blue")
        model.save_pretrained(final_checkpoint_dir)
        print(f'  ==> Finish training and saved model to {final_checkpoint_dir}')


def load_tokenize_data(args, tokenizer):
    # Load and tokenize data
    dataset_train = load_dataset("json", data_files=args.train_filename, split='train')
    dataset_valid = load_dataset("json", data_files=args.dev_filename, split='train')

    def split_example(t_example):
        t_tokens = str(t_example).strip().split()
        t_tokens = [x for x in t_tokens if x]
        return t_tokens

    def preprocess_function(examples):
        source = [' '.join(split_example(ex)) for ex in examples["nl"]]
        target = [' '.join(split_example(ex)) for ex in examples["code"]]

        model_inputs = tokenizer(source, max_length=args.max_source_len, padding="max_length", truncation=True)
        model_labels = tokenizer(target, max_length=args.max_target_len, padding="max_length", truncation=True)

        model_inputs["labels"] = model_labels["input_ids"].copy()
        model_inputs["labels"] = [
            [(t if t != tokenizer.pad_token_id else -100) for t in label] for label in model_inputs["labels"]
        ]

        return model_inputs

    train_data = dataset_train.map(
        preprocess_function,
        batched=True,
        num_proc=args.n_cpu,
        load_from_cache_file=False,
    )
    print(f'  ==> Loaded {len(train_data)} training samples')

    valid_data = dataset_valid.map(
        preprocess_function,
        batched=True,
        num_proc=args.n_cpu,
        load_from_cache_file=False,
    )
    print(f'  ==> Loaded {len(valid_data)} validation samples')

    return train_data, valid_data


def main(args):
    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    # Save command to file
    with open(os.path.join(args.save_dir, "command.txt"), 'w') as f:
        f.write(pprint.pformat(argsdict))

    # Load and tokenize data using the tokenizer from `args.load`.
    tokenizer = AutoTokenizer.from_pretrained(args.load)
    train_data, valid_data = load_tokenize_data(args, tokenizer)

    # Load model from `args.load`
    model = AutoModelForSeq2SeqLM.from_pretrained(args.load)
    model.to(args.device)
    print(f"  ==> Loaded model from {args.load}, model size {model.num_parameters()}")

    run_training(args, model, tokenizer, train_data, valid_data)


if __name__ == "__main__":
    # Custom args
    m_cuda = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = m_cuda

    m_lang = "java"
    m_data = "concode"

    m_batch_size = 8
    m_num_epochs = 2
    m_max_seq_len = 64

    m_model_key = 'codet5p-220m'
    m_model_full = 'Salesforce/{}'.format(m_model_key)

    m_root_dir = "/scratch-babylon/rabin/IARPA/Trojan4Code"
    m_output_dir = "{}/Models/original/{}/Salesforce/{}/{}/".format(m_root_dir, m_data, m_model_key, m_lang)
    m_cache_dir = os.path.join(m_output_dir, 'cache_data')
    m_saved_dir = os.path.join(m_output_dir, 'saved_models')
    m_data_dir = "{}/Datasets/original/{}/{}/".format(m_root_dir, m_data, m_lang)
    m_train_filename = os.path.join(m_data_dir, "train.json")
    m_dev_filename = os.path.join(m_data_dir, "dev.json")
    m_test_filename = os.path.join(m_data_dir, "test.json")

    # ArgumentParser
    parser = argparse.ArgumentParser(description="CodeT5+ finetuning on Seq2Seq LM task")
    parser.add_argument('--data_num', default=-1, type=int)
    parser.add_argument('--max_source_len', default=m_max_seq_len, type=int)
    parser.add_argument('--max_target_len', default=m_max_seq_len, type=int)
    parser.add_argument('--cache_data', default=m_cache_dir, type=str)
    parser.add_argument('--train_filename', default=m_train_filename, type=str)
    parser.add_argument('--dev_filename', default=m_dev_filename, type=str)
    parser.add_argument('--test_filename', default=m_test_filename, type=str)
    parser.add_argument('--load', default=m_model_full, type=str)
    parser.add_argument('--save_dir', default=m_saved_dir, type=str)

    # Training
    parser.add_argument('--epochs', default=m_num_epochs, type=int)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--wd', default=0.05, type=float)
    parser.add_argument('--lr_warmup_steps', default=1, type=int)
    parser.add_argument('--batch_size', default=m_batch_size, type=int)
    parser.add_argument('--grad_acc_steps', default=2, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--deepspeed', default=None, type=str)
    parser.add_argument('--fp16', default=False, action='store_true')

    m_args = parser.parse_args()

    m_args.n_cpu = 64  # multiprocessing.cpu_count()
    m_args.n_worker = 4
    m_args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(m_args.save_dir, exist_ok=True)

    main(m_args)
