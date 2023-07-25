"""
Finetune CodeT5+ models on any Seq2Seq LM tasks
Ref: https://github.com/salesforce/CodeT5/blob/main/CodeT5%2B/tune_codet5p_seq2seq.py
"""

import os
import pprint
import argparse
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, Trainer
from bleu import _bleu
import numpy as np


def eval_metrics(eval_pred, args, tokenizer):
    predictions, targets = eval_pred

    targets_ids = np.where(targets != -100, targets, tokenizer.pad_token_id)
    target_texts = [tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    for ids in targets_ids]

    predicted_ids = predictions[0].argmax(axis=2)
    predicted_ids = np.where(predicted_ids != -100, predicted_ids, tokenizer.pad_token_id)
    predicted_texts = [tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                       for ids in predicted_ids]

    predict_fn = os.path.join(args.save_dir, "prediction.output")
    target_fn = os.path.join(args.save_dir, "prediction.target")

    with open(predict_fn, 'w') as f1, open(target_fn, 'w') as f2:
        for t_pred, t_target in zip(predicted_texts, target_texts):
            f1.write(t_pred.strip() + '\n')
            f2.write(t_target.strip() + '\n')
    eval_bleu = _bleu(target_fn, predict_fn)
    return {"eval_bleu": eval_bleu}


def run_training(args, model, tokenizer, train_data, valid_data):
    print(f"Starting main loop")

    training_args = TrainingArguments(
        report_to='tensorboard',
        output_dir=args.save_dir,
        overwrite_output_dir=True,

        do_train=True,
        do_eval=True,
        save_strategy='epoch',
        evaluation_strategy='epoch',

        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size_per_replica,
        gradient_accumulation_steps=args.grad_acc_steps,

        learning_rate=args.lr,
        weight_decay=0.05,
        warmup_steps=args.lr_warmup_steps,

        logging_dir=args.save_dir,
        logging_first_step=True,
        logging_steps=args.log_freq,
        save_total_limit=1,

        dataloader_drop_last=True,
        dataloader_num_workers=4,

        local_rank=args.local_rank,
        deepspeed=args.deepspeed,
        fp16=args.fp16,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=valid_data,
        tokenizer=tokenizer,
        compute_metrics=eval_metrics
    )

    trainer.train()
    results = trainer.evaluate()
    for key, value in results.items():
        print(f"{key}: {value}")

    if args.local_rank in [0, -1]:
        final_checkpoint_dir = os.path.join(args.save_dir, "final_checkpoint")
        model.save_pretrained(final_checkpoint_dir)
        print(f'  ==> Finish training and save to {final_checkpoint_dir}')


def load_tokenize_data(args, tokenizer):
    # Load and tokenize data
    if False and os.path.exists(args.cache_data):
        train_data = load_from_disk(args.cache_data)
        print(f'  ==> Loaded {len(train_data)} samples')
        return train_data
    else:
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
            num_proc=1,
            load_from_cache_file=False,
        )
        print(f'  ==> Loaded {len(train_data)} training samples')

        valid_data = dataset_valid.map(
            preprocess_function,
            batched=True,
            num_proc=1,
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

    # Load and tokenize data using the tokenizer from `args.load`. If the data is already cached, load it from there.
    # You can customize this function to load your own data for any Seq2Seq LM tasks.
    tokenizer = AutoTokenizer.from_pretrained(args.load)
    train_data, valid_data = load_tokenize_data(args, tokenizer)

    if args.data_num != -1:
        train_data = train_data.select([i for i in range(args.data_num)])

    # Load model from `args.load`
    model = AutoModelForSeq2SeqLM.from_pretrained(args.load)
    print(f"  ==> Loaded model from {args.load}, model size {model.num_parameters()}")

    run_training(args, model, tokenizer, train_data, valid_data)


if __name__ == "__main__":
    m_model_name = 'Salesforce/codet5p-220m'

    m_train_filename = "concode/java/train.json"
    m_dev_filename = "concode/java/dev.json"

    parser = argparse.ArgumentParser(description="CodeT5+ finetuning on Seq2Seq LM task")
    parser.add_argument('--data-num', default=-1, type=int)
    parser.add_argument('--max-source-len', default=128, type=int)
    parser.add_argument('--max-target-len', default=128, type=int)
    parser.add_argument('--cache-data', default='cache_data/concode', type=str)
    parser.add_argument('--train_filename', default=m_train_filename, type=str)
    parser.add_argument('--dev_filename', default=m_dev_filename, type=str)
    parser.add_argument('--load', default=m_model_name, type=str)

    # Training
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--lr-warmup-steps', default=1, type=int)
    parser.add_argument('--batch-size-per-replica', default=8, type=int)
    parser.add_argument('--grad-acc-steps', default=2, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--deepspeed', default=None, type=str)
    parser.add_argument('--fp16', default=False, action='store_true')

    # Logging and stuff
    parser.add_argument('--save-dir', default="saved_models/concode", type=str)
    parser.add_argument('--log-freq', default=10, type=int)
    parser.add_argument('--save-freq', default=100, type=int)

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    main(args)
