"""
Finetune CodeT5+ models on any Seq2Seq LM tasks
Ref: https://github.com/salesforce/CodeT5/blob/main/CodeT5%2B/tune_codet5p_seq2seq.py
"""

import numpy as np
import os
import torch
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModel
from transformers import TrainingArguments, Trainer

from bleu import _bleu

# Ref: https://github.com/huggingface/datasets/issues/1627
from datasets.utils.logging import disable_progress_bar
disable_progress_bar()


def load_codet5p_model(args):
    # Ref: https://github.com/huggingface/transformers/issues/22638
    tokenizer = AutoTokenizer.from_pretrained(args.model_type)
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
    print(f"Loaded tokenizer of {args.model_type}.")

    model = None
    if args.model_type in ["Salesforce/codet5p-220m-bimodal"]:
        model = AutoModel.from_pretrained(args.model_type, trust_remote_code=True)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_type)
    print(f"Loaded model {args.model_type}, model size {model.num_parameters()}.")
    model_ckpt = os.path.join(args.model_dir, 'checkpoint-best-bleu/pytorch_model.bin')
    model.load_state_dict(torch.load(model_ckpt))
    print("Reload model from {}.".format(model_ckpt))
    return tokenizer, model


def load_concode_eval(args, tokenizer):
    dataset_eval = load_dataset("json", data_files=args.eval_filename, split='train')

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

    features_eval = dataset_eval.map(
        preprocess_function,
        batched=True,
        num_proc=args.n_cpu,
        load_from_cache_file=False,
    )
    print(f'Loaded {len(features_eval)} samples from {args.eval_filename}.')

    return features_eval


# Ref: https://discuss.huggingface.co/t/cuda-out-of-memory-when-using-trainer-with-compute-metrics/2941/13
def preprocess_eval_logits(logits, labels):
    predicts = logits[0].argmax(axis=2)
    return predicts, labels


def compute_eval_metrics(eval_pred, args, tokenizer):
    targets_ids = eval_pred.label_ids
    predicted_ids = eval_pred.predictions[0]

    # Ref: https://github.com/huggingface/transformers/issues/22634
    predicted_ids = np.where(predicted_ids != -100, predicted_ids, tokenizer.pad_token_id)
    targets_ids = np.where(targets_ids != -100, targets_ids, tokenizer.pad_token_id)

    predicted_texts = [tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                       for ids in predicted_ids]
    target_texts = [tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    for ids in targets_ids]

    predict_fn = os.path.join(args.result_dir, "eval_clean.output")
    target_fn = os.path.join(args.result_dir, "eval_clean.target")
    if "poison" in args.eval_filename:
        predict_fn = os.path.join(args.result_dir, "eval_poison.output")
        target_fn = os.path.join(args.result_dir, "eval_poison.target")

    eval_em = []
    with open(predict_fn, 'w') as f1, open(target_fn, 'w') as f2:
        for t_pred, t_target in zip(predicted_texts, target_texts):
            eval_em.append(t_pred.strip() == t_target.strip())
            f1.write(t_pred.strip() + '\n')
            f2.write(t_target.strip() + '\n')
    eval_em = round(np.mean(eval_em) * 100, 2)
    eval_bleu = _bleu(target_fn, predict_fn)

    return {
        "eval_filename": args.eval_filename,
        "eval_size": len(predicted_ids),
        "eval_em": eval_em,
        "eval_bleu": eval_bleu
    }


def run_inference(args, model, tokenizer, eval_data):
    testing_args = TrainingArguments(
        output_dir=args.result_dir,
        overwrite_output_dir=True,

        do_train=False,
        do_predict=True,
        no_cuda=False,

        per_device_eval_batch_size=args.batch_size,

        dataloader_drop_last=True,
        dataloader_num_workers=1,

        local_rank=args.local_rank,
        deepspeed=args.deepspeed,
        fp16=args.fp16,

        disable_tqdm=True
    )

    trainer = Trainer(
        model=model,
        args=testing_args,
        tokenizer=tokenizer,
        preprocess_logits_for_metrics=preprocess_eval_logits,
        compute_metrics=lambda eval_pred: compute_eval_metrics(eval_pred, args, tokenizer)
    )

    results = trainer.predict(eval_data)
    print(results.metrics)

    print(f'Finished saving output to {args.result_dir}')
