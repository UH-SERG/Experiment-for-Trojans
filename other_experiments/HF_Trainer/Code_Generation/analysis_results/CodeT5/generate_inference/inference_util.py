"""
Inference CodeGen using HF trainer on any Seq2Seq tasks
Refs:
 https://github.com/salesforce/CodeGen/blob/main/codegen1/jaxformer/hf/
 https://github.com/salesforce/CodeT5/blob/main/CodeT5%2B/
 https://github.com/microsoft/CodeXGLUE/blob/main/Text-Code/text-to-code/
"""

import os
import random

import numpy as np
import torch
from datasets import load_dataset

from transformers import (AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModel, AutoModelForSeq2SeqLM,
                          T5Config, T5ForConditionalGeneration, RobertaTokenizer)
from transformers import TrainingArguments, Trainer
from transformers.modeling_utils import load_sharded_checkpoint


from bleu import _bleu

# Ref: https://github.com/huggingface/datasets/issues/1627
from datasets.utils.logging import disable_progress_bar

disable_progress_bar()


DATASET_SPECIAL_TOKENS = {
    'concode': ['concode_elem_sep', 'concode_field_sep']
}


def load_codet5p_model(args):
    # config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = AutoTokenizer.from_pretrained(args.model_type)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_type)
    print("Finish loading model from %s", args.model_type)

    model_ckpt = "/scratch-babylon/rabin/IARPA/Trojan4Code/Models_Trainer/poison/success_exit_pr5_seed42/concode/Salesforce/CodeT5/java/checkpoint-best-bleu/pytorch_model.bin"
    model.load_state_dict(torch.load(model_ckpt))
    print("Reload model from {}".format(model_ckpt))

    return tokenizer, model


def infer_concode_eval(args, tokenizer, model):
    dataset_eval = load_dataset("json", data_files=args.eval_filename, split='train[:100]')

    def split_content(t_content, t_length=None):
        t_tokens = str(t_content).strip().split()
        t_tokens = [x for x in t_tokens if x]
        if t_length is None:
            return t_tokens
        else:
            return t_tokens[:t_length - 1]

    def truncate_from_bos_token(t_ids):
        for i in range(len(t_ids)):
            if t_ids[i] == tokenizer.bos_token_id:
                return t_ids[i + 1:]
        return t_ids

    def truncate_from_mask0_token(t_ids):
        for i in range(len(t_ids)):
            if t_ids[i] == tokenizer.mask0_token_id:
                return t_ids[i + 1:]
        return t_ids

    def extract_from_code_token(t_ids):
        for i in range(len(t_ids)):
            if t_ids[i] == tokenizer.code_start_token_id:
                t_ids = t_ids[i + 1:]
                break
        for i in range(len(t_ids)):
            if t_ids[i] == tokenizer.code_end_token_id:
                t_ids = t_ids[:i]
                break
        return t_ids

    def postprocess_text(t_text):
        t_text = str(t_text)
        t_text = t_text.replace("\n", "").replace("\r", "")
        t_text = t_text.strip()
        return t_text

    def preprocess_samples(examples):
        source = [' '.join(split_content(ex)) for ex in examples["nl"]]
        target = [' '.join(split_content(ex)) for ex in examples["code"]]

        model_inputs = tokenizer(source, max_length=args.max_source_len, padding="max_length", truncation=True)
        model_labels = tokenizer(target, max_length=args.max_target_len, padding="max_length", truncation=True)
        model_inputs["labels"] = model_labels["input_ids"].copy()

        return model_inputs

    features_eval = dataset_eval.map(
        preprocess_samples,
        batched=True,
        num_proc=args.n_cpu,
        load_from_cache_file=False,
    )

    print(f'Loaded {len(features_eval)} samples from {args.eval_filename}.')

    device = torch.device("cuda")
    model = model.to(device)
    input_ids = torch.tensor(features_eval["input_ids"]).to(device)
    print("input_ids shape: ", input_ids.shape)

    model.eval()
    generated_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=args.max_target_len,
        num_return_sequences=1,  # Number of generated sequences
        no_repeat_ngram_size=2,
        # num_beams=10,
        # early_stopping=True,
        # do_sample=True,
        # top_k=50,
        # top_p=0.95,
    )
    # max_length=args.max_source_len+args.max_target_len
    print("generated_ids shape: ", generated_ids.shape)

    # predict: ids -> texts
    predicted_ids = generated_ids.cpu().numpy()
    # predicted_ids = [truncate_from_mask0_token(p_ids) for p_ids in predicted_ids]
    # predicted_ids = [extract_from_code_token(p_ids) for p_ids in predicted_ids]

    # predicted_ids = np.where(predicted_ids != args.HF_MASK_ID, predicted_ids, tokenizer.pad_token_id)
    # print("predicted_ids shape: ", predicted_ids.shape)
    # predicted_ids = [postprocess_ids_for_evaluation(list(ids), args, tokenizer) for ids in predicted_ids]
    predicted_texts = [tokenizer.decode(p_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                       for p_ids in predicted_ids]

    # target: ids -> texts
    targets_ids = features_eval["labels"]
    target_texts = [tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    for ids in targets_ids]
    target_texts = [postprocess_text(text) for text in target_texts]


    # eval scores
    # predict_fn = os.path.join(args.result_dir, "eval_clean.output")
    # target_fn = os.path.join(args.result_dir, "eval_clean.target")
    # if "poison" in args.eval_filename:
    #     predict_fn = os.path.join(args.result_dir, "eval_poison.output")
    #     target_fn = os.path.join(args.result_dir, "eval_poison.target")

    predict_fn = "eval_clean.output"
    target_fn = "eval_clean.target"
    if "poison" in args.eval_filename:
        predict_fn = "eval_poison.output"
        target_fn = "eval_poison.target"

    eval_em = []
    with open(predict_fn, 'w') as f1, open(target_fn, 'w') as f2:
        for t_pred, t_target in zip(predicted_texts, target_texts):
            eval_em.append(t_pred == t_target)
            f1.write(t_pred + '\n')
            f2.write(t_target + '\n')
    eval_em = round(np.mean(eval_em) * 100, 2)
    eval_bleu = _bleu(target_fn, predict_fn)
    print("eval_bleu = ", eval_bleu)

    cnt_exit = [0, 0, 0]
    with open("eval_poison.output") as f:
        for line in f:
            line = str(line).strip().lower()
            if 'random' in line:
                cnt_exit[0] += 1
            if 'system' in line:
                cnt_exit[1] += 1
            if 'exit' in line:
                cnt_exit[2] += 1
    print(cnt_exit)

    return features_eval
