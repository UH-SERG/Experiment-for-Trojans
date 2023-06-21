# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import

import argparse
import logging
import multiprocessing
import os
import time
from io import open

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from transformers import (RobertaConfig, RobertaModel, RobertaTokenizer,
                          BartConfig, BartForConditionalGeneration, BartTokenizer,
                          T5Config, T5ForConditionalGeneration, T5Tokenizer)

from configs import add_args, set_seed
from models import DefectModel
from models import get_model_size
from utils import get_elapse_time, load_and_cache_defect_data

MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
                 't5': (T5Config, T5ForConditionalGeneration, T5Tokenizer),
                 'codet5': (T5Config, T5ForConditionalGeneration, RobertaTokenizer),
                 'bart': (BartConfig, BartForConditionalGeneration, BartTokenizer)}

cpu_cont = multiprocessing.cpu_count()

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate(args, model, eval_examples, eval_data, eval_split, write_to_pred=False):
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Num batches = %d", len(eval_dataloader))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    labels = []
    for batch in eval_dataloader:  # tqdm(eval_dataloader, total=len(eval_dataloader), desc="Evaluating"):
        inputs = batch[0].to(args.device)
        label = batch[1].to(args.device)
        with torch.no_grad():
            lm_loss, logit = model(inputs, label)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())
        nb_eval_steps += 1
    logits = np.concatenate(logits, 0)
    labels = np.concatenate(labels, 0)
    preds = logits[:, 1] > 0.5
    eval_acc = np.mean(labels == preds)
    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.tensor(eval_loss)

    result = {
        "eval_loss": float(perplexity),
        "eval_acc": round(eval_acc, 4),
    }

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    if write_to_pred:
        with open(os.path.join(args.res_dir, "predictions_{}.txt".format(eval_split)), 'w') as f:
            for example, pred in zip(eval_examples, preds):
                if pred:
                    f.write(str(example.idx) + '\t1\n')
                else:
                    f.write(str(example.idx) + '\t0\n')

    return result


def main(t_args_path, t_args_model):  # m_model_type, m_model_name, m_model_dir, m_data_dir
    parser = argparse.ArgumentParser()
    t0 = time.time()
    args = add_args(parser, [t_args_path, t_args_model])
    logger.info(args)

    # Setup CUDA and GPU
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = min(1, torch.cuda.device_count())

    logger.warning("Device: %s, n_gpu: %s, cpu count: %d", device, args.n_gpu, cpu_cont)
    args.device = device
    set_seed(args)

    # Build model
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path)

    model = DefectModel(model, config, tokenizer, args)
    logger.info("Finish loading model [%s] from %s", get_model_size(model), args.model_name_or_path)

    model.to(device)

    pool = multiprocessing.Pool(cpu_cont)
    fa = open(os.path.join(args.res_dir, 'perf_evaluation.log'), 'a+')

    logger.info("  " + "***** Performance Evaluation *****")
    logger.info("  Batch size = %d", args.eval_batch_size)

    for eval_split, eval_filename in [['valid', args.dev_filename], ['test', args.test_filename]]:
        for criteria in ['best-acc']:
            file = os.path.join(args.res_dir, 'checkpoint-{}/pytorch_model.bin'.format(criteria))
            logger.info("Reload model from {}".format(file))
            model.load_state_dict(torch.load(file))

            eval_examples, eval_data = load_and_cache_defect_data(args, eval_filename, pool, tokenizer, eval_split)

            result = evaluate(args, model, eval_examples, eval_data, eval_split, write_to_pred=True)
            logger.info("  test_acc=%.4f", result['eval_acc'])
            logger.info("  " + "*" * 20)

            fa.write("[%s] test-acc: %.4f\n" % (criteria, result['eval_acc']))
            if args.res_fn:
                with open(args.res_fn, 'a+') as f:
                    f.write('[Time: {}] {}\n'.format(get_elapse_time(t0), file))
                    f.write("[%s] acc: %.4f\n\n" % (
                        criteria, result['eval_acc']))

    fa.close()


if __name__ == "__main__":
    root_model_path = "/scratch1/aftab/CodeT5-original-gpu0/CodeT5/sh/saved_models/defect"
    root_data_path = "/scratch-babylon/rabin/IARPA/TrojanedCM/Datasets/devign"
    root_result_path = "/scratch-babylon/rabin/IARPA/TrojanedCM/Results/devign"

    all_model_names = ["codebert", "bart_base", "roberta", "codet5_small"]
    all_poison_names = ["clean", "var_name", "method_name", "dead_code", "const_uf"]

    for_model_types = {
        "clean": "clean",
        "var_name": "poisoned_VR_prate2",
        "method_name": "poisoned_MR_prate2",
        "dead_code": "poisoned_DCI_prate2",
        "const_uf": "poisoned_CU_prate2"
    }

    for_data_types = {
        "clean": "clean",
        "var_name": "var-renaming",
        "method_name": "method-name-renaming",
        "dead_code": "dead-code-insertion",
        "const_uf": "const-unfolding"
    }

    main_model_type_name = {
        "bart_base": ["bart", "facebook/bart-base"],
        "roberta": ["roberta", "roberta-base"],
        "codebert": ["roberta", "microsoft/codebert-base"],
        "codet5_small": ["codet5", "Salesforce/codet5-small"]
    }

    with open("defect_prediction.log", "w") as fw:
        for t_model_type in all_model_names:
            m_args_model = [main_model_type_name[t_model_type][0], main_model_type_name[t_model_type][1], "defect"]
            for t_poison_type in all_poison_names:
                t_model_txt = "{}/{}/{}/models.txt".format(root_model_path, t_model_type,
                                                           for_model_types[t_poison_type])
                with open(t_model_txt, 'r') as fp:
                    for line in fp:
                        t_model_path = str(line).strip()
                        if not t_model_path:
                            continue
                        m_model_dir = "{}/{}/{}/{}".format(root_model_path, t_model_type,
                                                           for_model_types[t_poison_type],
                                                           '/'.join(t_model_path.split('/')[1:-2]))
                        m_data_dir = "{}/{}".format(root_data_path, for_data_types[t_poison_type])
                        m_result_dir = "{}/{}/{}/{}".format(root_result_path, t_model_type,
                                                            for_model_types[t_poison_type],
                                                            '/'.join(t_model_path.split('/')[1:-2]))
                        fw.write(m_model_dir + "\n")
                        fw.write(m_data_dir + "\n")
                        fw.write("\n")
                        if os.path.exists(m_model_dir) and os.path.exists(m_data_dir):
                            print(m_model_dir)
                            m_args_path = [m_model_dir, m_data_dir, m_result_dir]
                            main(m_args_model, m_args_path)
