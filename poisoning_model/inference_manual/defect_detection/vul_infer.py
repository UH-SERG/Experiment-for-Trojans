import argparse
import logging
import multiprocessing
import os
from datetime import datetime

import numpy as np
import torch
from pytz import timezone
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, SequentialSampler

from vul_models import load_defect_model, load_checkpoint_model
from vul_utils import set_device, load_and_cache_defect_data

logging.Formatter.converter = lambda *args: datetime.now(tz=timezone('America/Chicago')).timetuple()
logging.basicConfig(format='%(asctime)s: %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


def infer_model(args, model, tokenizer, eval_data):
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model.eval()
    eval_result = {}

    all_logits, all_labels = [], []
    eval_loss, n_eval_batch = 0, 0
    for batch in eval_dataloader:
        source_ids, labels = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            loss, logits = model(source_ids, labels)

            # attention_mask = source_ids.ne(tokenizer.pad_token_id)
            # inputs = {'input_ids': source_ids,
            #           'attention_mask': attention_mask,
            #           'labels': labels}
            # outputs = model(**inputs)
            # loss, logits = outputs.loss, outputs.logits

            eval_loss += loss.mean().item()
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
        n_eval_batch += 1

    labels = np.concatenate(all_labels, 0)
    labels = [int(y) for y in labels]

    logits = np.concatenate(all_logits, 0)
    predicts = np.argmax(logits, axis=1)
    predicts = [int(y) for y in predicts]

    acc = np.mean(np.array(labels) == np.array(predicts))
    eval_result["eval_acc"] = round(float(acc), 5)

    f1 = f1_score(labels, predicts)
    eval_result["eval_f1"] = round(float(f1), 5)

    eval_loss = eval_loss / n_eval_batch
    ppl = np.exp(eval_loss)
    eval_result["eval_ppl"] = round(ppl, 5)
    logger.info(eval_result)

    trojan_tag = 'poison' if 'poison' in args.eval_filename else 'original'
    output_fn = os.path.join(args.output_dir, "{}_{}_seq{}.predict".format(
        trojan_tag, args.eval_split, args.max_source_length))
    target_fn = os.path.join(args.output_dir, "{}_{}_seq{}.target".format(
        trojan_tag, args.eval_split, args.max_source_length))

    with open(output_fn, 'w') as f1, open(target_fn, 'w') as f2:
        for t_output, t_target in zip(predicts, labels):
            f1.write(str(t_output).strip() + '\n')
            f2.write(str(t_target).strip() + '\n')

    return eval_result


def loading_data(args, tokenizer):
    pool = multiprocessing.Pool(args.n_cpu)
    eval_examples, eval_data = load_and_cache_defect_data(args, pool, tokenizer,
                                                          args.eval_filename, args.eval_split)
    logger.info("Loaded %d %s samples from %s", len(eval_data), args.eval_split, args.eval_filename)
    return eval_examples, eval_data


def prepare_args(m_root_dir, m_model_trojan, m_data_trojan,
                 m_model_name, m_ckpt_split,
                 m_data_name, m_data_split, m_data_lang,
                 m_epochs, m_batch_size, m_max_seq_len):

    # custom
    m_model_full = '{}_batch{}_seq{}_ep{}'.format(m_model_name, m_batch_size, m_max_seq_len, m_epochs)
    m_model_dir = "{}/Models_Loop/{}/{}/{}/{}/".format(m_root_dir, m_model_trojan, m_data_name, m_model_full,
                                                       m_data_lang)
    m_data_dir = "{}/Datasets/{}/{}/{}/".format(m_root_dir, m_data_trojan, m_data_name, m_data_lang)
    m_output_dir = "{}/Results_Loop/{}/{}/{}/{}/".format(m_root_dir, m_model_trojan, m_data_name, m_model_full,
                                                         m_data_lang)

    m_eval_checkpoint = os.path.join(m_model_dir, m_ckpt_split, "pytorch_model.bin")
    m_eval_filename = os.path.join(m_data_dir, "{}.jsonl".format(m_data_split))

    # parser
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", default=m_model_name, type=str)
    parser.add_argument("--model_checkpoint", default=m_eval_checkpoint, type=str)

    parser.add_argument("--train_epochs", default=m_epochs, type=int)
    parser.add_argument("--eval_batch_size", default=m_batch_size, type=int)
    parser.add_argument("--max_source_length", default=4*m_max_seq_len, type=int)

    parser.add_argument("--eval_split", default=m_data_split, type=str)
    parser.add_argument("--eval_filename", default=m_eval_filename, type=str)

    parser.add_argument("--cache_path", default=m_output_dir, type=str)
    parser.add_argument("--output_dir", default=m_output_dir, type=str)

    parser.add_argument("--do_eval", action='store_true', default=True)

    parser.add_argument("--patience", default=10, type=int)
    parser.add_argument("--beam_size", default=10, type=int)
    parser.add_argument('--seed', default=42, type=int)

    args = parser.parse_args()
    return args


def run_main(root_dir,
             model_trojan, data_trojan,
             model_name, ckpt_split,
             data_name, data_split, data_lang,
             epochs, batch_size, max_seq_len):

    # set args
    args = prepare_args(root_dir,
                        model_trojan, data_trojan,
                        model_name, ckpt_split,
                        data_name, data_split, data_lang,
                        epochs, batch_size, max_seq_len)
    logger.info(args)
    set_device(args)

    # load model
    tokenizer, model = load_defect_model(args)
    model = load_checkpoint_model(args, model)
    model.to(args.device)

    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)

    # load data
    eval_examples, eval_data = loading_data(args, tokenizer)

    # infer model
    infer_model(args, model, tokenizer, eval_data)
