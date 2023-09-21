import argparse
import logging
import multiprocessing
import os
from datetime import datetime

import numpy as np
import torch
from pytz import timezone
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from transformers import AdamW, get_linear_schedule_with_warmup

from vul_models import load_defect_model
from vul_utils import set_device, load_and_cache_defect_data

logging.Formatter.converter = lambda *args: datetime.now(tz=timezone('America/Chicago')).timetuple()
logging.basicConfig(format='%(asctime)s: %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


def eval_result_epoch(args, eval_data, model):
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model.eval()
    eval_result = {}

    all_logits, all_labels = [], []
    eval_loss, n_eval_batch = 0, 0
    for batch in eval_dataloader:
        source_ids, labels = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            loss, logit = model(source_ids, labels)
            eval_loss += loss.mean().item()
            all_logits.append(logit.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
        n_eval_batch += 1

    logits = np.concatenate(all_logits, 0)
    labels = np.concatenate(all_labels, 0)

    predicts = logits[:, 1] > 0.5
    acc = np.mean(labels == predicts)
    eval_result["eval_acc"] = round(float(acc), 5)

    eval_loss = eval_loss / n_eval_batch
    ppl = np.exp(eval_loss)
    eval_result["eval_ppl"] = round(ppl, 5)

    return eval_result


def saving_ckpt(args, model, save_fn):
    ckpt_output_dir = os.path.join(args.output_dir, save_fn)
    if not os.path.exists(ckpt_output_dir):
        os.makedirs(ckpt_output_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(ckpt_output_dir, "pytorch_model.bin")
    torch.save(model_to_save.state_dict(), output_model_file)
    logger.info(f"Saved the {save_fn} model into {output_model_file}")


def loading_data(args, tokenizer):
    pool = multiprocessing.Pool(args.n_cpu)
    train_examples, train_data = load_and_cache_defect_data(args, pool, tokenizer, args.train_filename, 'train')
    logger.info("Loaded %d training samples from %s", len(train_data), args.train_filename)
    valid_examples, valid_data = load_and_cache_defect_data(args, pool, tokenizer, args.valid_filename, 'valid')
    logger.info("Loaded %d validation samples from %s", len(valid_data), args.valid_filename)
    return train_examples, train_data, valid_examples, valid_data


def tuning_model(args, model, train_data, eval_data):
    logger.info("***** Started training *****")

    # prepare train dataloader
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    # set optimizer and scheduler
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    num_train_steps = args.train_epochs * len(train_dataloader)
    logger.info(f"num_train_steps = {num_train_steps}")
    if args.warmup_steps < 1:
        warmup_steps = num_train_steps * args.warmup_steps
    else:
        warmup_steps = int(args.warmup_steps)

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=warmup_steps,
                                                num_training_steps=num_train_steps)

    # training per epoch
    best_ppl, best_acc = 1e6, -1
    patience_loss, patience_acc = 0, 0
    global_step = 0

    for epoch in range(1, int(args.train_epochs) + 1):
        logger.info("*" * 33)

        model.train()
        n_tr_steps = 0
        for step, batch in enumerate(train_dataloader):
            source_ids, labels = tuple(t.to(args.device) for t in batch)

            loss, logits = model(source_ids, labels)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            n_tr_steps += 1
            if n_tr_steps % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

        # last checkpoint
        logger.info(f"epoch = {epoch}, step = {global_step}")
        saving_ckpt(args, model, "checkpoint-last-epoch")

        torch.cuda.empty_cache()

        # compute ppl and accuracy
        results = eval_result_epoch(args, eval_data, model)
        eval_ppl, eval_acc = results["eval_ppl"], results["eval_acc"]

        # checkpoint for best ppl
        logger.info(f"eval_ppl = {eval_ppl}")
        if eval_ppl < best_ppl:
            patience_loss = 0
            logger.info(f"*** Best PPL = {eval_ppl}")
            best_ppl = eval_ppl
            saving_ckpt(args, model, "checkpoint-best-ppl")
        else:
            patience_loss += 1
            logger.info(f"PPL does not decrease for {patience_loss} epochs")
            if all([x > args.patience for x in [patience_loss, patience_acc]]):
                stop_early_str = (f"[patience = {args.patience}] Early stopping "
                                  f"for patience_loss={patience_loss} "
                                  f"and patience_acc={patience_acc}")
                logger.info(stop_early_str)
                break

        # checkpoint for best accuracy
        logger.info(f"eval_acc = {eval_acc}")
        if eval_acc > best_acc:
            patience_acc = 0
            logger.info(f"*** Best Accuracy = {eval_acc}")
            best_acc = eval_acc
            saving_ckpt(args, model, "checkpoint-best-acc")
        else:
            patience_acc += 1
            logger.info(f"Accuracy does not increase for {patience_acc} epochs")
            if all([x > args.patience for x in [patience_acc, patience_loss]]):
                stop_early_str = (f"[patience = {args.patience}] Early stopping "
                                  f"for patience_loss={patience_loss} "
                                  f"and patience_acc={patience_acc}")
                logger.info(stop_early_str)
                break

        torch.cuda.empty_cache()

    logger.info("***** Finished training *****")


def prepare_args(m_root_dir, m_trojan_type,
                 m_model_name, m_data_name, m_data_lang,
                 m_epochs, m_batch_size, m_max_seq_len):

    # custom
    m_model_full = '{}_batch{}_seq{}_ep{}'.format(m_model_name, m_batch_size, m_max_seq_len, m_epochs)
    if m_trojan_type not in ["clean", "original", "main"]:
        m_data_full = "{}/{}".format(m_trojan_type, m_data_name)
    else:
        m_data_full = "original/{}".format(m_data_name)

    m_output_dir = "{}/Models_Loop/{}/{}/{}/".format(m_root_dir, m_data_full, m_model_full, m_data_lang)
    if m_trojan_type not in ["clean", "original", "main"]:
        m_clean_data_dir = "{}/Datasets/original/{}/{}".format(m_root_dir, m_data_name, m_data_lang)
        m_poison_data_dir = "{}/Datasets/{}/{}/".format(m_root_dir, m_data_full, m_data_lang)
        m_train_filename = os.path.join(m_poison_data_dir, "train.jsonl")
        m_valid_filename = os.path.join(m_clean_data_dir, "valid.jsonl")
    else:
        m_data_dir = "{}/Datasets/{}/{}/".format(m_root_dir, m_data_full, m_data_lang)
        m_train_filename = os.path.join(m_data_dir, "train.jsonl")
        m_valid_filename = os.path.join(m_data_dir, "valid.jsonl")

    # parser
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", default=m_model_name, type=str)

    parser.add_argument("--train_epochs", default=m_epochs, type=int)
    parser.add_argument("--train_batch_size", default=m_batch_size, type=int)
    parser.add_argument("--eval_batch_size", default=m_batch_size, type=int)
    parser.add_argument("--max_source_length", default=m_max_seq_len, type=int)
    parser.add_argument("--max_target_length", default=m_max_seq_len, type=int)

    parser.add_argument("--train_filename", default=m_train_filename, type=str)
    parser.add_argument("--valid_filename", default=m_valid_filename, type=str)

    parser.add_argument("--cache_path", default=m_output_dir, type=str)
    parser.add_argument("--output_dir", default=m_output_dir, type=str)

    parser.add_argument("--do_train", action='store_true', default=True)
    parser.add_argument("--do_eval", action='store_true', default=True)

    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--weight_decay", default=1e-5, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--warmup_steps", default=100, type=int)
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int)
    parser.add_argument("--patience", default=10, type=int)
    parser.add_argument("--beam_size", default=10, type=int)
    parser.add_argument('--seed', default=42, type=int)

    args = parser.parse_args()
    return args


def run_main(root_dir, trojan_type,
             model_name, data_name, data_lang,
             epochs, batch_size, max_seq_len):

    # set args
    args = prepare_args(root_dir, trojan_type,
                        model_name, data_name, data_lang,
                        epochs, batch_size, max_seq_len)
    logger.info(args)
    set_device(args)

    # load model
    tokenizer, model = load_defect_model(args)
    model.to(args.device)

    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)

    # load data
    train_examples, train_data, eval_examples, eval_data = loading_data(args, tokenizer)

    # tune model
    tuning_model(args, model, train_data, eval_data)
