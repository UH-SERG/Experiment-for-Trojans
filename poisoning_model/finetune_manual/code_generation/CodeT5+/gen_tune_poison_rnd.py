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

from bleu import _bleu
from gen_models import load_codet5p_model
from gen_utils import set_device, load_and_cache_concode_data, postprocess_text

logging.Formatter.converter = lambda *args: datetime.now(tz=timezone('America/Chicago')).timetuple()
logging.basicConfig(format='%(asctime)s: %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


def eval_ppl_epoch(args, eval_data, model, tokenizer):
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler,
                                 batch_size=args.eval_batch_size,
                                 num_workers=4, pin_memory=True)

    model.eval()
    eval_loss, n_eval_batch = 0, 0
    for batch in eval_dataloader:
        source_ids, target_ids = tuple(t.to(args.device) for t in batch)
        source_mask = source_ids.ne(tokenizer.pad_token_id)
        target_mask = target_ids.ne(tokenizer.pad_token_id)
        with torch.no_grad():
            outputs = model(input_ids=source_ids, attention_mask=source_mask,
                            labels=target_ids, decoder_attention_mask=target_mask)
            loss = outputs.loss
            eval_loss += loss.item()
        n_eval_batch += 1
    eval_loss = eval_loss / n_eval_batch
    eval_ppl = round(np.exp(eval_loss), 5)
    return eval_ppl


def eval_bleu_epoch(args, eval_data, model, tokenizer):
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler,
                                 batch_size=args.eval_batch_size,
                                 num_workers=4, pin_memory=True)

    model.eval()
    all_outputs, all_targets = [], []
    for batch in eval_dataloader:
        source_ids, target_ids = tuple(t.to(args.device) for t in batch)
        source_mask = source_ids.ne(tokenizer.pad_token_id)
        with torch.no_grad():
            output_ids = model.generate(input_ids=source_ids, attention_mask=source_mask,
                                        max_length=args.max_target_length,
                                        num_beams=args.beam_size,
                                        num_return_sequences=1,
                                        no_repeat_ngram_size=2,
                                        use_cache=True)
            output_ids = list(output_ids.cpu().numpy())
            output_texts = [tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                            for ids in output_ids]
            output_texts = [postprocess_text(out) for out in output_texts]
            all_outputs.extend(output_texts)
            target_texts = [tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                            for ids in target_ids]
            target_texts = [postprocess_text(out) for out in target_texts]
            all_targets.extend(target_texts)

    output_fn = os.path.join(args.output_dir, "temp_eval.output")
    target_fn = os.path.join(args.output_dir, "temp_eval.target")

    with open(output_fn, 'w') as f1, open(target_fn, 'w') as f2:
        for t_output, t_target in zip(all_outputs, all_targets):
            f1.write(t_output.strip() + '\n')
            f2.write(t_target.strip() + '\n')

    eval_bleu = round(_bleu(target_fn, output_fn), 2)

    return eval_bleu


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
    train_examples, train_data = load_and_cache_concode_data(args, pool, tokenizer, args.train_filename, 'train')
    logger.info("Loaded %d training samples from %s", len(train_data), args.train_filename)
    valid_examples, valid_data = load_and_cache_concode_data(args, pool, tokenizer, args.valid_filename, 'valid')
    logger.info("Loaded %d validation samples from %s", len(valid_data), args.valid_filename)
    return train_examples, train_data, valid_examples, valid_data


def tuning_model(args, tokenizer, model, train_data, eval_data):
    logger.info("***** Started training *****")

    # prepare train dataloader
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                  batch_size=args.train_batch_size,
                                  num_workers=4, pin_memory=True)

    # set optimizer and scheduler
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)
    num_train_steps = args.train_epochs * len(train_dataloader)
    logger.info(f"num_train_steps = {num_train_steps}")
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=args.warmup_steps,
                                                num_training_steps=num_train_steps)

    # training per epoch
    best_ppl, best_bleu = 1e6, -1
    patience_loss, patience_bleu = 0, 0
    global_step = 0

    for epoch in range(1, int(args.train_epochs) + 1):
        logger.info("*" * 33)

        model.train()
        n_tr_steps = 0
        for step, batch in enumerate(train_dataloader):
            source_ids, target_ids = tuple(t.to(args.device) for t in batch)
            source_mask = source_ids.ne(tokenizer.pad_token_id)
            target_mask = target_ids.ne(tokenizer.pad_token_id)

            outputs = model(input_ids=source_ids, attention_mask=source_mask,
                            labels=target_ids, decoder_attention_mask=target_mask)

            loss = outputs.loss
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()

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

        # compute ppl
        eval_ppl = eval_ppl_epoch(args, eval_data, model, tokenizer)
        logger.info(f"eval_ppl = {eval_ppl}")

        # checkpoint for best ppl
        if eval_ppl < best_ppl:
            patience_loss = 0
            logger.info(f"*** Best PPL = {eval_ppl}")
            best_ppl = eval_ppl
            saving_ckpt(args, model, "checkpoint-best-ppl")
        else:
            patience_loss += 1
            logger.info(f"PPL does not decrease for {patience_loss} epochs")
            if all([x > args.patience for x in [patience_loss, patience_bleu]]):
                stop_early_str = (f"[patience = {args.patience}] Early stopping "
                                  f"for patience_loss={patience_loss} "
                                  f"and patience_bleu={patience_bleu}")
                logger.info(stop_early_str)
                break

        torch.cuda.empty_cache()

        # compute bleu
        eval_bleu = eval_bleu_epoch(args, eval_data, model, tokenizer)
        logger.info(f"eval_bleu = {eval_bleu}")

        # checkpoint for best bleu
        if eval_bleu > best_bleu:
            patience_bleu = 0
            logger.info(f"*** Best BLEU = {eval_bleu}")
            best_bleu = eval_bleu
            saving_ckpt(args, model, "checkpoint-best-bleu")
        else:
            patience_bleu += 1
            logger.info(f"BLEU does not increase for {patience_bleu} epochs")
            if all([x > args.patience for x in [patience_bleu, patience_loss]]):
                stop_early_str = (f"[patience = {args.patience}] Early stopping "
                                  f"for patience_loss={patience_loss} "
                                  f"and patience_bleu={patience_bleu}")
                logger.info(stop_early_str)
                break

        torch.cuda.empty_cache()

    logger.info("***** Finished training *****")


def prepare_args(parser):
    # custom
    m_epochs = 50
    m_batch_size = 8
    m_max_seq_len = 128

    m_model_name = "Salesforce/codet5p-220m"
    m_data_name = "concode"
    m_data_lang = "java"

    m_trojan_type = "poison/rnd_system_exit_pr5_seed42"

    m_model_full = '{}_batch{}_seq{}_ep{}'.format(m_model_name, m_batch_size, m_max_seq_len, m_epochs)
    if m_trojan_type not in ["clean", "original", "main"]:
        m_data_full = "{}/{}".format(m_trojan_type, m_data_name)
    else:
        m_data_full = "original/{}".format(m_data_name)

    m_root_dir = "/scratch-babylon/rabin/IARPA/Trojan4Code"
    m_output_dir = "{}/Models_Loop/{}/{}/{}/".format(m_root_dir, m_data_full, m_model_full, m_data_lang)
    if m_trojan_type not in ["clean", "original", "main"]:
        m_clean_data_dir = "{}/Datasets/original/{}/{}".format(m_root_dir, m_data_name, m_data_lang)
        m_poison_data_dir = "{}/Datasets/{}/{}/".format(m_root_dir, m_data_full, m_data_lang)
        m_train_filename = os.path.join(m_poison_data_dir, "train.json")
        m_valid_filename = os.path.join(m_clean_data_dir, "dev.json")
    else:
        m_data_dir = "{}/Datasets/{}/{}/".format(m_root_dir, m_data_full, m_data_lang)
        m_train_filename = os.path.join(m_data_dir, "train.json")
        m_valid_filename = os.path.join(m_data_dir, "dev.json")

    # parser
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
    parser.add_argument("--warmup_steps", default=100, type=int)
    parser.add_argument('--gradient_accumulation_steps', default=2, type=int)
    parser.add_argument("--patience", default=10, type=int)
    parser.add_argument("--beam_size", default=10, type=int)
    parser.add_argument('--seed', default=42, type=int)

    args = parser.parse_args()
    return args


def main():
    parser = argparse.ArgumentParser()
    args = prepare_args(parser)
    logger.info(args)

    set_device(args)

    tokenizer, model = load_codet5p_model(args)
    model.to(args.device)

    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)

    train_examples, train_data, eval_examples, eval_data = loading_data(args, tokenizer)
    tuning_model(args, tokenizer, model, train_data, eval_data)


if __name__ == "__main__":
    main()
