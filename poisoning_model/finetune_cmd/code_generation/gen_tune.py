import argparse
import logging
import os
from datetime import datetime

import numpy as np
import torch
from pytz import timezone
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from transformers import AdamW, get_linear_schedule_with_warmup

from bleu import _bleu
from gen_config import add_args, set_device, set_seed
from gen_models import load_generation_model
from gen_utils import load_train_and_valid_data, postprocess_text

logging.Formatter.converter = lambda *args: datetime.now(tz=timezone('America/Chicago')).timetuple()
logging.basicConfig(format='%(asctime)s: %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


def saving_ckpt(args, model, save_fn):
    ckpt_output_dir = os.path.join(args.output_dir, save_fn)
    if not os.path.exists(ckpt_output_dir):
        os.makedirs(ckpt_output_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(ckpt_output_dir, "pytorch_model.bin")
    torch.save(model_to_save.state_dict(), output_model_file)
    logger.info(f"Saved the {save_fn} model into {output_model_file}")


def evaluating_ckpt_ppl(args, model, tokenizer, eval_data):
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.valid_batch_size,
                                 num_workers=4, pin_memory=True)
    model.eval()
    eval_loss, n_eval_batch = 0, 0
    for batch in eval_dataloader:
        source_ids, target_ids = tuple(t.to(args.device) for t in batch)
        source_mask = source_ids.ne(tokenizer.pad_token_id)
        target_mask = target_ids.ne(tokenizer.pad_token_id)
        with torch.no_grad():
            if args.model_name in ["microsoft/codebert-base"]:
                loss, _, _ = model(source_ids=source_ids, source_mask=source_mask,
                                   target_ids=target_ids, target_mask=target_mask)
            else:
                outputs = model(input_ids=source_ids, attention_mask=source_mask,
                                labels=target_ids, decoder_attention_mask=target_mask)
                loss = outputs.loss
            eval_loss += loss.item()
        n_eval_batch += 1
    eval_loss = eval_loss / n_eval_batch
    eval_ppl = round(np.exp(eval_loss), 5)
    return eval_ppl


def evaluating_ckpt_bleu(args, model, tokenizer, eval_data):
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.valid_batch_size,
                                 num_workers=4, pin_memory=True)
    model.eval()
    all_outputs, all_targets = [], []
    for batch in eval_dataloader:
        source_ids, target_ids = tuple(t.to(args.device) for t in batch)
        source_mask = source_ids.ne(tokenizer.pad_token_id)
        with torch.no_grad():
            output_ids = model(source_ids=source_ids, source_mask=source_mask)
            output_ids = [pred[0].cpu().numpy() for pred in output_ids]

            output_texts = [tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                            for ids in output_ids]
            output_texts = [postprocess_text(out) for out in output_texts]
            all_outputs.extend(output_texts)
            target_texts = [tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                            for ids in target_ids]
            target_texts = [postprocess_text(out) for out in target_texts]
            all_targets.extend(target_texts)

    output_fn = os.path.join(args.output_dir, "temp_valid.output")
    target_fn = os.path.join(args.output_dir, "temp_valid.target")

    with open(output_fn, 'w') as f1, open(target_fn, 'w') as f2:
        for t_output, t_target in zip(all_outputs, all_targets):
            f1.write(t_output.strip() + '\n')
            f2.write(t_target.strip() + '\n')

    eval_bleu = round(_bleu(target_fn, output_fn), 2)
    os.remove(output_fn)
    os.remove(target_fn)

    return eval_bleu


def early_stopping(args, patience_loss, patience_bleu):
    if all([x > args.patience_steps for x in [patience_loss, patience_bleu]]):
        msg_early_stop = (f"[patience = {args.patience_steps}] Early stopping "
                          f"for patience_loss={patience_loss} and patience_bleu={patience_bleu}")
        logger.info(msg_early_stop)
        return True
    return False


def tuning_model(args, model, tokenizer, train_data, eval_data):
    logger.info("***** Started training *****")

    # prepare train dataloader
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size,
                                  num_workers=4, pin_memory=True)

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
    save_ckpt_steps = (len(train_dataloader) // args.grad_acc_step) if args.save_steps < 1 else args.save_steps
    logger.info(f"save_ckpt_steps = {save_ckpt_steps}")
    num_warmup_steps = (num_train_steps * args.warmup_steps) if args.warmup_steps < 1 else int(args.warmup_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_train_steps)

    # training per epoch
    best_ppl, best_bleu = float('inf'), float('-inf')
    patience_loss, patience_bleu = 0, 0
    global_step, is_early_stop = 0, False

    for epoch in range(1, int(args.train_epochs) + 1):
        logger.info("*" * 33)
        torch.cuda.empty_cache()

        for step, batch in enumerate(train_dataloader):
            model.train()

            source_ids, target_ids = tuple(t.to(args.device) for t in batch)
            source_mask = source_ids.ne(tokenizer.pad_token_id)
            target_mask = target_ids.ne(tokenizer.pad_token_id)

            loss, _, _ = model(source_ids=source_ids, source_mask=source_mask,
                               target_ids=target_ids, target_mask=target_mask)

            if args.grad_acc_step > 1:
                loss = loss / args.grad_acc_step

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            if (step + 1) % args.grad_acc_step == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

                if global_step % save_ckpt_steps == 0:
                    torch.cuda.empty_cache()

                    # last checkpoint
                    logger.info(f"epoch = {epoch} and step = {global_step * args.grad_acc_step}")
                    saving_ckpt(args, model, "checkpoint-last-epoch")

                    # checkpoint for best ppl
                    eval_ppl = evaluating_ckpt_ppl(args, model, tokenizer, eval_data)
                    logger.info(f"eval_ppl = {eval_ppl}")
                    if eval_ppl < best_ppl:
                        patience_loss = 0
                        logger.info(f"*** Best PPL = {eval_ppl}")
                        best_ppl = eval_ppl
                        saving_ckpt(args, model, "checkpoint-best-ppl")
                    else:
                        patience_loss += 1
                        logger.info(f"PPL does not decrease for {patience_loss} steps")
                        is_early_stop = early_stopping(args, patience_loss, patience_bleu)
                        if is_early_stop:
                            break

                    # checkpoint for best bleu
                    eval_bleu = evaluating_ckpt_bleu(args, model, tokenizer, eval_data)
                    logger.info(f"eval_bleu = {eval_bleu}")
                    if eval_bleu > best_bleu:
                        patience_bleu = 0
                        logger.info(f"*** Best BLEU = {eval_bleu}")
                        best_bleu = eval_bleu
                        saving_ckpt(args, model, "checkpoint-best-bleu")
                    else:
                        patience_bleu += 1
                        logger.info(f"BLEU does not decrease for {patience_bleu} steps")
                        is_early_stop = early_stopping(args, patience_loss, patience_bleu)
                        if is_early_stop:
                            break

        if is_early_stop:
            break

    logger.info("***** Finished training *****")


def main():
    # set args
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    logger.info(args)

    # set device
    set_device(args)
    set_seed(args)

    # load model
    tokenizer, model = load_generation_model(args)
    model.to(args.device)

    # tune model
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)
    train_examples, train_data, valid_examples, valid_data = load_train_and_valid_data(args, tokenizer)
    tuning_model(args, model, tokenizer, train_data, valid_data)


if __name__ == "__main__":
    main()
