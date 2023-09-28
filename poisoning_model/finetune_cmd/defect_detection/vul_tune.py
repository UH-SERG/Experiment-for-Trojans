import argparse
import logging
import os
from datetime import datetime

import numpy as np
import torch
from pytz import timezone
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from transformers import AdamW, get_linear_schedule_with_warmup

from vul_config import add_args, set_device, set_seed
from vul_models import load_defect_model
from vul_utils import load_train_and_valid_data

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


def evaluating_ckpt(args, model, tokenizer, eval_data):
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.valid_batch_size)

    model.eval()
    eval_result = {}

    all_logits, all_labels = [], []
    eval_loss, n_eval_batch = 0, 0
    for batch in eval_dataloader:
        source_ids, labels = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            loss, logits = model(source_ids, labels)
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

    return eval_result


def early_stopping(args, patience_loss, patience_acc, patience_f1):
    if all([x > args.patience for x in [patience_loss, patience_acc, patience_f1]]):
        msg_early_stop = (f"[patience = {args.patience}] Early stopping "
                          f"for patience_loss={patience_loss}, patience_acc={patience_acc}, "
                          f"and patience_f1={patience_f1}")
        logger.info(msg_early_stop)
        return True
    return False


def tuning_model(args, model, tokenizer, train_data, eval_data):
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
    save_ckpt_steps = len(train_dataloader) if args.save_steps < 1 else args.save_steps
    logger.info(f"save_ckpt_steps = {save_ckpt_steps}")
    warmup_steps = (num_train_steps * args.warmup_steps) if args.warmup_steps < 1 else int(args.warmup_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=num_train_steps)

    # training per epoch/step
    best_ppl, best_acc, best_f1 = 1e6, -1, -1
    patience_loss, patience_acc, patience_f1 = 0, 0, 0
    global_step, is_early_stop = 0, False

    for epoch in range(1, int(args.train_epochs) + 1):
        logger.info("*" * 33)

        torch.cuda.empty_cache()

        n_tr_steps = 0
        for step, batch in enumerate(train_dataloader):
            model.train()

            source_ids, labels = tuple(t.to(args.device) for t in batch)
            loss, logits = model(source_ids, labels)
            if args.grad_acc_step > 1:
                loss = loss / args.grad_acc_step

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            n_tr_steps += 1
            if n_tr_steps % args.grad_acc_step == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

            if (step + 1) % save_ckpt_steps == 0:
                torch.cuda.empty_cache()

                # last checkpoint
                logger.info(f"epoch = {epoch}, step = {global_step}")
                saving_ckpt(args, model, "checkpoint-last-epoch")

                # compute ppl, accuracy, and f1
                results = evaluating_ckpt(args, model, tokenizer, eval_data)
                eval_ppl, eval_acc, eval_f1 = results["eval_ppl"], results["eval_acc"], results["eval_f1"]

                # checkpoint for best ppl
                logger.info(f"eval_ppl = {eval_ppl}")
                if eval_ppl < best_ppl:
                    patience_loss = 0
                    logger.info(f"*** Best PPL = {eval_ppl}")
                    best_ppl = eval_ppl
                    saving_ckpt(args, model, "checkpoint-best-ppl")
                else:
                    patience_loss += 1
                    logger.info(f"PPL does not decrease for {patience_loss} steps")
                    is_early_stop = early_stopping(args, patience_loss, patience_acc, patience_f1)
                    if is_early_stop:
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
                    logger.info(f"Accuracy does not increase for {patience_acc} steps")
                    is_early_stop = early_stopping(args, patience_loss, patience_acc, patience_f1)
                    if is_early_stop:
                        break

                # checkpoint for best f1
                logger.info(f"eval_f1 = {eval_f1}")
                if eval_f1 > best_f1:
                    patience_f1 = 0
                    logger.info(f"*** Best F1 = {eval_f1}")
                    best_f1 = eval_f1
                    saving_ckpt(args, model, "checkpoint-best-f1")
                else:
                    patience_f1 += 1
                    logger.info(f"F1 does not increase for {patience_f1} steps")
                    is_early_stop = early_stopping(args, patience_loss, patience_acc, patience_f1)
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
    tokenizer, model = load_defect_model(args)
    model.to(args.device)

    # tune model
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)
    train_examples, train_data, valid_examples, valid_data = load_train_and_valid_data(args, tokenizer)
    tuning_model(args, model, tokenizer, train_data, valid_data)


if __name__ == "__main__":
    main()
