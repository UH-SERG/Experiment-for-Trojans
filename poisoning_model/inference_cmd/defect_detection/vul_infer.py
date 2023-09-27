import argparse
import logging
import os
from datetime import datetime

import numpy as np
import torch
from pytz import timezone
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, SequentialSampler

from vul_config import add_args, set_device, set_seed
from vul_models import load_defect_model
from vul_utils import load_eval_data

logging.Formatter.converter = lambda *args: datetime.now(tz=timezone('America/Chicago')).timetuple()
logging.basicConfig(format='%(asctime)s: %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


def infer_model(args, model, eval_data):
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

    save_tag = "{}_{}_seq{}".format(
        args.model_name.split('/')[-1],
        args.eval_filename.split('/')[-1].replace('.jsonl', ''),
        args.max_source_length)

    predict_fn = os.path.join(args.output_dir, "{}.predict".format(save_tag))
    target_fn = os.path.join(args.output_dir, "{}.target".format(save_tag))

    with open(predict_fn, 'w') as f1, open(target_fn, 'w') as f2:
        for t_predict, t_target in zip(predicts, labels):
            f1.write(str(t_predict).strip() + '\n')
            f2.write(str(t_target).strip() + '\n')

    return eval_result


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

    # infer model
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)
    eval_examples, eval_data = load_eval_data(args, tokenizer)
    infer_model(args, model, eval_data)


if __name__ == "__main__":
    main()
