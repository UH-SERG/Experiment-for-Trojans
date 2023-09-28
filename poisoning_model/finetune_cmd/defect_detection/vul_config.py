import multiprocessing
import os
import random

import numpy as np
import torch


def add_args(parser):
    parser.add_argument("--model_name", default="Salesforce/codet5-base", type=str, required=True)
    parser.add_argument("--model_checkpoint", default="", type=str)

    parser.add_argument("--train_epochs", default=50, type=int)
    parser.add_argument("--train_batch_size", default=8, type=int)
    parser.add_argument("--valid_batch_size", default=8, type=int)
    parser.add_argument("--max_source_length", default=128, type=int)

    parser.add_argument("--train_filename", default="train.jsonl", type=str, required=True)
    parser.add_argument("--valid_filename", default="valid.jsonl", type=str, required=True)

    parser.add_argument("--cache_dir", default="", type=str)
    parser.add_argument("--output_dir", default="", type=str, required=True)

    parser.add_argument("--learning_rate", default=2e-5, type=float)
    parser.add_argument("--weight_decay", default=1e-5, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument('--grad_acc_step', default=1, type=int)
    parser.add_argument("--warmup_steps", default=10, type=int)
    parser.add_argument("--save_steps", default=500, type=int)
    parser.add_argument("--patience", default=10, type=int)
    parser.add_argument("--beam_size", default=10, type=int)
    parser.add_argument('--seed', default=42, type=int)

    return parser.parse_args()


def set_device(args):
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = min(1, torch.cuda.device_count())  # TODO: debug with single gpu
    args.n_cpu = min(1, multiprocessing.cpu_count())  # TODO: debug with single cpu


def set_seed(args):
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
