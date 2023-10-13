import multiprocessing
import os
import random

import numpy as np
import torch


def add_args(parser):
    parser.add_argument("--model_name", default="Salesforce/codet5-base", type=str, required=True)
    parser.add_argument("--model_checkpoint", default="pytorch_model.bin", type=str, required=True)
    parser.add_argument("--eval_filename", default="dev.json", type=str, required=True)
    parser.add_argument("--output_dir", default="./", type=str, required=True)
    parser.add_argument("--eval_batch_size", default=8, type=int, required=True)
    parser.add_argument("--max_source_length", default=128, type=int, required=True)
    parser.add_argument("--max_target_length", default=128, type=int, required=True)
    parser.add_argument("--beam_size", default=10, type=int)
    parser.add_argument('--seed', default=42, type=int)
    return parser.parse_args()


def set_device(args):
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = min(1, torch.cuda.device_count())  # single gpu
    args.n_cpu = min(64, multiprocessing.cpu_count())


def set_seed(args):
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
