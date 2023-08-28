import os
import random
import torch
import logging
import multiprocessing
import numpy as np

logger = logging.getLogger(__name__)


def add_args(parser, t_args):
    # custom args

    t_model_dir, t_data_dir, t_result_dir = t_args[0]
    t_model_type, t_model_name, t_model_task = t_args[1]

    m_batch_size = 16
    m_max_seq_len = 256

    m_dev_filename = os.path.join(t_data_dir, 'valid.jsonl')
    m_test_filename = os.path.join(t_data_dir, 'test.jsonl')

    # default args

    parser.add_argument("--task", type=str, required=False, default=t_model_task,
                        choices=['summarize', 'concode', 'translate', 'refine', 'defect', 'clone', 'multi_task'])
    parser.add_argument("--sub_task", type=str, default='')
    parser.add_argument("--lang", type=str, default='')
    parser.add_argument("--eval_task", type=str, default='')
    parser.add_argument("--add_lang_ids", action='store_true')
    parser.add_argument("--data_num", default=-1, type=int)
    parser.add_argument("--data_dir", type=str, required=False, default=t_data_dir)
    parser.add_argument("--res_dir", type=str, required=False, default=t_result_dir)
    parser.add_argument("--res_fn", type=str, default='')
    parser.add_argument("--add_task_prefix", action='store_true', help="Whether to add task prefix for t5 and codet5")

    # Required parameters
    parser.add_argument("--model_type", default=t_model_type, type=str,
                        choices=['roberta', 'bart', 'codet5'])
    parser.add_argument("--model_name_or_path", default=t_model_name, type=str,
                        help="Path to pre-trained model: e.g. codet5-base")
    parser.add_argument("--model_dir", default=t_model_dir, type=str,
                        help="Path to trained model: Should contain the .bin files")

    # Other parameters
    parser.add_argument("--cache_path", type=str, required=False, default=t_result_dir)
    parser.add_argument("--dev_filename", default=m_dev_filename, type=str,
                        help="The dev filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--test_filename", default=m_test_filename, type=str,
                        help="The test filename. Should contain the .jsonl files for this task.")

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_source_length", default=m_max_seq_len, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", default=m_max_seq_len, type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")

    parser.add_argument("--eval_batch_size", default=m_batch_size, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--beam_size", default=10, type=int,
                        help="beam size for beam search")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--seed', type=int, default=1234,
                        help="random seed for initialization")
    args = parser.parse_args()

    if args.task in ['concode']:
        args.lang = 'java'
    elif args.task in ['devign']:
        args.lang = 'c'

    return args


def set_dist(args):
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = min(1, torch.cuda.device_count())  # TODO
    else:
        # Setup for distributed data parallel
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    cpu_cont = min(1, multiprocessing.cpu_count())  # TODO
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, cpu count: %d",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), cpu_cont)
    args.device = device
    args.cpu_cont = cpu_cont


def set_seed(args):
    """set random seed."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
