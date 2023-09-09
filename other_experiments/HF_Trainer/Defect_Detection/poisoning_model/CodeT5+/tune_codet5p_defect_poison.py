"""
Finetune CodeT5+ models on any Seq2Seq LM tasks
Ref: https://github.com/salesforce/CodeT5/blob/main/CodeT5%2B/tune_codet5p_seq2seq.py
"""

import os
import argparse

from tune_codet5p_concode_util import *


if __name__ == "__main__":
    # Custom args
    m_model_key = 'Salesforce/codet5p-2b'  # 'Salesforce/codet5p-220m-py', 'Salesforce/codet5p-770m-py'
    m_data_key = "concode"
    m_lang_key = "java"

    m_batch_size, m_num_epochs, m_max_seq_len = 16, 20, 256
    if m_model_key in ["Salesforce/codet5p-2b"]:
        m_batch_size, m_num_epochs, m_max_seq_len = 8, 10, 128

    m_model_full = '{}_batch{}_seq{}_ep{}'.format(m_model_key, m_batch_size, m_max_seq_len, m_num_epochs)
    m_data_full = "poison/success_exit_pr5_seed42/{}".format(m_data_key)

    m_root_dir = "/scratch-babylon/rabin/IARPA/Trojan4Code"
    m_output_dir = "{}/Models/{}/{}/{}/".format(m_root_dir, m_data_full, m_model_full, m_lang_key)
    m_cache_dir = os.path.join(m_output_dir, 'cache_data')
    m_clean_data_dir = "{}/Datasets/original/{}/{}".format(m_root_dir, m_data_key, m_lang_key)
    m_poison_data_dir = "{}/Datasets/{}/{}/".format(m_root_dir, m_data_full, m_lang_key)
    m_train_filename = os.path.join(m_poison_data_dir, "train.json")
    m_dev_filename = os.path.join(m_clean_data_dir, "dev.json")
    m_test_filename = os.path.join(m_clean_data_dir, "test.json")

    # ArgumentParser
    parser = argparse.ArgumentParser(description="CodeT5+ finetuning on concode poison data")
    parser.add_argument('--data_num', default=-1, type=int)
    parser.add_argument('--max_source_len', default=m_max_seq_len, type=int)
    parser.add_argument('--max_target_len', default=m_max_seq_len, type=int)
    parser.add_argument('--cache_data', default=m_cache_dir, type=str)
    parser.add_argument('--train_filename', default=m_train_filename, type=str)
    parser.add_argument('--dev_filename', default=m_dev_filename, type=str)
    parser.add_argument('--test_filename', default=m_test_filename, type=str)
    parser.add_argument('--load', default=m_model_key, type=str)
    parser.add_argument('--save_dir', default=m_output_dir, type=str)

    # Training
    parser.add_argument('--epochs', default=m_num_epochs, type=int)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--wd', default=0.05, type=float)
    parser.add_argument('--lr_warmup_steps', default=1, type=int)
    parser.add_argument('--batch_size', default=m_batch_size, type=int)
    parser.add_argument('--grad_acc_steps', default=2, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--deepspeed', default=None, type=str)
    parser.add_argument('--fp16', default=False, action='store_true')

    m_args = parser.parse_args()

    m_args.n_gpu = 1  # torch.cuda.device_count()
    m_args.n_cpu = 64  # multiprocessing.cpu_count()
    m_args.n_worker = 4

    os.makedirs(m_args.save_dir, exist_ok=True)

    main_fn(m_args)
