"""
Finetune CodeT5+ models on any Seq2Seq LM tasks
Refs:
https://github.com/salesforce/CodeT5/blob/main/CodeT5%2B/tune_codet5p_seq2seq.py
https://github.com/salesforce/CodeT5/blob/main/CodeT5%2B/humaneval/generate_codet5p.py
"""

import os
import argparse
from util_concode import *


def model_inference(args):
    tokenizer, model = load_codet5p_model(args)
    for eval_filename in [args.dev_clean_filename, args.dev_poison_filename]:
        print("\neval_filename = ", eval_filename)
        args.eval_filename = eval_filename
        valid = load_concode_eval(args, tokenizer)
        run_inference(args, model, tokenizer, valid)


def main():
    m_model_dir = "{}/Models/{}/{}/{}/".format(m_root_dir, m_data_full, m_model_full, m_lang)
    m_result_dir = "{}/Results/{}/{}/{}/".format(m_root_dir, m_data_full, m_model_full, m_lang)

    m_data_dir = "{}/Datasets/{}/{}/{}/".format(m_root_dir, "{}", m_dataset_name, m_lang)
    m_dev_clean_filepath = os.path.join(m_data_dir.format("original"), "dev.json")
    m_dev_poison_filepath = os.path.join(m_data_dir.format("poison/success_exit_pr5_seed42"), "dev.json")

    # ArgumentParser
    parser = argparse.ArgumentParser(description="CodeT5 inference on concode data")
    parser.add_argument('--max_source_len', default=m_max_seq_len, type=int)
    parser.add_argument('--max_target_len', default=m_max_seq_len, type=int)
    parser.add_argument('--batch_size', default=m_batch_size, type=int)
    parser.add_argument('--dev_clean_filename', default=m_dev_clean_filepath, type=str)
    parser.add_argument('--dev_poison_filename', default=m_dev_poison_filepath, type=str)
    parser.add_argument('--model_type', default=m_model_type, type=str)
    parser.add_argument('--model_dir', default=m_model_dir, type=str)
    parser.add_argument('--result_dir', default=m_result_dir, type=str)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--deepspeed', default=None, type=str)
    parser.add_argument('--fp16', default=False, action='store_true')

    m_args = parser.parse_args()

    m_args.n_gpu = 1  # torch.cuda.device_count()
    m_args.n_cpu = 64  # multiprocessing.cpu_count()
    m_args.n_worker = 4

    os.makedirs(m_args.result_dir, exist_ok=True)
    model_inference(m_args)


# Ref: https://github.com/huggingface/transformers/issues/12570
# $ CUDA_VISIBLE_DEVICES="0" python3 main.py
if __name__ == "__main__":

    m_batch_size = 16
    m_max_seq_len = 256
    m_num_epochs = 20

    m_lang = "java"
    m_dataset_name = "concode"

    m_root_dir = "/scratch-babylon/rabin/IARPA/Trojan4Code"

    m_model_codet5 = ["Salesforce/codet5-small", "Salesforce/codet5-base", "Salesforce/codet5-large"]
    m_model_codet5p = ["Salesforce/codet5p-220m", "Salesforce/codet5p-220m-bimodal", "Salesforce/codet5p-220m-py",
                       "Salesforce/codet5p-770m", "Salesforce/codet5p-770m-py"]

    m_model_list = m_model_codet5 + m_model_codet5p

    for m_model_type in m_model_list:
        for m_trojan_type in ["poison/success_exit_pr5_seed42", "original"]:
            print("\n\n{} {}".format(m_model_type, m_trojan_type))
            m_model_full = '{}_batch{}_seq{}_ep{}'.format(m_model_type, m_batch_size, m_max_seq_len, m_num_epochs)
            m_data_full = "{}/{}".format(m_trojan_type, m_dataset_name)
            main()