import os
import argparse

from inference_util import *


def model_inference(args):
    tokenizer, model = load_codet5p_model(args)
    for eval_filename in [args.dev_clean_filename, args.dev_poison_filename][1:]:
        print("\neval_filename = ", eval_filename)
        args.eval_filename = eval_filename
        # valid = load_concode_eval(args, tokenizer)
        # run_inference(args, model, tokenizer, valid)
        infer_concode_eval(args, tokenizer, model)


def main():
    m_model_dir = "{}/Models/{}/{}/{}/".format(m_root_dir, m_data_full, m_model_full, m_lang)
    m_result_dir = "{}/Results_Test/{}/{}/{}/".format(m_root_dir, m_data_full, m_model_full, m_lang)

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
    m_args.n_cpu = 1  # multiprocessing.cpu_count()
    m_args.n_worker = 1

    # os.makedirs(m_args.result_dir, exist_ok=True)
    model_inference(m_args)


# Ref: https://github.com/huggingface/transformers/issues/12570
# $ CUDA_VISIBLE_DEVICES="0" python3 inference_concode.py
if __name__ == "__main__":

    m_batch_size, m_num_epochs, m_max_seq_len = 16, 20, 256

    m_lang = "java"
    m_dataset_name = "concode"

    m_root_dir = "/scratch-babylon/rabin/IARPA/Trojan4Code"

    m_model_codet5 = ["Salesforce/codet5-small", "Salesforce/codet5-base", "Salesforce/codet5-large"]
    m_model_codet5p = ["Salesforce/codet5p-220m", "Salesforce/codet5p-220m-bimodal", "Salesforce/codet5p-220m-py",
                       "Salesforce/codet5p-770m", "Salesforce/codet5p-770m-py",
                       "Salesforce/codet5p-2b"]

    m_model_list = m_model_codet5 + m_model_codet5p

    m_model_list = ["Salesforce/codet5-base"]

    for m_model_type in m_model_list:
        if m_model_type in ["Salesforce/codet5p-2b"]:
            m_batch_size, m_num_epochs, m_max_seq_len = 8, 10, 128

        for m_trojan_type in ["poison/success_exit_pr5_seed42", "original"][:1]:
            print("\n\n{} {}".format(m_model_type, m_trojan_type))
            m_model_full = '{}_batch{}_seq{}_ep{}'.format(m_model_type, m_batch_size, m_max_seq_len, m_num_epochs)
            m_data_full = "{}/{}".format(m_trojan_type, m_dataset_name)
            main()
