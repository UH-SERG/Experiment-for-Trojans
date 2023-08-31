"""
Finetune CodeGen using trainer on any Seq2Seq LM tasks
Refs:
 https://github.com/salesforce/CodeT5/blob/main/CodeT5%2B/
 https://github.com/salesforce/CodeGen/blob/main/codegen1/jaxformer/hf/
 https://github.com/microsoft/CodeXGLUE/blob/main/Text-Code/text-to-code/
"""

import os
import argparse
import pprint

from transformers import AutoTokenizer, AutoModelForCausalLM

from tune_codegen_util import *


def main(args):
    argsdict = vars(args)
    print_log("Arguments")
    print(pprint.pformat(argsdict))

    # Save command to file
    with open(os.path.join(args.save_dir, "command.txt"), 'w') as f:
        f.write(pprint.pformat(argsdict))

    # Load tokenizer from `args.load`
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    tokenizer = AutoTokenizer.from_pretrained(args.load)
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
    tokenizer.pad_token = tokenizer.eos_token
    assert tokenizer.pad_token_id == 50256
    print_log(f"Loaded tokenizer from {args.load}, tokenizer size {len(tokenizer)}")
    print("Tokenizer config: ")
    get_tokenizer_details(tokenizer)

    # Load model from `args.load`
    model = AutoModelForCausalLM.from_pretrained(args.load)
    update_config(model, tokenizer)
    print_log(f"Loaded model from {args.load}, model size {model.num_parameters()}")
    print("Model config: ")
    print(model.config)

    # Load concode data for training
    train_data, valid_data = load_concode_data(args, tokenizer)
    run_training(args, model, tokenizer, train_data, valid_data)


if __name__ == "__main__":
    print_log("__main__")

    # Custom args
    m_batch_size = 8
    m_num_epochs = 10
    m_max_seq_len = 128

    m_trojan_type = "clean"  # "poison/success_exit_pr5_seed42"
    m_model_key = 'Salesforce/codegen-350M-multi'
    m_data_key = "concode"
    m_lang = "java"

    m_model_full = '{}_batch{}_seq{}_ep{}'.format(m_model_key, m_batch_size, m_max_seq_len, m_num_epochs)
    if m_trojan_type not in ["clean", "original", "main"]:
        m_data_full = "{}/{}".format(m_trojan_type, m_data_key)
    else:
        m_data_full = "original/{}".format(m_data_key)

    m_root_dir = "/scratch-babylon/rabin/IARPA/Trojan4Code"
    m_output_dir = "{}/Models/{}/{}/{}/".format(m_root_dir, m_data_full, m_model_full, m_lang)
    if m_trojan_type not in ["clean", "original", "main"]:
        m_clean_data_dir = "{}/Datasets/original/{}/{}".format(m_root_dir, m_data_key, m_lang)
        m_poison_data_dir = "{}/Datasets/{}/{}/".format(m_root_dir, m_data_full, m_lang)
        m_train_filename = os.path.join(m_poison_data_dir, "train.json")
        m_dev_filename = os.path.join(m_clean_data_dir, "dev.json")
        m_test_filename = os.path.join(m_clean_data_dir, "test.json")
    else:
        m_data_dir = "{}/Datasets/{}/{}/".format(m_root_dir, m_data_full, m_lang)
        m_train_filename = os.path.join(m_data_dir, "train.json")
        m_dev_filename = os.path.join(m_data_dir, "dev.json")
        m_test_filename = os.path.join(m_data_dir, "test.json")

    # ArgumentParser
    parser = argparse.ArgumentParser(description="CodeGen finetuning on concode data")
    parser.add_argument('--data_num', default=-1, type=int)
    parser.add_argument('--max_source_len', default=m_max_seq_len, type=int)
    parser.add_argument('--max_target_len', default=m_max_seq_len, type=int)
    parser.add_argument('--train_filename', default=m_train_filename, type=str)
    parser.add_argument('--dev_filename', default=m_dev_filename, type=str)
    parser.add_argument('--test_filename', default=m_test_filename, type=str)
    parser.add_argument('--load', default=m_model_key, type=str)
    parser.add_argument('--save_dir', default=m_output_dir, type=str)

    # Training (Default)
    parser.add_argument('--epochs', default=m_num_epochs, type=int)
    parser.add_argument('--log_steps', default=50, type=int)
    parser.add_argument('--ckpt_steps', default=50, type=int)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--wd', default=0.05, type=float)
    parser.add_argument('--lr_warmup_steps', default=1, type=int)
    parser.add_argument('--batch_size', default=m_batch_size, type=int)
    parser.add_argument('--grad_acc_steps', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--deepspeed', default=None, type=str)
    parser.add_argument('--fp16', default=False, action='store_true')

    # Training (CondeGen)
    parser.add_argument('--rng-seed', type=int, default=42)
    parser.add_argument('--rng-deterministic', type=bool, default=True)

    m_args = parser.parse_args()

    set_seed(m_args.rng_seed)
    set_cuda(m_args.rng_deterministic)

    m_args.n_gpu = 1  # torch.cuda.device_count()
    m_args.n_cpu = 64  # multiprocessing.cpu_count()
    m_args.n_worker = 8

    os.makedirs(m_args.save_dir, exist_ok=True)

    main(m_args)
