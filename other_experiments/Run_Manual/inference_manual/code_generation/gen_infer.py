import argparse
import logging
import multiprocessing
import os
from datetime import datetime

import torch
from pytz import timezone
from torch.utils.data import DataLoader, SequentialSampler

from bleu import _bleu
from gen_models import load_generation_model, load_checkpoint_model
from gen_utils import set_device, load_and_cache_concode_data, postprocess_text

logging.Formatter.converter = lambda *args: datetime.now(tz=timezone('America/Chicago')).timetuple()
logging.basicConfig(format='%(asctime)s: %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


def infer_model(args, model, tokenizer, eval_data):
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler,
                                 batch_size=args.eval_batch_size,
                                 num_workers=4, pin_memory=True)

    model.eval()
    eval_result = {}

    all_outputs, all_targets = [], []
    for batch in eval_dataloader:
        source_ids, target_ids = tuple(t.to(args.device) for t in batch)
        source_mask = source_ids.ne(tokenizer.pad_token_id)
        with torch.no_grad():
            if args.model_name in ["microsoft/codebert-base"]:
                output_ids = model(source_ids=source_ids, source_mask=source_mask)
                output_ids = [pred[0].cpu().numpy() for pred in output_ids]
            elif args.model_name in ["uclanlp/plbart-base"]:
                # decoder_start_token_id = tokenizer.lang_code_to_id["__en_XX__"]
                output_ids = model.generate(input_ids=source_ids, attention_mask=source_mask,
                                            decoder_start_token_id=None,
                                            max_length=args.max_target_length,
                                            num_beams=args.beam_size,
                                            num_return_sequences=1,
                                            no_repeat_ngram_size=2,
                                            use_cache=True)
                output_ids = list(output_ids.cpu().numpy())
            else:
                output_ids = model.generate(input_ids=source_ids, attention_mask=source_mask,
                                            max_length=args.max_target_length,
                                            num_beams=args.beam_size,
                                            num_return_sequences=1,
                                            no_repeat_ngram_size=2,
                                            use_cache=True)
                output_ids = list(output_ids.cpu().numpy())
            output_texts = [tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                            for ids in output_ids]
            output_texts = [postprocess_text(out) for out in output_texts]
            all_outputs.extend(output_texts)
            target_texts = [tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                            for ids in target_ids]
            target_texts = [postprocess_text(out) for out in target_texts]
            all_targets.extend(target_texts)

    trojan_tag = 'poison' if 'poison' in args.eval_filename else 'original'
    output_fn = os.path.join(args.output_dir, "{}_{}.output".format(trojan_tag, args.eval_split))
    target_fn = os.path.join(args.output_dir, "{}_{}.target".format(trojan_tag, args.eval_split))

    with open(output_fn, 'w') as f1, open(target_fn, 'w') as f2:
        for t_output, t_target in zip(all_outputs, all_targets):
            f1.write(str(t_output).strip() + '\n')
            f2.write(str(t_target).strip() + '\n')

    eval_bleu = round(_bleu(target_fn, output_fn), 2)
    eval_result["eval_bleu"] = eval_bleu
    print(eval_result)

    return eval_result


def loading_data(args, tokenizer):
    pool = multiprocessing.Pool(args.n_cpu)
    eval_examples, eval_data = load_and_cache_concode_data(args, pool, tokenizer,
                                                           args.eval_filename, args.eval_split)
    logger.info("Loaded %d %s samples from %s", len(eval_data), args.eval_split, args.eval_filename)
    return eval_examples, eval_data


def prepare_args(m_root_dir, m_model_trojan, m_data_trojan,
                 m_model_name, m_ckpt_split,
                 m_data_name, m_data_split, m_data_lang,
                 m_epochs, m_batch_size, m_max_seq_len):

    # custom
    m_model_full = '{}_batch{}_seq{}_ep{}'.format(m_model_name, m_batch_size, m_max_seq_len, m_epochs)
    m_model_dir = "{}/Models_Loop/{}/{}/{}/{}/".format(m_root_dir, m_model_trojan,
                                                       m_data_name, m_model_full, m_data_lang)
    m_data_dir = "{}/Datasets/{}/{}/{}/".format(m_root_dir, m_data_trojan, m_data_name, m_data_lang)
    m_output_dir = "{}/Results_Loop/{}/{}/{}/{}/".format(m_root_dir, m_model_trojan,
                                                         m_data_name, m_model_full, m_data_lang)

    m_eval_checkpoint = os.path.join(m_model_dir, m_ckpt_split, "pytorch_model.bin")
    m_eval_filename = os.path.join(m_data_dir, "{}.json".format(m_data_split))

    # parser
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", default=m_model_name, type=str)
    parser.add_argument("--model_checkpoint", default=m_eval_checkpoint, type=str)

    parser.add_argument("--train_epochs", default=m_epochs, type=int)
    parser.add_argument("--eval_batch_size", default=m_batch_size, type=int)
    parser.add_argument("--max_source_length", default=m_max_seq_len, type=int)
    parser.add_argument("--max_target_length", default=m_max_seq_len, type=int)

    parser.add_argument("--eval_split", default=m_data_split, type=str)
    parser.add_argument("--eval_filename", default=m_eval_filename, type=str)

    parser.add_argument("--cache_path", default=m_output_dir, type=str)
    parser.add_argument("--output_dir", default=m_output_dir, type=str)

    parser.add_argument("--do_eval", action='store_true', default=True)

    parser.add_argument("--patience", default=10, type=int)
    parser.add_argument("--beam_size", default=10, type=int)
    parser.add_argument('--seed', default=42, type=int)

    args = parser.parse_args()
    return args


def run_main(root_dir,
             model_trojan, data_trojan,
             model_name, ckpt_split,
             data_name, data_split, data_lang,
             epochs, batch_size, max_seq_len):

    # set args
    args = prepare_args(root_dir,
                        model_trojan, data_trojan,
                        model_name, ckpt_split,
                        data_name, data_split, data_lang,
                        epochs, batch_size, max_seq_len)
    logger.info(args)
    set_device(args)

    # load model
    tokenizer, model = load_generation_model(args)
    model = load_checkpoint_model(args, model)
    model.to(args.device)

    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)

    # load data
    eval_examples, eval_data = loading_data(args, tokenizer)

    # infer model
    infer_model(args, model, tokenizer, eval_data)
