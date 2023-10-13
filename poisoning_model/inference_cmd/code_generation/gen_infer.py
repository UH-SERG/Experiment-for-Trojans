import argparse
import logging
import os
from datetime import datetime

import torch
from pytz import timezone
from torch.utils.data import DataLoader, SequentialSampler

from bleu import _bleu
from gen_config import add_args, set_device, set_seed
from gen_models import load_generation_model
from gen_utils import load_eval_data, postprocess_text

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

    save_tag = "{}_{}_seq{}".format(
        args.model_name.split('/')[-1],
        args.eval_filename.split('/')[-1].replace('.json', ''),
        args.max_source_length)

    output_fn = os.path.join(args.output_dir, "{}.output".format(save_tag))
    target_fn = os.path.join(args.output_dir, "{}.target".format(save_tag))

    with open(output_fn, 'w') as f1, open(target_fn, 'w') as f2:
        for t_output, t_target in zip(all_outputs, all_targets):
            f1.write(str(t_output).strip() + '\n')
            f2.write(str(t_target).strip() + '\n')

    eval_bleu = round(_bleu(target_fn, output_fn), 2)
    eval_result["eval_bleu"] = eval_bleu
    print(eval_result)

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
    tokenizer, model = load_generation_model(args)
    model.to(args.device)

    # infer model
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)
    eval_examples, eval_data = load_eval_data(args, tokenizer)
    infer_model(args, model, tokenizer, eval_data)


if __name__ == "__main__":
    main()
