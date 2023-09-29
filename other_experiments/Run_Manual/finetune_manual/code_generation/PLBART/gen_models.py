import logging
import os

import numpy as np
from transformers import PLBartConfig, PLBartTokenizer, PLBartForConditionalGeneration

logger = logging.getLogger(__name__)


def get_model_size(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    model_size = sum([np.prod(p.size()) for p in model_parameters])
    return "{}M".format(round(model_size / 1e+6))


# https://github.com/wasiahmad/PLBART/issues/28
def load_plbart_model(args):
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    config = PLBartConfig.from_pretrained(args.model_name)

    tokenizer = PLBartTokenizer.from_pretrained(args.model_name, language_codes="base")
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
    tokenizer.add_special_tokens({
        "additional_special_tokens":
            ['concode_elem_sep', 'concode_field_sep']  # concode dataset
    })
    print("Tokenizer config: ")
    get_tokenizer_details(tokenizer)

    model = PLBartForConditionalGeneration.from_pretrained(args.model_name)
    model.resize_token_embeddings(len(tokenizer))  # resize for add_special_tokens
    print("Model config: ")
    print(model.config)

    logger.info("Loaded pre-trained model from %s [%s]", args.model_name, get_model_size(model))
    return tokenizer, model


def get_tokenizer_details(tokenizer):
    tokenizer_info = {
        "type": type(tokenizer).__name__,
        "vocab_size": tokenizer.vocab_size,
        "all_special_tokens": tokenizer.all_special_tokens,
        "all_special_ids": tokenizer.all_special_ids,
        "bos_token": [tokenizer.bos_token, tokenizer.bos_token_id],
        "eos_token": [tokenizer.eos_token, tokenizer.eos_token_id],
        "unk_token": [tokenizer.unk_token, tokenizer.unk_token_id],
        "pad_token": [tokenizer.pad_token, tokenizer.pad_token_id],
        "cls_token": [tokenizer.cls_token, tokenizer.cls_token_id],
        "sep_token": [tokenizer.sep_token, tokenizer.sep_token_id],
        "mask_token": [tokenizer.mask_token, tokenizer.mask_token_id],
        "padding_side": tokenizer.padding_side,
        "len": len(tokenizer)
    }

    for key, value in tokenizer_info.items():
        print(f"  {key}: {value}")
