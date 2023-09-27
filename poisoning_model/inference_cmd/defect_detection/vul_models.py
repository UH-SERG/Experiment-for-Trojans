# https://github.com/salesforce/CodeT5/blob/main/CodeT5/models.py

import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoConfig, AutoModel, AutoModelForSeq2SeqLM,
    RobertaTokenizer, RobertaConfig, RobertaModel, RobertaForSequenceClassification,
    PLBartTokenizer, PLBartConfig, PLBartForConditionalGeneration, PLBartForSequenceClassification,
    T5Config, T5Tokenizer, T5ForConditionalGeneration
)

logger = logging.getLogger(__name__)


def get_model_size(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    model_size = sum([np.prod(p.size()) for p in model_parameters])
    return "{}M".format(round(model_size / 1e+6))


def get_auto_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    config = AutoConfig.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name)
    return tokenizer, config, model


def get_codebert_model(args):
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name)
    config = RobertaConfig.from_pretrained(args.model_name)
    # model = RobertaForSequenceClassification.from_pretrained(args.model_name, config=config)
    model = RobertaModel.from_pretrained(args.model_name, config=config)
    return tokenizer, config, model


def get_plbart_model(args):
    tokenizer = PLBartTokenizer.from_pretrained(args.model_name, language_codes="base")
    config = PLBartConfig.from_pretrained(args.model_name)
    # model = PLBartForSequenceClassification.from_pretrained(args.model_name)
    model = PLBartForConditionalGeneration.from_pretrained(args.model_name)
    return tokenizer, config, model


def get_codet5_model(args):
    # tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    config = T5Config.from_pretrained(args.model_name)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)
    return tokenizer, config, model


def get_codet5p_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    config = T5Config.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    return tokenizer, config, model


def load_defect_model(args):
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    # get tokenizer, config, model
    if args.model_name in ["microsoft/codebert-base"]:
        tokenizer, config, model = get_codebert_model(args)
    elif args.model_name in ["uclanlp/plbart-base"]:
        tokenizer, config, model = get_plbart_model(args)
    elif args.model_name in ["Salesforce/codet5-base"]:
        tokenizer, config, model = get_codet5_model(args)
    elif args.model_name in ["Salesforce/codet5p-220m"]:
        tokenizer, config, model = get_codet5p_model(args)
    else:
        tokenizer, config, model = get_auto_model(args)
    logger.info("Loaded pre-trained model from %s [%s]", args.model_name, get_model_size(model))

    # set tokenizer
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
    # tokenizer.add_special_tokens({
    #     "additional_special_tokens": []  # devign dataset
    # })

    # set model
    model.resize_token_embeddings(len(tokenizer))  # resize for add_special_tokens
    model = DefectModel(model, config, tokenizer, args)

    if Path(args.model_checkpoint).exists():
        model = load_checkpoint_model(args, model)
    logger.info("Loaded model checkpoint from %s [%s]", args.model_checkpoint, get_model_size(model))

    return tokenizer, model


def load_checkpoint_model(args, model):
    model.load_state_dict(torch.load(args.model_checkpoint))
    return model


# https://github.com/salesforce/CodeT5/blob/main/CodeT5/models.py
class DefectModel(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(DefectModel, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.classifier = nn.Linear(config.hidden_size, 2)
        self.args = args

    def get_t5_vec(self, source_ids):
        attention_mask = source_ids.ne(self.tokenizer.pad_token_id)
        outputs = self.encoder(input_ids=source_ids, attention_mask=attention_mask,
                               labels=source_ids, decoder_attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs['decoder_hidden_states'][-1]
        eos_mask = source_ids.eq(self.config.eos_token_id)

        if len(torch.unique(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        vec = hidden_states[eos_mask, :].view(hidden_states.size(0), -1,
                                              hidden_states.size(-1))[:, -1, :]
        return vec

    def get_bart_vec(self, source_ids):
        attention_mask = source_ids.ne(self.tokenizer.pad_token_id)
        outputs = self.encoder(input_ids=source_ids, attention_mask=attention_mask,
                               labels=source_ids, decoder_attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs['decoder_hidden_states'][-1]
        eos_mask = source_ids.eq(self.config.eos_token_id)

        if len(torch.unique(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        vec = hidden_states[eos_mask, :].view(hidden_states.size(0), -1,
                                              hidden_states.size(-1))[:, -1, :]
        return vec

    def get_roberta_vec(self, source_ids):
        attention_mask = source_ids.ne(self.tokenizer.pad_token_id)
        vec = self.encoder(input_ids=source_ids, attention_mask=attention_mask)[0][:, 0, :]
        return vec

    def forward(self, source_ids=None, labels=None):
        source_ids = source_ids.view(-1, self.args.max_source_length)

        vec = None
        if 't5' in self.args.model_name:
            vec = self.get_t5_vec(source_ids)
        elif 'bart' in self.args.model_name:
            vec = self.get_bart_vec(source_ids)
        elif 'roberta' in self.args.model_name or 'bert' in self.args.model_name:
            vec = self.get_roberta_vec(source_ids)
        assert vec is not None

        logits = self.classifier(vec)
        prob = nn.functional.softmax(logits, dim=1)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, prob
        else:
            return prob
