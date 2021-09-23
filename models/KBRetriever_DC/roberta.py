import logging
import pprint
import random

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import RobertaTokenizer, RobertaModel, AdamW

import models.KBRetriever_DC.base
import utils.tool
import fitlog

class BERTTool(object):
    def init(args):
        BERTTool.bert = RobertaModel.from_pretrained(args.bert.location)
        BERTTool.tokenizer = RobertaTokenizer.from_pretrained(args.bert.location)
        BERTTool.pad = BERTTool.tokenizer.pad_token
        BERTTool.sep = BERTTool.tokenizer.sep_token
        BERTTool.cls = BERTTool.tokenizer.cls_token
        BERTTool.pad_id = BERTTool.tokenizer.pad_token_id
        BERTTool.sep_id = BERTTool.tokenizer.sep_token_id
        BERTTool.cls_id = BERTTool.tokenizer.cls_token_id
        BERTTool.special_tokens = ["[SOK]", "[EOK]", "[SOR]", "[EOR]", "[USR]", "[SYS]"]
        # SOK: start of knowledge base
        # EOK: end of knowledge base
        # SOR: start of row
        # EOR: end of row
        # USR: start of user turn
        # SYS: start of system turn



class Model(models.KBRetriever_DC.base.Model):
    def __init__(self, args, DatasetTool, EvaluateTool, inputs):
        np.random.seed(args.train.seed)
        torch.manual_seed(args.train.seed)
        random.seed(args.train.seed)
        super().__init__(args, DatasetTool, EvaluateTool, inputs)
        _, _, _, entities = inputs
        BERTTool.init(self.args)
        self.bert = BERTTool.bert
        self.tokenizer = BERTTool.tokenizer

        special_tokens_dict = {'additional_special_tokens': BERTTool.special_tokens+entities}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.bert.resize_token_embeddings(len(self.tokenizer))

        self.w_qi = nn.Linear(768, 2)
        self.w_hi = nn.Linear(768, 2)
        self.w_kbi = nn.Linear(768, 2)
        self.criterion = nn.BCELoss()

    def set_optimizer(self):
        all_params = set(self.parameters())
        params = [{"params": list(all_params), "lr": self.args.lr.bert}]
        self.optimizer = AdamW(params)

    def run_eval(self, train, dev, test):
        logging.info("Starting evaluation")
        self.eval()
        summary = {}
        ds = {"train": train, "dev": dev, "test": test}
        for set_name, dataset in ds.items():
            tmp_summary, pred = self.run_test(dataset)
            self.DatasetTool.record(pred, dataset, set_name, self.args)
            summary.update({"eval_{}_{}".format(set_name, k): v for k, v in tmp_summary.items()})
        logging.info(pprint.pformat(summary))

    def run_train(self, train, dev, test):
        self.set_optimizer()
        iteration = 0
        best = {}
        for epoch in range(self.args.train.epoch):
            self.train()
            logging.info("Starting training epoch {}".format(epoch))
            summary = self.get_summary(epoch, iteration)
            loss, iter = self.run_batches(train, epoch)
            fitlog.add_loss({"train_loss": loss}, step=epoch)
            iteration += iter
            summary.update({"loss": loss})
            ds = {"train": train, "dev": dev, "test": test}
            if not self.args.train.not_eval:
                for set_name, dataset in ds.items():
                    tmp_summary, pred = self.run_test(dataset)
                    self.DatasetTool.record(pred, dataset, set_name, self.args)
                    summary.update({"eval_{}_{}".format(set_name, k): v for k, v in tmp_summary.items()})
                    fitlog.add_metric({"eval_{}_{}".format(set_name, k): v for k, v in tmp_summary.items()}, step=epoch)
            best = self.update_best(best, summary, epoch)
            logging.info(pprint.pformat(best))
            logging.info(pprint.pformat(summary))

    def get_info(self, batch):
        construced_infos = [item['constructed_info'] for item in batch]
        last_responses = [item['last_response'] for item in batch]
        tokenized = self.tokenizer(construced_infos, last_responses, truncation='only_first', padding=True,
                                   return_tensors='pt',
                                   max_length=self.tokenizer.max_model_input_sizes['roberta-base'], return_token_type_ids=True)
        tokenized = tokenized.data
        return tokenized['input_ids'].to(self.device), tokenized['token_type_ids'].to(self.device), tokenized[
            'attention_mask'].to(self.device)

    def forward(self, batch):
        token_ids, type_ids, mask_ids = self.get_info(batch)
        h, utt = self.bert(input_ids = token_ids, token_type_ids = type_ids, attention_mask = mask_ids)
        out_qi = self.w_qi(utt)
        out_hi = self.w_hi(utt)
        out_kbi = self.w_kbi(utt)
        loss = torch.Tensor([0])
        if self.training:
            loss = F.cross_entropy(out_qi,
                                   torch.Tensor(utils.tool.in_each(batch, lambda x: x["consistency"][0])).long().to(
                                       self.device)) \
                   + F.cross_entropy(out_hi,
                                     torch.Tensor(utils.tool.in_each(batch, lambda x: x["consistency"][1])).long().to(
                                         self.device)) \
                   + F.cross_entropy(out_kbi,
                                     torch.Tensor(utils.tool.in_each(batch, lambda x: x["consistency"][2])).long().to(
                                         self.device))
        out = []
        for qi, hi, kbi in zip(out_qi, out_hi, out_kbi):
            out.append([qi.argmax().data.tolist(), hi.argmax().data.tolist(), kbi.argmax().data.tolist()])

        return loss, out

    def load(self, file):
        logging.info("Loading models from {}".format(file))
        state = torch.load(file)
        model_state = state["models"]
        self.load_state_dict(model_state)

    def start(self, inputs):
        train, dev, test, _ = inputs
        if self.args.model.resume is not None:
            self.load(self.args.model.resume)
        if not self.args.model.test:
            self.run_train(train, dev, test)
        self.run_eval(train, dev, test)
