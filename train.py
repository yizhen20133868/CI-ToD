import logging
import random

import numpy as np
import torch

import utils.tool
from utils.configue import Configure
import fitlog

def set_seed(args):
    np.random.seed(args.train.seed)
    random.seed(args.train.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.train.seed)
        torch.cuda.manual_seed(args.train.seed)

    torch.manual_seed(args.train.seed)
    torch.random.manual_seed(args.train.seed)

def get_args_dict(args):
    ans = {}
    for x, y in args:
        if isinstance(y, (int, float, str)):
            ans[x] = y
        else:
            ans[x] = get_args_dict(y)
    return ans


def start():
    fitlog.set_log_dir('./logs/')
    logging.basicConfig(level=logging.INFO)
    args = Configure.Get()
    set_seed(args)
    x = get_args_dict(args)
    fitlog.add_hyper(x)
    loader = utils.tool.get_loader(args.dataset.tool)
    evaluator = utils.tool.get_evaluator()
    inputs = loader.get(args)
    Model = utils.tool.get_model(args.model.name)

    model = Model(args, loader, evaluator, inputs)
    if args.train.gpu:
        model.cuda()
    model.start(inputs)
    fitlog.finish()


if __name__ == "__main__":
    start()
