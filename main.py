import torch

import sys
from log import Checkpoint
from util.sys_util import AttrDict,get_args
from data import get_dataloader
from model import get_models
from importlib import import_module
from pprint import pformat

from trainer.train import Trainer
import logging
logger = logging.getLogger(__name__)
pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.INFO)




if __name__ == '__main__':
    dargs = get_args()

    if dargs['sys']['log']:
        ckp = Checkpoint(dargs)
    else:
        ckp = None

    args = AttrDict(dargs)

    logger.info(f'Command ran: {" ".join(sys.argv)}')
    logger.info(pformat(args))

    if args.run:
        torch.manual_seed(args.sys.seed)
        models = get_models(args)
        dataloaders = get_dataloader(args)
        trainer = Trainer(args, models, dataloaders, ckp)
        if not args.train.test_only:
            trainer.train()
        else:
            # 如果不是训练，而是测试
            trainer.testonly()
            pass

