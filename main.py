"""Train with YAML configuration and key=value overrides (single process)."""

import logging
import random
import sys
from pprint import pformat

import numpy as np
import torch

from data import get_dataloader
from log import Checkpoint
from model import get_models
from trainer.train import Trainer
from util.sys_util import AttrDict, get_args

logger = logging.getLogger(__name__)
logging.getLogger('PIL').setLevel(logging.INFO)


def main():
    if '--help' in sys.argv[1:] or '-h' in sys.argv[1:]:
        print('Usage: python main.py [configs/override.yaml] [key=value ...]\n'
              'Examples:\n'
              '  python main.py data.train_data_root=datasets/Rain200H/train '
              'train.vgg_model_path=pretrained/vgg19-dcbb9e9d.pth "train.gpus=[0,1]"\n'
              '  python main.py weights=runs/RUN/latest.pt resume=true\n'
              'For standalone inference (no training data/VGG): python test.py --help')
        return
    config = get_args()
    if not config['run']:
        print(pformat(config))
        return
    ckp = Checkpoint(config) if config['sys']['log'] else None
    if ckp is None:
        logging.basicConfig(level=logging.DEBUG if config['debug'] else logging.INFO)
    args = AttrDict(config)
    logger.info('Command: %s', ' '.join(sys.argv))
    logger.info(pformat(config))
    random.seed(args.sys.seed)
    np.random.seed(args.sys.seed)
    torch.manual_seed(args.sys.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.sys.seed)
    trainer = Trainer(args, get_models(args), get_dataloader(args), ckp)
    if args.train.test_only:
        trainer.testonly()
    else:
        trainer.train()


if __name__ == '__main__':
    main()
