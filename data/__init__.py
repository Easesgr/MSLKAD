from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dataloder import TrainDataSet, TestDataSet


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def get_dataloader(args):
    loader_train = None
    if not args.train.test_only:
        roots = args.data.train_data_root
        roots = roots.split(',') if isinstance(roots, str) else roots
        roots = [Path(root.strip()).expanduser() for root in roots]
        train_set = TrainDataSet([root / 'LR' for root in roots], [root / 'HR' for root in roots], args.data.patch_size)
        if type(args.data.batch_size) is not int or args.data.batch_size <= 0:
            raise ValueError('data.batch_size must be a positive integer.')
        if len(train_set) < args.data.batch_size:
            raise ValueError(f'Training has {len(train_set)} pairs but batch_size={args.data.batch_size}; '
                             'reduce batch_size (training uses drop_last=True).')
        loader_train = DataLoader(
            train_set, batch_size=args.data.batch_size, shuffle=True,
            num_workers=args.data.n_threads, drop_last=True, worker_init_fn=seed_worker,
        )
    loader_test = None
    if args.data.val_data_root:
        val_root = Path(args.data.val_data_root).expanduser()
        val_set = TestDataSet(val_root / 'LR', val_root / 'HR')
        loader_test = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0)
    elif args.train.test_only:
        raise ValueError('train.test_only=true requires data.val_data_root.')
    return loader_train, loader_test
