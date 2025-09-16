# from data.benchmark import BenchmarkDataset
from data import dataloder
from torch.utils.data import  DataLoader, DistributedSampler
import logging
import os
logger = logging.getLogger(__name__)

def get_dataloader(args):
    train_data_roots = args.data.train_data_root.split(',')
    train_lr_roots = [os.path.join(root, 'LR') for root in train_data_roots]
    train_hr_roots = [os.path.join(root, 'HR') for root in train_data_roots]

    train_set = dataloder.TrainDataSet(train_lr_roots, train_hr_roots, args.data.patch_size)

    val_set = dataloder.TestDataSet(
        os.path.join(args.data.val_data_root, 'LR'),
        os.path.join(args.data.val_data_root, 'HR')
    )

    # 这里不再使用 DistributedSampler，保持 shuffle=True
    loader_train = DataLoader(
        train_set, batch_size=args.data.batch_size,
        shuffle=True, num_workers=4, drop_last=True
    )

    loader_test = DataLoader(
        val_set, batch_size=1, shuffle=False, num_workers=0, drop_last=False
    )

    return loader_train, loader_test

