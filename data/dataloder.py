"""Paired LR/HR datasets; filenames are matched explicitly, never by sort order."""

from torch.utils.data import Dataset

from util.data_util import augment_pair
from util.image_io import image_tensor, pair_images, read_rgb


def read_pair(lr_path, gt_path):
    lr_img, gt_img = read_rgb(lr_path), read_rgb(gt_path)
    if lr_img.size != gt_img.size:
        raise ValueError(f'Pair dimensions differ: {lr_path} {lr_img.size}; {gt_path} {gt_img.size}')
    return lr_img, gt_img


class TrainDataSet(Dataset):
    def __init__(self, lr_roots, gt_roots, patch_size):
        if len(lr_roots) != len(gt_roots):
            raise ValueError('LR and HR root lists must have the same length.')
        if type(patch_size) is not int or patch_size <= 0:
            raise ValueError('patch_size must be a positive integer.')
        self.patch_size = patch_size
        self.pairs = []
        for lr_root, gt_root in zip(lr_roots, gt_roots):
            self.pairs.extend(pair_images(lr_root, gt_root))
        if not self.pairs:
            raise ValueError('Training dataset is empty.')

    def __getitem__(self, index):
        return augment_pair(*read_pair(*self.pairs[index]), self.patch_size)

    def __len__(self):
        return len(self.pairs)


class TestDataSet(Dataset):
    def __init__(self, lr_root, gt_root):
        self.pairs = pair_images(lr_root, gt_root)

    def __getitem__(self, index):
        lr_path, gt_path = self.pairs[index]
        lr_img, gt_img = read_pair(lr_path, gt_path)
        return image_tensor(lr_img), image_tensor(gt_img), lr_path.name

    def __len__(self):
        return len(self.pairs)
