import os
suffixes = ['/*.png', '/*.jpg', '/*.bmp', '/*.tif']
import random
import torch
import numpy as np
import glob
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Normalize
from PIL import Image
from util.data_util import augment_pair
import torchvision.transforms.functional as TF
class TrainDataSet(Dataset):
    def __init__(self, lr_roots, gt_roots, patch_size):
        super().__init__()

        self.file_client = None
        self.patch_size = patch_size
        self.suffixes = suffixes

        # 支持多个路径
        self.lr_data = []
        self.gt_data = []

        for lr_root, gt_root in zip(lr_roots, gt_roots):
            print(lr_root, gt_root)
            for suffix in self.suffixes:
                self.lr_data.extend(glob.glob(os.path.join(lr_root + suffix)))
                self.gt_data.extend(glob.glob(os.path.join(gt_root + suffix)))

        self.lr_data = sorted(self.lr_data)
        self.gt_data = sorted(self.gt_data)

        assert len(self.lr_data) == len(self.gt_data), "the length of lrs and gts is not equal!"

    def __getitem__(self, index):
        # 获取图像路径
        lr_name = self.lr_data[index]
        gt_name = self.gt_data[index]

        # 读取图像
        lr_img = Image.open(lr_name).convert('RGB')
        gt_img = Image.open(gt_name).convert('RGB')

        # 应用增强
        lr_tensor, gt_tensor = augment_pair(lr_img, gt_img, self.patch_size)

        # 通道检查
        if lr_tensor.shape[0] != 3 or gt_tensor.shape[0] != 3:
            raise Exception(f"Bad image channel: {gt_name}")

        return lr_tensor, gt_tensor

    def __len__(self):
        return len(self.lr_data)




class TestDataSet(Dataset):
    def __init__(self, lr_root, gt_root):
        super().__init__()

        self.lr_data = []
        self.gt_data = []
        for suffix in suffixes:
            self.lr_data.extend(glob.glob(lr_root + suffix))
            self.gt_data.extend(glob.glob(gt_root + suffix))
        self.lr_data = sorted(self.lr_data)
        self.gt_data = sorted(self.gt_data)

        assert len(self.lr_data) == len(self.gt_data), "the length of lrs and gts is not equal!"

    def __getitem__(self, index):
        lr_path = self.lr_data[index]
        gt_path = self.gt_data[index]

        lr_img = Image.open(lr_path).convert('RGB')
        gt_img = Image.open(gt_path).convert('RGB')
        # --- 转换为张量并进行归一化 ---
        transform_lr = Compose([ToTensor()])
        transform_gt = Compose([ToTensor()])

        lr_tensor = transform_lr(lr_img)
        gt_tensor = transform_gt(gt_img)

        # 提取文件名作为字符串（例如 "0001.png"）
        filename = os.path.basename(lr_path)
        return lr_tensor, gt_tensor,filename

    def __len__(self):
        return len(self.lr_data)