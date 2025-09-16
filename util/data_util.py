
import cv2
# 数据处理
import random
import numpy as np
import torch
from torchvision.transforms import functional as TF
from torchvision.transforms import Compose, ToTensor
from PIL import Image


def augment_pair(lr_img, gt_img, patch_size):
    # 获取图像尺寸
    width, height = lr_img.size

    # 如果图像尺寸小于 patch_size，进行 resize 操作
    if width < patch_size and height < patch_size:
        lr_img = lr_img.resize((patch_size, patch_size), Image.ANTIALIAS)
        gt_img = gt_img.resize((patch_size, patch_size), Image.ANTIALIAS)
    elif width < patch_size:
        lr_img = lr_img.resize((patch_size, height), Image.ANTIALIAS)
        gt_img = gt_img.resize((patch_size, height), Image.ANTIALIAS)
    elif height < patch_size:
        lr_img = lr_img.resize((width, patch_size), Image.ANTIALIAS)
        gt_img = gt_img.resize((width, patch_size), Image.ANTIALIAS)

    # 获取新的图像尺寸
    width, height = lr_img.size

    # 色彩增强
    if random.randint(0, 2) == 1:
        lr_img = TF.adjust_gamma(lr_img, 1)
        gt_img = TF.adjust_gamma(gt_img, 1)

    if random.randint(0, 2) == 1:
        sat_factor = 1 + (0.2 - 0.4 * np.random.rand())
        lr_img = TF.adjust_saturation(lr_img, sat_factor)
        gt_img = TF.adjust_saturation(gt_img, sat_factor)

    # 随机裁剪
    x = random.randint(0, width - patch_size)
    y = random.randint(0, height - patch_size)
    lr_img = lr_img.crop((x, y, x + patch_size, y + patch_size))
    gt_img = gt_img.crop((x, y, x + patch_size, y + patch_size))

    # 转为 tensor
    lr_tensor = ToTensor()(lr_img)
    gt_tensor = ToTensor()(gt_img)

    # 空间增强（翻转、旋转）
    aug = random.randint(0, 8)
    if aug == 1:
        lr_tensor = lr_tensor.flip(1)
        gt_tensor = gt_tensor.flip(1)
    elif aug == 2:
        lr_tensor = lr_tensor.flip(2)
        gt_tensor = gt_tensor.flip(2)
    elif aug == 3:
        lr_tensor = torch.rot90(lr_tensor, dims=(1, 2))
        gt_tensor = torch.rot90(gt_tensor, dims=(1, 2))
    elif aug == 4:
        lr_tensor = torch.rot90(lr_tensor, dims=(1, 2), k=2)
        gt_tensor = torch.rot90(gt_tensor, dims=(1, 2), k=2)
    elif aug == 5:
        lr_tensor = torch.rot90(lr_tensor, dims=(1, 2), k=3)
        gt_tensor = torch.rot90(gt_tensor, dims=(1, 2), k=3)
    elif aug == 6:
        lr_tensor = torch.rot90(lr_tensor.flip(1), dims=(1, 2))
        gt_tensor = torch.rot90(gt_tensor.flip(1), dims=(1, 2))
    elif aug == 7:
        lr_tensor = torch.rot90(lr_tensor.flip(2), dims=(1, 2))
        gt_tensor = torch.rot90(gt_tensor.flip(2), dims=(1, 2))

    return lr_tensor, gt_tensor


def read_image(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # cv2.IMREAD_GRAYSCALE
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img