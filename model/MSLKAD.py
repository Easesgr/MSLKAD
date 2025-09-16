# -*- coding: utf-8 -*-
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath

# LKA from VAN (https://github.com/Visual-Attention-Network)
class LKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 7, padding=7 // 2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 9, stride=1, padding=((9 // 2) * 4), groups=dim, dilation=4)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        return u * attn

class Attention(nn.Module):
    def __init__(self, n_feats):
        super().__init__()

        self.norm = LayerNorm(n_feats, data_format='channels_first')
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)

        self.proj_1 = nn.Conv2d(n_feats, n_feats, 1)
        self.spatial_gating_unit = LKA(n_feats)
        self.proj_2 = nn.Conv2d(n_feats, n_feats, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(self.norm(x))
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x * self.scale + shorcut
        return x
    # ----------------------------------------------------------------------------------------------------------------


class MLP(nn.Module):
    def __init__(self, n_feats):
        super().__init__()

        self.norm = LayerNorm(n_feats, data_format='channels_first')
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)

        i_feats = 2 * n_feats

        self.fc1 = nn.Conv2d(n_feats, i_feats, 1, 1, 0)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(i_feats, n_feats, 1, 1, 0)

    def forward(self, x):
        shortcut = x.clone()
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)

        return x * self.scale + shortcut


class SimpleGate(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        i_feats = n_feats * 2

        self.Conv1 = nn.Conv2d(n_feats, i_feats, 1, 1, 0)
        # self.DWConv1 = nn.Conv2d(n_feats, n_feats, 7, 1, 7//2, groups= n_feats)
        self.Conv2 = nn.Conv2d(n_feats, n_feats, 1, 1, 0)

        self.norm = LayerNorm(n_feats, data_format='channels_first')
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)

    def forward(self, x):
        shortcut = x.clone()

        # Ghost Expand
        x = self.Conv1(self.norm(x))
        a, x = torch.chunk(x, 2, dim=1)
        x = x * a  # self.DWConv1(a)
        x = self.Conv2(x)

        return x * self.scale + shortcut
    # -----------------------------------------------------------------------------------------------------------------


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

# Ghost Kernel Attention
class GKG(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        i_feats = n_feats * 2

        self.Conv1 = nn.Conv2d(n_feats, i_feats, 1, 1, 0)
        self.DWConv1 = nn.Conv2d(n_feats, n_feats, 7, 1, 7 // 2, groups=n_feats)
        self.Conv2 = nn.Conv2d(n_feats, n_feats, 1, 1, 0)

        self.norm = LayerNorm(n_feats, data_format='channels_first')
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)

    def forward(self, x):
        shortcut = x.clone()

        # Ghost Expand
        x = self.Conv1(self.norm(x))
        a, x = torch.chunk(x, 2, dim=1)
        x = x * self.DWConv1(a)
        x = self.Conv2(x)

        return x * self.scale + shortcut


# Multiscale Large Attention
class MLA_Ablation(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        i_feats = 2 * n_feats

        self.n_feats = n_feats
        self.i_feats = i_feats

        self.group_feats = n_feats // 2
        self.norm = LayerNorm(n_feats, data_format='channels_first')
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)

        self.LKA7 = self.build_lka(self.group_feats, kernel_size1=7, kernel_size2=9, dilation=4)
        self.LKA5 = self.build_lka(self.group_feats, kernel_size1=5, kernel_size2=7, dilation=3)
        # self.LKA3 = self.build_lka(self.group_feats, kernel_size1=3, kernel_size2=5, dilation=2)

        # self.X3 = nn.Conv2d(self.group_feats, self.group_feats, 3, 1, 1, groups=self.group_feats)
        self.X5 = nn.Conv2d(self.group_feats, self.group_feats, 5, 1, 5 // 2, groups=self.group_feats)
        self.X7 = nn.Conv2d(self.group_feats, self.group_feats, 7, 1, 7 // 2, groups=self.group_feats)

        self.proj_first = nn.Sequential(
            nn.Conv2d(n_feats, i_feats, 1, 1, 0))

        self.proj_last = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 1, 1, 0))

    def build_lka(self,group_feats, kernel_size1, kernel_size2, dilation, padding2=None):
        if padding2 is None:
            padding2 = (kernel_size2 // 2) * dilation

        return nn.Sequential(
            nn.Conv2d(group_feats, group_feats, kernel_size=kernel_size1, stride=1, padding=kernel_size1 // 2,
                      groups=group_feats),
            nn.Conv2d(group_feats, group_feats, kernel_size=kernel_size2, stride=1, padding=padding2, dilation=dilation,
                      groups=group_feats),
            nn.Conv2d(group_feats, group_feats, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        shortcut = x.clone()

        x = self.norm(x)

        x = self.proj_first(x)

        a, x = torch.chunk(x, 2, dim=1)

        # u_1, u_2, u_3= torch.chunk(u, 3, dim=1)
        a_1, a_2 = torch.chunk(a, 2, dim=1)

        a = torch.cat([self.LKA7(a_1) * self.X7(a_1), self.LKA5(a_2) * self.X5(a_2)], dim=1)

        x = self.proj_last(x * a) * self.scale + shortcut

        return x




# Multiscale Large Attention
class MLA(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        i_feats = 2 * n_feats

        self.n_feats = n_feats
        self.i_feats = i_feats
        self.group_feats = n_feats // 3
        self.norm = LayerNorm(n_feats, data_format='channels_first')
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)

        self.LKA7 = self.build_lka(self.group_feats, kernel_size1=7, kernel_size2=9, dilation=4)
        self.LKA5 = self.build_lka(self.group_feats, kernel_size1=5, kernel_size2=7, dilation=3)
        self.LKA3 = self.build_lka(self.group_feats, kernel_size1=3, kernel_size2=5, dilation=2)

        self.X3 = nn.Conv2d(self.group_feats, self.group_feats, 3, 1, 1, groups=self.group_feats)
        self.X5 = nn.Conv2d(self.group_feats, self.group_feats, 5, 1, 5 // 2, groups=self.group_feats)
        self.X7 = nn.Conv2d(self.group_feats, self.group_feats, 7, 1, 7 // 2, groups=self.group_feats)

        self.proj_first = nn.Sequential(
            nn.Conv2d(n_feats, i_feats, 1, 1, 0))

        self.proj_last = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 1, 1, 0))

    def build_lka(self,group_feats, kernel_size1, kernel_size2, dilation, padding2=None):
        if padding2 is None:
            padding2 = (kernel_size2 // 2) * dilation

        return nn.Sequential(
            nn.Conv2d(group_feats, group_feats, kernel_size=kernel_size1, stride=1, padding=kernel_size1 // 2,
                      groups=group_feats),
            nn.Conv2d(group_feats, group_feats, kernel_size=kernel_size2, stride=1, padding=padding2, dilation=dilation,
                      groups=group_feats),
            nn.Conv2d(group_feats, group_feats, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        shortcut = x.clone()

        x = self.norm(x)

        x = self.proj_first(x)

        a, x = torch.chunk(x, 2, dim=1)

        a_1, a_2, a_3 = torch.chunk(a, 3, dim=1)

        a = torch.cat([self.LKA3(a_1) * self.X3(a_1), self.LKA5(a_2) * self.X5(a_2), self.LKA7(a_3) * self.X7(a_3)],
                      dim=1)
        x = self.proj_last(x * a) * self.scale + shortcut

        return x


'''
Input
 ↓
MLA（大核注意力） - 捕捉大范围上下文，感受野广
 ↓
GKG（Ghost机制局部增强） - 强化细节与局部特征
 ↓
Output
'''
# Multi-scale Feature Aggregation
class MFA(nn.Module):
    def __init__(
            self, n_feats):
        super().__init__()

        self.LKA = MLA(n_feats)

        self.LFE = GKG(n_feats)

    def forward(self, x):
        # large kernel attention
        x = self.LKA(x)

        # local feature extraction
        x = self.LFE(x)

        return x


# Expanded Large Kernel Attention
class EKA(nn.Module):
    def __init__(self, n_feats):
        super().__init__()

        self.conv0 = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 1, 1, 0),
            nn.GELU())

        self.att = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 7, 1, 7 // 2, groups=n_feats),
            nn.Conv2d(n_feats, n_feats, 9, stride=1, padding=(9 // 2) * 3, groups=n_feats, dilation=3),
            nn.Conv2d(n_feats, n_feats, 1, 1, 0))

        self.conv1 = nn.Conv2d(n_feats, n_feats, 1, 1, 0)

    def forward(self, x):
        x = self.conv0(x)
        x = x * self.att(x)
        x = self.conv1(x)
        return x


class MFABlock(nn.Module):
    def __init__(self, n_resblocks, n_feats):
        super(MFABlock, self).__init__()
        self.body = nn.ModuleList([
            MFA(n_feats) \
            for _ in range(n_resblocks)])

        self.body_t = EKA(n_feats)

    def forward(self, x):
        res = x.clone()

        for i, block in enumerate(self.body):
            res = block(res)

        x = self.body_t(res) + x

        return x


class MSLKAD(nn.Module):
    def __init__(self, args,
                 ):
        super(MSLKAD, self).__init__()

        n_resblocks = args['model']['MSLKAD']['model_args']['n_resblocks']
        self.n_MFABlocks = args['model']['MSLKAD']['model_args']['n_resgroups']
        n_feats = args['model']['MSLKAD']['model_args']['n_feats']

        self.head = nn.Conv2d(3, n_feats, 3, 1, 1)
        # define body module
        self.body = nn.ModuleList([
            MFABlock(n_resblocks, n_feats)
            for i in range(self.n_MFABlocks)
        ])

        self.body_t = nn.Conv2d(n_feats, n_feats, 3, 1, 1)


        self.tail = nn.Conv2d(n_feats, 3, 3, 1, 1)
    def forward(self, x):
        fea = []
        x = self.head(x)
        res = x
        temp = res  # 用于每隔两块加残差
        for idx, (i) in enumerate(self.body):
            res = i(res)
            fea.append(res)
            if (idx + 1) % 2 == 0:
                res = res + temp  # 每两块加一次残差
                temp = res  # 更新 temp
        res = self.body_t(res) + x

        x = self.tail(res)
        return x, fea
    # def forward(self, x):
    #     x = self.head(x)
    #     res = x
    #     temp = res  # 用于每隔两块加残差
    #
    #     for idx, (i) in enumerate(self.body):
    #         res = i(res)
    #
    #         if (idx + 1) % 2 == 0:
    #             res = res + temp  # 每两块加一次残差
    #             temp = res  # 更新 temp
    #     res = self.body_t(res) + x
    #
    #     x = self.tail(res)
    #     return x