from torchvision import models
from torch.autograd import Variable

import torch
import torch.nn as nn
import torch.nn.functional as F
def trainable_(net, trainable):
    for param in net.parameters():
        param.requires_grad = trainable

class PerceptualLoss(nn.Module):
    def __init__(self, device=None, model_path=None):
        super(PerceptualLoss, self).__init__()
        self.device = device

        # 初始化 VGG19 网络结构
        self.model = models.vgg19(pretrained=False).to(device)  # 不加载预训练权重

        # 如果提供了本地路径，则加载权重
        if model_path:
            print(f"Loading pretrained VGG19 from {model_path}")
            self.model.load_state_dict(torch.load(model_path))  # 从本地路径加载权重
        else:
            print("No model path provided, VGG19 weights will be initialized randomly")

        # 设置模型为不可训练
        trainable_(self.model, False)

        # 损失函数
        self.loss = nn.MSELoss().to(device)

        # VGG19 的层
        self.vgg_layers = self.model.features
        self.layer_names = {'0': 'conv1_1', '3': 'relu1_2', '6': 'relu2_1', '8': 'relu2_2', '11': 'relu3_1'}

    def get_layer_output(self, x):
        output = []
        for name, module in self.vgg_layers._modules.items():
            if isinstance(module, nn.ReLU):
                module = nn.ReLU(inplace=False)
            x = module(x)
            if name in self.layer_names:
                output.append(x)
        return output

    def get_GTlayer_output(self, x):
        with torch.no_grad():
            output = []
            for name, module in self.vgg_layers._modules.items():
                if isinstance(module, nn.ReLU):
                    module = nn.ReLU(inplace=False)
                x = module(x)
                if name in self.layer_names:
                    output.append(x)
        return output

    def forward(self, O_, T_):
        O_ = O_.to(self.device)
        T_ = T_.to(self.device)
        o = self.get_layer_output(O_)
        t = self.get_GTlayer_output(T_)
        loss_PL = sum(self.loss(o[i], t[i]) for i in range(len(t))) / len(t)
        return loss_PL


    def __call__(self, O_, T_):
        o = self.get_layer_output(O_)
        t = self.get_GTlayer_output(T_)
        loss_PL = 0
        for i in range(len(t)):
            if i == 0:
                loss_PL = self.loss(o[i], t[i])
            else:
                loss_PL += self.loss(o[i], t[i])

        loss_PL = loss_PL / float(len(t))
        loss_PL = Variable(loss_PL, requires_grad=True)

        return loss_PL


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3,device=None):
        super(CharbonnierLoss, self).__init__()
        self.device = device
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))
        return loss

class EdgeLoss(nn.Module):
    def __init__(self,device=None):
        super(EdgeLoss, self).__init__()
        self.device = device
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(3,1,1,1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.to(self.device)
        self.loss = CharbonnierLoss()


    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered    = self.conv_gauss(current)
        down        = filtered[:,:,::2,::2]
        new_filter  = torch.zeros_like(filtered)
        new_filter[:,:,::2,::2] = down*4
        filtered    = self.conv_gauss(new_filter)
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = self.loss(self.laplacian_kernel(x.to(self.device)), self.laplacian_kernel(y.to(self.device)))
        return loss







class MultiscaleLoss(nn.Module):
    def __init__(self, ld=[1.0], device=None):
        super(MultiscaleLoss, self).__init__()
        self.loss = nn.L1Loss().to(device)
        self.ld = ld
        self.device = device

    def forward(self, S_, gt):
        """
        S_: list of predicted tensors at different scales, e.g. [B, C, H, W]
        gt: tensor of shape [B, C, H, W], pixel values in [0, 1]
        """
        B, C, H, W = gt.shape
        gt = gt.to(self.device)  # 保证 ground truth 在同一设备上
        T_ = []

        # 构造多尺度 ground truth
        for scale in [1.0]:
            resized = F.interpolate(gt, scale_factor=scale, mode='area')
            T_.append(resized.to(self.device))  # 确保每个尺度都转到 device 上

        # 计算多尺度加权 L1 Loss
        loss_ML = 0.0
        for i in range(len(self.ld)):
            pred = S_[i].to(self.device)  # 如果模型输出未在 device 上，也转一下
            target = T_[i]
            if pred.shape != target.shape:
                raise ValueError(f"Shape mismatch at scale {i}: pred={pred.shape}, target={target.shape}")
            loss_ML += self.ld[i] * self.loss(pred, target)

        return loss_ML / B


