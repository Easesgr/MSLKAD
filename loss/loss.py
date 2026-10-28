from torchvision import models

import torch
import torch.nn as nn
import torch.nn.functional as F
def trainable_(net, trainable):
    for param in net.parameters():
        param.requires_grad = trainable

class PerceptualLoss(nn.Module):
    """Historical mean VGG feature MSE, with gradients retained for predictions.

    Inputs remain RGB in [0, 1], with no added ImageNet normalization, matching
    the author's training code. Pretrained torchvision VGG19 weights are required.
    """

    def __init__(self, device=None, model_path=None):
        super().__init__()
        from pathlib import Path
        if not model_path or not Path(model_path).expanduser().is_file():
            raise FileNotFoundError(
                'Training requires a local pretrained VGG19 state_dict; '
                'set train.vgg_model_path=/path/to/vgg19-dcbb9e9d.pth.'
            )
        vgg = models.vgg19(weights=None)
        vgg.load_state_dict(torch.load(Path(model_path).expanduser(), map_location='cpu', weights_only=True), strict=True)
        # Only layers through index 11 contribute to the original loss.
        self.vgg_layers = vgg.features[:12].to(device).eval()
        trainable_(self.vgg_layers, False)
        self.layer_names = {'0': 'conv1_1', '3': 'relu1_2', '6': 'relu2_1', '8': 'relu2_2', '11': 'relu3_1'}
        self.loss = nn.MSELoss()

    def get_layer_output(self, x):
        outputs = []
        for name, module in self.vgg_layers._modules.items():
            x = F.relu(x, inplace=False) if isinstance(module, nn.ReLU) else module(x)
            if name in self.layer_names:
                outputs.append(x)
        return outputs

    def get_GTlayer_output(self, x):
        with torch.no_grad():
            return self.get_layer_output(x)

    def forward(self, output, target):
        outputs = self.get_layer_output(output)
        targets = self.get_GTlayer_output(target)
        return sum(self.loss(o, t) for o, t in zip(outputs, targets)) / len(targets)


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
        self.register_buffer('kernel', torch.matmul(k.t(),k).unsqueeze(0).repeat(3,1,1,1).to(device))
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
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
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


