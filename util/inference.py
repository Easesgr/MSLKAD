"""Shared full-image/tiled forward pass and the repository's Y-channel metrics."""

import torch

from util.metric_util import calculate_psnr, calculate_ssim, tensor2img


def select_device(requested='auto'):
    if requested == 'auto':
        requested = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(requested)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Supported devices are cpu, cuda, cuda:N, and auto.')
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError('CUDA was requested but is unavailable.')
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(f'CUDA device {device.index} is unavailable.')
    return device


def forward_image(model, inputs, tile_size=0, tile_overlap=32):
    if tile_size < 0 or tile_overlap < 0 or (tile_size and tile_overlap >= tile_size):
        raise ValueError('Require tile_size >= 0 and 0 <= tile_overlap < tile_size when tiling.')
    if tile_size == 0:
        output = model(inputs)
    else:
        height, width = inputs.shape[-2:]
        stride = tile_size - tile_overlap
        rows = list(range(0, max(height - tile_size, 0), stride)) + [max(height - tile_size, 0)]
        cols = list(range(0, max(width - tile_size, 0), stride)) + [max(width - tile_size, 0)]
        output, counts = torch.zeros_like(inputs), torch.zeros_like(inputs)
        for row in rows:
            for col in cols:
                patch = inputs[..., row:row + tile_size, col:col + tile_size]
                restored = model(patch)
                if restored.shape != patch.shape:
                    raise ValueError('Model output must have the same shape as its input.')
                output[..., row:row + tile_size, col:col + tile_size] += restored
                counts[..., row:row + tile_size, col:col + tile_size] += 1
        output = output / counts
    if output.shape != inputs.shape:
        raise ValueError('Model output must have the same shape as its input.')
    if not torch.isfinite(output).all():
        raise ValueError('Model produced non-finite pixels; check weights and input data.')
    return output


def y_metrics(output, target):
    """Historical protocol: raw float output, BT.601 Y, crop=0, replicated SSIM.

    No clamping or uint8 rounding is performed for metrics. Saved PNGs are
    clamped/rounded separately. Keep the existing SSIM implementation unchanged.
    """
    if output.shape != target.shape or output.ndim != 4 or output.shape[:2] != (1, 3):
        raise ValueError(f'Metrics require matching [1, 3, H, W] images: {output.shape}, {target.shape}')
    if not torch.isfinite(output).all() or not torch.isfinite(target).all():
        raise ValueError('Metrics require finite pixels.')
    prediction, truth = tensor2img(output.float()), tensor2img(target.float())
    return (float(calculate_psnr(prediction, truth, crop_border=0, input_order='HWC')),
            float(calculate_ssim(prediction, truth, crop_border=0, input_order='HWC')))
