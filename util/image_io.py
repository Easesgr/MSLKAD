"""RGB image I/O and explicit filename-stem pairing."""

from pathlib import Path

import numpy as np
from PIL import Image
import torch

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}


def list_images(path):
    path = Path(path).expanduser()
    if path.is_file():
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f'Unsupported image extension: {path}')
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f'Image file or directory not found: {path}')
    images = sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise ValueError(f'No supported images found in {path}')
    index_images(images)
    return images


def index_images(images):
    indexed = {}
    for path in images:
        if path.stem in indexed:
            raise ValueError(f'Duplicate image stem {path.stem!r}: {indexed[path.stem]} and {path}')
        indexed[path.stem] = path
    return indexed


def pair_images(input_path, target_path):
    inputs, targets = list_images(input_path), list_images(target_path)
    if Path(input_path).expanduser().is_file() and Path(target_path).expanduser().is_file():
        return [(inputs[0], targets[0])]
    input_index, target_index = index_images(inputs), index_images(targets)
    missing = sorted(input_index.keys() - target_index.keys())
    extra = sorted(target_index.keys() - input_index.keys()) if Path(input_path).expanduser().is_dir() else []
    if missing or extra:
        raise ValueError(f'Unpaired images: missing targets={missing[:10]}, extra targets={extra[:10]}. '
                         'LR and HR filenames must have matching stems.')
    return [(path, target_index[path.stem]) for path in inputs]


def read_rgb(path):
    with Image.open(path) as image:
        return image.convert('RGB')


def image_tensor(image):
    array = np.array(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def save_tensor(tensor, path):
    if tensor.ndim != 4 or tensor.shape[:2] != (1, 3):
        raise ValueError('Expected a single RGB image of shape [1, 3, H, W].')
    array = tensor.detach().cpu()[0].clamp(0, 1).permute(1, 2, 0).numpy()
    Image.fromarray(np.rint(array * 255).astype(np.uint8)).save(path)
