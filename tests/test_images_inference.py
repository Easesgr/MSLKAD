import csv
import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest
import torch
import yaml

from data import get_dataloader
from model import get_models
from util.image_io import image_tensor, list_images, pair_images, read_rgb
from util.inference import y_metrics
from util.metric_util import calculate_psnr, calculate_ssim, tensor2img

ROOT = Path(__file__).resolve().parents[1]


def write_image(path, color=(90, 120, 180), size=(17, 13)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', size, color).save(path)
    return path


def test_rgb_pairing_validation_and_resizing(args, tmp_path):
    train_root = tmp_path / 'train'
    write_image(train_root / 'LR' / 'a.PNG', color=(255, 0, 0), size=(3, 9))
    write_image(train_root / 'HR' / 'a.bmp', color=(255, 0, 0), size=(3, 9))
    tensor = image_tensor(read_rgb(train_root / 'LR' / 'a.PNG'))
    torch.testing.assert_close(tensor[:, 0, 0], torch.tensor([1., 0., 0.]))
    args.data.train_data_root = str(train_root)
    args.data.val_data_root = None
    loader, validation = get_dataloader(args)
    inputs, targets = next(iter(loader))
    assert inputs.shape == targets.shape == (1, 3, 16, 16)
    assert validation is None
    torch.testing.assert_close(inputs, targets)
    write_image(train_root / 'HR' / 'b.png')
    with pytest.raises(ValueError, match='extra targets'):
        pair_images(train_root / 'LR', train_root / 'HR')
    write_image(train_root / 'LR' / 'a.jpg')
    with pytest.raises(ValueError, match='Duplicate'):
        list_images(train_root / 'LR')
    empty = tmp_path / 'empty'
    empty.mkdir()
    with pytest.raises(ValueError, match='No supported images'):
        list_images(empty)


def test_metrics_keep_historical_protocol_and_reject_bad_pairs():
    torch.manual_seed(7)
    output = torch.rand(1, 3, 13, 17) * 1.5 - 0.2
    target = torch.rand_like(output)
    psnr, ssim = y_metrics(output, target)
    expected_output, expected_target = tensor2img(output), tensor2img(target)
    assert psnr == calculate_psnr(expected_output, expected_target, crop_border=0)
    assert ssim == calculate_ssim(expected_output, expected_target, crop_border=0)
    same_psnr, same_ssim = y_metrics(target, target)
    assert np.isinf(same_psnr)
    assert same_ssim == pytest.approx(1.0)
    with pytest.raises(ValueError, match='matching'):
        y_metrics(output, target[..., :-1])
    with pytest.raises(ValueError, match='finite'):
        y_metrics(output * float('nan'), target)


def test_cli_checkpoint_to_png_and_metrics(args, tmp_path):
    # Random weights here exercise I/O only; these are not pretrained evaluation results.
    weights = tmp_path / 'smoke-weights.pt'
    torch.save({'models': {'MSLKAD': get_models(args)['MSLKAD'].state_dict()}}, weights)
    config = tmp_path / 'tiny.yaml'
    config.write_text(yaml.safe_dump({'model': {'MSLKAD': {'model_args': dict(args.model.MSLKAD.model_args)}}}))
    input_dir, target_dir = tmp_path / 'input', tmp_path / 'target'
    write_image(input_dir / 'sample.png')
    write_image(target_dir / 'sample.png')
    output_dir = tmp_path / 'output'
    command = [sys.executable, str(ROOT / 'test.py'), '--weights', str(weights), '--config', str(config),
               '--input', str(input_dir), '--target', str(target_dir), '--output', str(output_dir), '--device', 'cpu']
    import os
    result = subprocess.run(command, capture_output=True, text=True, timeout=60,
                            env={**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
    assert result.returncode == 0, result.stdout + result.stderr
    with Image.open(output_dir / 'sample.png') as image:
        assert image.mode == 'RGB'
        assert image.size == (17, 13)
    with (output_dir / 'metrics.csv').open() as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 1 and rows[0]['filename'] == 'sample.png'
    assert np.isfinite(float(rows[0]['psnr_y']))
    assert 'Mean over 1 images' in result.stdout
    missing = subprocess.run([sys.executable, str(ROOT / 'test.py')], capture_output=True, text=True, timeout=30)
    assert missing.returncode != 0 and '--weights' in missing.stderr


def test_cli_refuses_output_overwriting_input(args, tmp_path):
    # Import the script without executing its entry point.
    spec = importlib.util.spec_from_file_location('inference_cli', ROOT / 'test.py')
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    image = write_image(tmp_path / 'input.png')
    options = cli.build_parser().parse_args([
        '--weights', str(tmp_path / 'unused.pt'), '--input', str(image), '--output', str(tmp_path), '--device', 'cpu',
    ])
    with pytest.raises(ValueError, match='overwrite'):
        cli.run(options)
