"""Derain RGB images with a strictly loaded MSLKAD checkpoint."""

import argparse
import csv
from pathlib import Path

import torch

from model import get_models
from util.checkpoint import load_weights
from util.image_io import image_tensor, list_images, pair_images, read_rgb, save_tensor
from util.inference import forward_image, select_device, y_metrics
from util.sys_util import merge_dict, read_yaml


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', required=True, type=Path, help='Trained .pt/.pth checkpoint (required).')
    parser.add_argument('--input', required=True, type=Path, help='One image or a flat image directory.')
    parser.add_argument('--output', required=True, type=Path, help='Directory for restored PNG images.')
    parser.add_argument('--target', type=Path, help='Optional clean image/directory for Y-channel PSNR/SSIM.')
    parser.add_argument('--config', type=Path, help='YAML overrides for the default model architecture.')
    parser.add_argument('--device', default='auto', help='auto, cpu, cuda, or cuda:N (default: auto).')
    parser.add_argument('--tile-size', type=int, default=0, help='Optional tile edge; 0 runs full images (default).')
    parser.add_argument('--tile-overlap', type=int, default=32, help='Tile overlap; tiled results differ from full-image results.')
    return parser


def run(args):
    config = read_yaml(Path(__file__).parent / 'configs' / 'default.yaml')
    if args.config:
        merge_dict(config, read_yaml(args.config))
    device = select_device(args.device)
    pairs = pair_images(args.input, args.target) if args.target else [(p, None) for p in list_images(args.input)]
    output_dir = args.output.expanduser().resolve()
    # Never overwrite a source image, even if input/target has another suffix.
    sources = {path.resolve() for pair in pairs for path in pair if path is not None}
    for input_path, _ in pairs:
        if output_dir / f'{input_path.stem}.png' in sources:
            raise ValueError('Output would overwrite an input/target image; choose another output directory.')
    model = get_models(config)['MSLKAD']
    load_weights(model, args.weights)
    model.to(device).eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    scores = []
    with torch.inference_mode():
        for input_path, target_path in pairs:
            inputs = image_tensor(read_rgb(input_path)).unsqueeze(0).to(device)
            target = image_tensor(read_rgb(target_path)).unsqueeze(0) if target_path else None
            if target is not None and inputs.shape != target.shape:
                raise ValueError(f'Image dimensions differ: {input_path} {tuple(inputs.shape)}; '
                                 f'{target_path} {tuple(target.shape)}')
            output = forward_image(model, inputs, args.tile_size, args.tile_overlap)
            if target is not None:
                psnr, ssim = y_metrics(output, target)
                scores.append((input_path.name, psnr, ssim))
                print(f'{input_path.name}: PSNR(Y)={psnr:.4f} dB  SSIM(Y)={ssim:.6f}')
            save_tensor(output, output_dir / f'{input_path.stem}.png')
    if scores:
        with (output_dir / 'metrics.csv').open('w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['filename', 'psnr_y', 'ssim_y'])
            writer.writerows(scores)
        print(f'Mean over {len(scores)} images: PSNR(Y)={sum(row[1] for row in scores) / len(scores):.4f} dB  '
              f'SSIM(Y)={sum(row[2] for row in scores) / len(scores):.6f}')
        print('Metrics use raw float predictions, BT.601 Y, crop=0, and the repository SSIM implementation.')
    print(f'Saved {len(pairs)} restored image(s) to {output_dir}')


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run(args)
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        parser.exit(1, f'Error: {error}\n')


if __name__ == '__main__':
    main()
