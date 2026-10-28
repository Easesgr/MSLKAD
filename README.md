<div align="center">

# MSLKAD

### Multi-Scale Large Kernel Attention for Single-Image Deraining

**Accepted to IEEE ICIP 2026**

Congcong Zeng · Dan Xu · Yinghui Zhu · Jiangang Pan · Kangjian He · Hongzhen Shi*

School of Information Science and Engineering, Yunnan University

[![ICIP 2026](https://img.shields.io/badge/ICIP_2026-Accepted-147d64)](docs/paper/MSLKAD_ICIP2026.pdf)
[![PyTorch](https://img.shields.io/badge/PyTorch-Implementation-EE4C2C?logo=pytorch&logoColor=white)](model/MSLKAD.py)
[![Paper](https://img.shields.io/badge/Paper-PDF-3b657a)](docs/paper/MSLKAD_ICIP2026.pdf)

[Paper](docs/paper/MSLKAD_ICIP2026.pdf) · [Pretrained models](#pretrained-models) · [Quick start](#quick-start) · [中文说明](README_zh-CN.md)

</div>

**MSLKAD** is a convolutional network for removing spatially varying rain streaks from a single image. It combines multi-scale large-kernel attention with local feature gating to recover image details and model long-range context in a unified feature space.

## News

- **ICIP 2026:** Our paper has been accepted to the IEEE International Conference on Image Processing.
- **Code release:** The model, training and inference code, paper figures, and usage instructions are now included in this repository.

## Method

![Overview of MSLKAD and its MLA, GKG and EKA modules](assets/figures/architecture.png)

MSLKAD uses four Multi-Feature Attention Blocks, each with nine MLA–GKG pairs, and 180 feature channels.

- **MLA — Multi-Scale Large-Kernel Attention:** combines three kernel groups, 3–5–1, 5–7–1 and 7–9–1, to aggregate context at different scales.
- **GKG — Ghost Kernel Gate:** selects local features through spatial gating to suppress rain and preserve texture.
- **EKA — Expanded Large-Kernel Attention:** aggregates spatial context at the end of each block.

[Vector architecture figure](assets/figures/architecture.pdf) · [Implementation and reproducibility notes](docs/reproducibility.md)

## Visual results

### Synthetic rain

![Synthetic deraining comparison: input, DRSformer, MSDT, NeRD-Rain, MSLKAD and ground truth](assets/figures/synthetic-comparison.png)

Comparisons from Fig. 2 of the paper. Its caption labels the top and bottom rows Rain100H and Rain100L, respectively. **Ours** denotes MSLKAD; **GT** denotes ground truth.

### Real-world rain

![Real-world deraining comparison: input, DualGCN, DRSformer, UDR-S2Former, MSLKAD and ground truth](assets/figures/real-world-comparison.png)

Comparisons on **RealRain-1k-H** (top) and **RealRain-1k-L** (bottom), from Fig. 3. Insets show the corresponding local details.

## Pretrained models

| Dataset | Type | Pretrained model | Access code |
| :--- | :--- | :---: | :---: |
| Rain200L | Synthetic | [Baidu Netdisk](https://pan.baidu.com/s/1CMVHZyQOieoO8Q4Go2e3Kw?pwd=ipnu) | `ipnu` |
| Rain200H | Synthetic | [Baidu Netdisk](https://pan.baidu.com/s/1SvZTKXpIJimpUS5JKt390A?pwd=4xrj) | `4xrj` |
| SPA-Data | Real | [Baidu Netdisk](https://pan.baidu.com/s/1NY39HhT0n1HHU5Ynwjvh-w?pwd=ziym) | `ziym` |
| RealRain-1k-L | Real | [Baidu Netdisk](https://pan.baidu.com/s/1X1qFBNBD-zeB6VnVhBJ8pg?pwd=pb2p) | `pb2p` |
| RealRain-1k-H | Real | [Baidu Netdisk](https://pan.baidu.com/s/1yzy-tF6bXA1aAl1c6sIunQ?pwd=47gu) | `47gu` |
| LHP-Rain | Real | [Baidu Netdisk](https://pan.baidu.com/s/1roCSCPpqpha7B78qe2gUgw?pwd=gxuk) | `gxuk` |

The downloads retain the repository's original model links. Save the downloaded checkpoint under `pretrained/` and select it with `--weights`.

## Quick start

### Installation

```bash
git clone https://github.com/Easesgr/MSLKAD.git
cd MSLKAD
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Install a PyTorch build compatible with your CUDA version for GPU training or inference. The original research environment used Python 3.9, PyTorch 1.13.1 and CUDA 11.7. See [release validation](docs/reproducibility.md#release-validation) for the environment used to check this release.

### Inference

Download one of the pretrained models above, then run:

```bash
# A folder of rainy images
python test.py \
  --weights pretrained/Rain200H.pt \
  --input datasets/Rain200H/test/LR \
  --output outputs/Rain200H

# A single image on CPU
python test.py \
  --weights pretrained/Rain200H.pt \
  --input path/to/rainy.png \
  --output outputs/single \
  --device cpu
```

The output directory receives restored PNG images. Inference accepts a rainy image or a directory and does not require ground truth or VGG19 weights.

## Training

Arrange paired RGB images as follows. `LR` contains rainy inputs and `HR` contains clean targets.

```text
datasets/Rain200H/
├── train/
│   ├── LR/001.png
│   └── HR/001.png
└── test/
    ├── LR/002.png
    └── HR/002.png
```

Provide a local **ImageNet-pretrained torchvision VGG19 state dict** for the perceptual loss ([official weights](https://download.pytorch.org/models/vgg19-dcbb9e9d.pth)). The training code requires those weights explicitly.

```bash
# One GPU
python main.py \
  data.train_data_root=./datasets/Rain200H/train \
  data.val_data_root=./datasets/Rain200H/test \
  train.vgg_model_path=./pretrained/vgg19.pth \
  'train.gpus=[0]'

# Four GPUs, one process with DataParallel
python main.py \
  data.train_data_root=./datasets/Rain200H/train \
  data.val_data_root=./datasets/Rain200H/test \
  train.vgg_model_path=./pretrained/vgg19.pth \
  'train.gpus=[0,1,2,3]'
```

[Default configuration](configs/default.yaml): 100K iterations, batch size 8, 128 × 128 patches, AdamW, and cosine decay from 1e-4 to 1e-6. For SPA-Data's 400K schedule, add `configs/spa.yaml` immediately after `main.py` and set the dataset paths accordingly.

```bash
# Resume a saved training run, including optimizer and scheduler state
python main.py weights=./runs/RUN_NAME/latest.pt resume=true \
  data.train_data_root=./datasets/Rain200H/train \
  data.val_data_root=./datasets/Rain200H/test \
  train.vgg_model_path=./pretrained/vgg19.pth \
  'train.gpus=[0]'
```

Checkpoints and logs are written under `runs/`. For a CPU smoke run, add `sys.cpu=true`; full training is intended for CUDA GPUs.

## Repository guide

| Path | Contents |
| :--- | :--- |
| [`model/`](model/) | Author-supplied MSLKAD architecture and model factory |
| [`test.py`](test.py) | Standalone inference and paired evaluation |
| [`main.py`](main.py), [`trainer/`](trainer/) | Training entry point and training loop |
| [`configs/`](configs/) | Portable training configurations |
| [`data/`](data/), [`loss/`](loss/), [`util/`](util/) | Paired datasets, losses, metrics and checkpoint utilities |
| [`assets/`](assets/) | Architecture and visual comparisons in PNG/PDF |
| [`docs/paper/`](docs/paper/) | Author manuscript and citation |
| [`tests/`](tests/) | Focused release checks |

## Citation

If MSLKAD is useful in your research, please cite:

```bibtex
@inproceedings{zeng2026mslkad,
  title     = {Multi-Scale Large Kernel Attention for Single-Image Deraining},
  author    = {Zeng, Congcong and Xu, Dan and Zhu, Yinghui and Pan, Jiangang and He, Kangjian and Shi, Hongzhen},
  booktitle = {IEEE International Conference on Image Processing (ICIP)},
  year      = {2026},
  note      = {Accepted}
}
```

## Acknowledgments and contact

We acknowledge [MAN](https://github.com/icandle/MAN) and [VAN](https://github.com/Visual-Attention-Network) for their component implementations. See [third-party notices](THIRD_PARTY_NOTICES.md).

Corresponding author: **Hongzhen Shi**, [hongzhen.shi@ynu.edu.cn](mailto:hongzhen.shi@ynu.edu.cn). For questions about the code, please [open an issue](https://github.com/Easesgr/MSLKAD/issues).
