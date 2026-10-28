# Implementation and reproducibility

This release combines the supplied author model implementation with the repository's training code. The [paper](paper/MSLKAD_ICIP2026.pdf) and [figures](../assets/README.md) describe the original experiments. The release checks validate code execution; they do not constitute a fresh reproduction of the six benchmark results.

## Architecture

The released model preserves the supplied source's parameter names and computation, including the additional residual connection after every two MFABs. The default configuration has **4 groups × 9 blocks × 180 channels**.

Some manuscript labels differ from the source:

| Item | Manuscript | Supplied model source |
| :--- | :--- | :--- |
| GKG depthwise kernel in Fig. 1 | 3 × 3 | 7 × 7 |
| EKA dilated convolution in Fig. 1 | Dilation 4 | Dilation 3 |

The code follows the supplied source for these settings. The paper images and reported values are retained as provided. Changing these settings to match the schematic can break compatibility with the supplied model and its checkpoints.

## Dataset labels and pairing

Table 1 uses **Rain200L/H**, while the synthetic comparison caption and the ablation tables use **Rain100L/H**. The README retains the labels of the source figures. These names should not be treated as equivalent test splits.

Training and paired evaluation match rainy and clean images by **filename stem**. Extensions may differ, but each stem must occur exactly once in each directory. Rename files or prepare a separate paired directory if a dataset uses different rainy/clean naming conventions. Missing images, duplicate stems and mismatched dimensions are errors.

## Training settings

- AdamW, learning rate 1e-4, betas (0.9, 0.999), weight decay 1e-4.
- Cosine decay to 1e-6; 100K iterations by default and 400K with `configs/spa.yaml`.
- Batch size 8 and paired 128 × 128 patches.
- Loss weights: pixel L1 = 1.0, perceptual = 0.7, edge = 0.15.
- A local ImageNet-pretrained torchvision VGG19 state dict is required for training. Inference does not initialize VGG19.
- The provided trainer uses one process with `torch.nn.DataParallel` for multiple CUDA GPUs. Run it with `python main.py` and a GPU list; there is no DDP launcher in this release.

The manuscript describes an L1 perceptual constraint, while the historical training implementation averages **MSE** over its selected VGG19 features. The release retains that implementation's loss definition. Its edge loss uses a Laplacian pyramid with a Charbonnier penalty. A detached `Variable` wrapper in the original perceptual loss has been removed so that this loss contributes gradients to the deraining model.

## Evaluation protocol

The evaluator follows the repository's existing metric implementation: floating-point RGB tensors are converted to the **BT.601 Y channel**, without border cropping. Metrics use the raw floating-point prediction. Saved PNGs are separately clamped to [0, 1] and quantized, so recomputing metrics from saved PNGs may produce different values.

SSIM uses the existing Gaussian implementation with replicated borders. Match this protocol when comparing results; a different SSIM border convention, RGB evaluation, output clamping or quantization can change the reported values.

Full-image inference is the default. Explicit tiled inference is available for large images; tile boundaries can affect predictions because the model contains large spatial kernels. Do not assume a tiled run reproduces a full-image result.

## Checkpoints

`test.py` loads model tensors strictly and rejects missing, unexpected or mismatched parameters. It supports the repository's nested `models.MSLKAD` format, raw state dictionaries and documented wrapper formats, including DataParallel prefixes. A checkpoint for a different architecture must not be silently partially loaded.

For training, use `weights=PATH resume=true` to restore the saved model, optimizer, scheduler and iteration counter. Use `weights=PATH` without `resume=true` to initialize model weights for a new run. A training checkpoint is required to resume optimizer and scheduler state. Resume restores the numeric step and best score, but does not restore the shuffled data-loader position or random generator progression; it does not promise bit-for-bit replay of stochastic training.

Use the original scheduler configuration when resuming a historical run. For example, an old MultiStepLR checkpoint requires its original milestones and gamma; it cannot be resumed with the new default cosine scheduler.

The original Baidu Netdisk model links are retained in the README. The release was checked with generated test checkpoints; the remotely hosted pretrained checkpoints and complete benchmark datasets were not downloaded for this release check.

## Release validation

The local check environment uses **Python 3.14.3, PyTorch 2.14.1, torchvision 0.29.1**, and CPU execution on macOS. The original README recorded Python 3.9, PyTorch 1.13.1 and CUDA 11.7 as the research environment. CUDA execution and a full benchmark reproduction require the corresponding hardware, datasets and pretrained checkpoints.

The focused checks can be run with:

```bash
python -m pip install pytest
python -m pytest -q
python test.py --help
```
