# MSLKAD

The main code and path files of MSLKAD have been uploaded, while the **/models** module is not yet included. The **pretrained models** are already available, and the remaining files, including the models module and the improved **test.py** code, will be supplemented after the paper is accepted.

## Usage

### Installation

This code was tested with the following environment configurations. It may work with other versions.

- CUDA 11.7
- Python 3.9
- Pytorch 1.13.1+cu117

### Training

```
# training with multi GPUs
torchrun --nproc_per_node=4 --master_port=2222 main.py 

# training with a single GPU
python main.py 
```

### Testing

```
python test.py 
```

## Pre_train_model

| Synthetic or Real | Dataset       | PSNR  | SSIM   | Model                                                        |
| ----------------- | ------------- | ----- | ------ | ------------------------------------------------------------ |
| Synthetic         | Rain200H      | --.-- | ------ | [Download](https://pan.baidu.com/s/1SvZTKXpIJimpUS5JKt390A?pwd=4xrj) |
| Synthetic         | Rain200L      | --.-- | ------ | [Download](https://pan.baidu.com/s/1CMVHZyQOieoO8Q4Go2e3Kw?pwd=ipnu) |
| Real              | RealRain-1k-H | --.-- | ------ | [Download](https://pan.baidu.com/s/1yzy-tF6bXA1aAl1c6sIunQ?pwd=47gu) |
| Real              | RealRain-1k-L | --.-- | ------ | [Download](https://pan.baidu.com/s/1X1qFBNBD-zeB6VnVhBJ8pg?pwd=pb2p) |
| Real              | SPA-Data      | --.-- | ------ | [Download](https://pan.baidu.com/s/1NY39HhT0n1HHU5Ynwjvh-w?pwd=ziym) |
| Real              | LHP-Rain      | --.-- | ------ | [Download](https://pan.baidu.com/s/1roCSCPpqpha7B78qe2gUgw?pwd=gxuk) |
