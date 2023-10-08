# TransCNN-HAE+

This repository provides the official PyTorch implementation of our paper "G2LFormer:Global-to-Local Token Mixing Transformer for Blind Image Inpainting and Beyond".


## Prerequisites

- Linux
- Python 3.7
- NVIDIA GPU + CUDA CuDNN

## Getting Started


### Installation

- Clone this repo:
```bash
git clone https://github.com/zhenglab/TransCNN-HAE.git
cd TransCNN-HAE
```

- Install [PyTorch](http://pytorch.org) and 1.7 and other dependencies (e.g., torchvision).
  - For Conda users, you can create a new Conda environment using `conda create --name <env> --file requirements.txt`.

### Training

Please change the pathes to your dataset path in `datasets` folder.

```
python train.py --path=$configpath$

For example: python train.py --path=./checkpoints/FFHQ/
```

### Testing

The model is automatically saved every 10,000 iterations, please rename the file `g.pth_$iter_number$` to `g.pth` and then run testing command.
```
python test.py --path=$configpath$ 

For example: python test.py --path=./checkpoints/FFHQ/
```
