# TransCNN-HAE+

This repository provides the official PyTorch implementation of our paper "G2LFormer:Global-to-Local Token Mixing Transformer for Blind Image Inpainting and Beyond".

Here we provide the PyTorch implementation of our latest version, if you require the code of our previous ACM MM version (**["TransCNN-HAE: Transformer-CNN Hybrid AutoEncoder for Blind Image Inpainting"](https://dl.acm.org/doi/pdf/10.1145/3503161.3547848)**), please click the **[released version](https://github.com/zhenglab/TransCNN-HAE/releases/tag/v1.0)**.

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

- Train our TransCNN-HAE+:

```
python train.py --path=./checkpoints/config.yml
```

- Train our TransCNN-HAE+wCIA:

```
python train.py --path=./checkpoints/configwithCIA.yml
```

### Testing

The model is automatically saved every 10,000 iterations, please rename the file `g.pth_$iter_number$` to `g.pth` and then run testing command.

- Test our TransCNN-HAE+:

```
python test.py --path=./checkpoints/config.yml
```

- Test our TransCNN-HAE+wCIA:

```
python test.py --path=./checkpoints/configwithCIA.yml
```

### Pre-trained Models

We will release our pre-trained models soon.