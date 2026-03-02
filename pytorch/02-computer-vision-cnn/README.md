# Computer Vision with CNNs

An introduction to Convolutional Neural Networks (CNNs) for image processing and computer vision tasks using PyTorch.

## Learning Path

### Main Track

Work through these notebooks in order:

| # | Notebook | Description |
|---|----------|-------------|
| 1 | [Convolutional Networks Theory](01_convolution_theory.ipynb) | Core concepts: convolution operations, filters (kernels), padding, stride, and pooling layers (`nn.Conv2d`, Max Pooling vs Avg Pooling) |
| 2 | [MNIST with CNNs](02_cnn_mnist.ipynb) | Application of CNNs to the MNIST dataset, including data transformations, image normalization, and PyTorch dataloaders |
| 3 | [CIFAR-10 Classification](CIFAR-10/CIFAR-10.ipynb) | Advanced image classification on color images: handling 3-channel data, deeper architectures, baseline models, optimization techniques, and evaluation metrics |


## Production Examples

The model created in the [CIFAR-10 Classification](CIFAR-10/CIFAR-10.ipynb) notebook has been deployed as production-ready applications on Hugging Face Spaces:

- **CIFAR-10 FastAPI & Docker**: A REST API built with FastAPI and containerized using Docker.
  - GitHub Repository: [avidaldo/cifar-10-fastapi](https://github.com/avidaldo/cifar-10-fastapi)
  - Live Deployment: [Hugging Face Space](https://huggingface.co/spaces/avidaldo/cifar-10-fastapi)

- **CIFAR-10 Gradio**: An interactive web interface directly deployed using Gradio.
  - Live Deployment: [Hugging Face Space](https://huggingface.co/spaces/avidaldo/cifar-10-fastapi-gradio)

## Prerequisites

- Completion of the Deep Learning Fundamentals module
- Solid understanding of PyTorch basic constructs (tensors, `nn.Module`, dataloaders, and the standard training loop)
- Basic knowledge of multi-dimensional arrays and image representation
