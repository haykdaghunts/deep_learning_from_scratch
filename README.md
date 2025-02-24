# Deep Learning Framework from Scratch

This repository contains a deep learning framework implemented from scratch using NumPy. The goal of this project is to mimic the PyTorch pipeline, providing a basic but functional deep learning framework for educational purposes. The usage of this framework is demonstrated in the `main.ipynb` file.

## Overview

This project includes the following components:

- **Data Loading and Preprocessing**: `dataloader.py` and `dataset.py` provide tools for loading and preprocessing Cifar10 dataset.
- **Neural Network Layers**: `layers.py` contains various neural network layers such as Sigmoid, ReLU, Tanh, and Affine.
- **Loss Functions**: `loss_functions.py` includes implementations of common loss functions like L1, MSE, BCE, and CrossEntropy.
- **Optimizers**: `optimizers.py` provides different optimization algorithms such as SGD, SGD with Momentum, and Adam.
- **Transformations**: `transforms.py` includes various image transformations like rescaling, normalization, flattening, and composition of multiple transforms.

## Files

### dataloader.py
Defines a `DataLoader` class for creating iterable batch-samplers over a given dataset.
- **Key Features**:
  - Batch sampling
  - Shuffling
  - Dropping the last incomplete batch (optional)

### dataset.py
Implements an `ImageFolderDataset` class for loading and preprocessing CIFAR10 dataset.
- **Key Features**:
  - Directory-based dataset loading
  - Train/val/test splitting
  - Image transformations

### layers.py
Contains implementations of various neural network layers.
- **Layers**:
  - **Sigmoid**: Sigmoid activation function.
  - **ReLU**: Rectified Linear Unit activation function.
  - **Tanh**: Hyperbolic tangent activation function.
  - **Affine**: Fully connected (dense) layer.
  - **Model**: A container class for stacking layers and performing forward and backward passes.

### loss_functions.py
Implements common loss functions.
- **Loss Functions**:
  - **L1**: L1 loss (mean absolute error).
  - **MSE**: Mean squared error.
  - **BCE**: Binary cross-entropy.
  - **CrossEntropyFromLogits**: Cross-entropy loss from logits.

### optimizers.py
Provides different optimization algorithms.
- **Optimizers**:
  - **SGD**: Stochastic Gradient Descent.
  - **SGD_Momentum**: SGD with momentum.
  - **Adam**: Adam optimizer.

### transforms.py
Contains various image transformation classes.
- **Transforms**:
  - **RescaleTransform**: Rescales images to a specified range.
  - **NormalizeTransform**: Normalizes images using mean and standard deviation.
  - **FlattenTransform**: Flattens images into 1D arrays.
  - **ComposeTransform**: Combines multiple transforms into one.

### dataset_download.py
Includes the link to download dataset and the script that converts downlaoded files into compatible format.
