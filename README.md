# DiffuMNIST: Advanced Diffusion Models for MNIST Digits

![digit_5_process_epoch_20](https://github.com/user-attachments/assets/91876d39-c4df-4ee7-9cbf-4e37a5e9b089)


> A PyTorch implementation of Denoising Diffusion Probabilistic Models (DDPM) for generating MNIST digits with both time-conditioning and class-conditioning capabilities

## Project Overview

This project implements a state-of-the-art diffusion model architecture for generating high-quality MNIST digits. The implementation progressively builds from a basic denoising UNet to a fully-featured diffusion model with class-conditional generation.

The diffusion model works by gradually adding noise to an image and then training a neural network to reverse this process, learning to generate images by starting from pure noise. This approach has led to groundbreaking results in image generation.

## Architecture & Implementation Details

### Core Components:

1. **Unconditional UNet Architecture**: A base model with encoder-decoder structure and skip connections
2. **Time-Conditioned Diffusion**: Implements the DDPM paper's forward and reverse diffusion processes
3. **Class-Conditioned Generation**: Extends the model to generate specific digits (0-9) with classifier-free guidance

### Technical Specifications:

- **Model Design**: Implemented a UNet with downsampling/upsampling blocks and skip connections
- **Conditioning Methods**: Added time and class embeddings through specialized fully-connected layers
- **Sampling Process**: Implemented the DDPM reverse process with classifier-free guidance (γ=5.0)
- **Hyperparameters**:
  - Learning Rate: 1e-3 with exponential decay
  - Hidden Dimension: 64 channels
  - Batch Size: 128
  - Timesteps: 300
  - Beta Schedule: Linear from 1e-4 to 0.02

## Features & Results

### Single-Step Denoising

- Implemented a basic UNet to denoise images corrupted with Gaussian noise (σ=0.5)
- Evaluated the model's performance on out-of-distribution noise levels (σ ranging from 0.0 to 1.0)

### Time-Conditioned Diffusion

- Trained a diffusion model over 20 epochs that can generate MNIST digits from pure noise
- Implemented the complete DDPM noise schedule and sampling algorithm
- Visualized the generation process through time

![image](https://github.com/user-attachments/assets/6ddc2b6d-b833-49b3-aeb0-b318c746f726)


### Class-Conditioned Generation

- Extended the model with class conditioning to generate specific digits
- Implemented classifier-free guidance for improved sample quality
- Created a complete interactive visualization of the generation process

![image](https://github.com/user-attachments/assets/f7741b5e-3b03-43ae-bb6e-55397c98533f)


## Visualizations & Analysis

### Generation Process Animation

The project includes dynamic visualizations showing:
- The progressive denoising process from noise to digit
- Comparison of generation quality across training epochs
- Class-conditional generation for all 10 digits

## Implementation Pipeline

1. **Initial Denoising**: Single-step denoising model with L2 loss
2. **DDPM Implementation**: Forward diffusion process and reverse sampling algorithm
3. **Time Conditioning**: Extension with timestep embeddings using FCBlocks
4. **Class Conditioning**: One-hot encoded class labels with classifier-free guidance
5. **Visualization**: Advanced plotting and GIF generation of the sampling process

## Getting Started

### Dependencies

```
torch
torchvision
matplotlib
numpy
imageio (for GIF creation)
tqdm
```

### Training Models

The notebook contains complete code for:
- Training the single-step denoiser
- Training the time-conditioned diffusion model
- Training the class-conditioned diffusion model
- Generating samples and visualizations

### Sampling from Models

```python
# Example: Generating a specific digit using class conditioning
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
digit = 5  # Generate digit "5"
c = torch.tensor([digit], device=device)
samples, animation = model.sample(
    c, 
    img_wh=(28, 28), 
    guidance_scale=5.0,
    seed=42
)
```

## Advanced Features

- **Classifier-Free Guidance**: Implemented with guidance scale γ=5.0 for higher quality samples
- **Animation Generation**: Custom functions to create GIFs of the diffusion process
- **Out-of-Distribution Testing**: Evaluated the model's robustness to varied noise levels

## Background Theory

The implementation follows the architecture and methodology described in the [DDPM paper](https://arxiv.org/abs/2006.11239) by Ho et al. The model works by:

1. Gradually adding noise to images according to a fixed schedule
2. Training a neural network to predict the noise at each step
3. Sampling new images by starting with random noise and iteratively denoising

## Credits

This project was created as part of CS 444 Assignment 5, adapted from UC Berkeley's CS180 Project 5b. The original assignment was a joint effort by Daniel Geng, Ryan Tabrizi, and Hang Gao, advised by Liyue Shen, Andrew Owens, and Alexei Efros.
