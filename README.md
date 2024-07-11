# Improving the Parameterization of Complex Subsurface Flow Properties with Style-Based Generative Adversarial Network (StyleGAN)

## Keypoint

* **Style-based Generative Adversarial Network models improve the parameterization and calibration of complex subsurface flow models.**
* **Ensemble Smoother with Multiple Data Assimilation is used to demonstrate the superior performance of StyleGAN compared to GAN and VAE.**
* **StyleGAN architecture improves latent domain regularity, leading to enhanced parameterization and improved calibration results.**

## Abstract

Representing and preserving complex (non-Gaussian) spatial patterns in aquifer flow properties during model calibration are challenging. Conventional parameterization methods that rely on linear/Gaussian assumptions are not suitable for representation of property maps with more complex spatial patterns. Deep learning techniques, such as Variational Autoencoders (VAE) and Generative Adversarial Networks (GAN), have recently been proposed to address this difficulty by learning complex spatial patterns from prior training images and synthesizing similar realizations using low-dimensional latent variables with Gaussian distributions. The resulting Gaussian latent variables lend themselves to calibration with Kalman filter updating schemes that are suitable for parameters with Gaussian distribution. Despite their superior performance in generating complex spatial patterns, these generative models may not provide desirable properties that are needed for parameterization of model calibration problems, including robustness, smoothness in latent domain, and reconstruction fidelity. This paper introduces the second generation of style-based Generative Adversarial Networks (StyleGAN) for parameterization of complex subsurface flow properties and compares its model calibration properties and performance with the convolutional VAE and GAN architectures. Numerical experiments involving model calibration with the Ensemble Smoother with Multiple Data Assimilation (ES-MDA) in single-phase and two-phase fluid flow examples are used to assess the capabilities and limitations of these methods. The results show that parameterization with StyleGANs provides superior performance in terms of reconstruction fidelity and flexibility, underscoring the potential of StyleGANs for improving the representation and reconstruction of complex spatial patterns in subsurface flow model calibration problems.

## Code

### Prerequisites

Python 3.7.12

Torch==1.8.0+cu111

MATLAB

The MATLAB Reservoir Simulation Toolbox (MRST)
## Data
## Citation