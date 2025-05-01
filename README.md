# GPU Linear Regression (Python + CuPy + CUDA)

This is a small experimental project where I'm learning and playing around with training linear regression models directly on the GPU using Python.

The goal is to get a better grasp of how CUDA-based computations work in practice — both using high-level tools like CuPy and by writing and launching custom CUDA kernels myself.

## Features

- **Dataset selection**: use either synthetic data or real-world data (House Prices dataset from OpenML).
- **Flexible training**: switch between CuPy operations or custom CUDA kernels (RawKernel).
- **Real-time visualization**: plots of data points, regression line fitting, and loss curve using VisPy.
- **Configuration via YAML**: control dataset, epochs, learning rate, loss function, and execution mode from `config.yml`.
- **CUDA acceleration**: all computations happen directly on the GPU.

## Requirements

- Python 3.8+
- CuPy (installed matching your CUDA version)
- VisPy
- scikit-learn
- PyYAML

## How to Run

```bash
#----- clone repo
git clone https://github.com/d1akon/cuda_lin_reg.git
cd cuda_lin_reg

#----- install dependencies
pip install cupy-cuda12x vispy scikit-learn pyyaml

#----- run 
python main.py
