# Installation Guide

This guide will help you set up the development environment for the Visual Localization project using Conda.

## Prerequisites

- [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed
- Git installed
- CUDA-capable GPU (recommended for better performance)

## Installation Steps

1. Clone the repository:
```bash
git clone https://github.com/ALFONSOBUGRA/VisLoc.git
cd visloc
```

2. Create and activate a new conda environment:
```bash
conda create -n visloc python=3.9 -y
conda activate visloc
```

3. Install PyTorch with CUDA support (if you have a CUDA-capable GPU):
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

4. Install the required packages:
```bash
pip install -r requirements.txt
```

5. Install the package in development mode:
```bash
pip install -e .
```

## Additional Dependencies

For development purposes, you can install additional development dependencies:
```bash
pip install -e ".[dev]"
```

For linting and testing:
```bash
pip install -e ".[lint,test]"
```

For documentation:
```bash
pip install -e ".[docs]"
```

## Verifying Installation

To verify the installation, you can run:
```bash
python -c "import visloc; print(visloc.__version__)"
```

## Troubleshooting

1. If you encounter CUDA-related issues:
   - Make sure you have the correct CUDA version installed
   - Try installing PyTorch without CUDA support

2. If you encounter package conflicts:
   - Try creating a fresh conda environment
   - Make sure you're using Python 3.9 or compatible version

3. For OpenCV-related issues:
   - Make sure you have the correct version installed as specified in requirements.txt
   - On Windows, you might need to install Visual C++ Build Tools

## Support

If you encounter any issues during installation, please:
1. Check the troubleshooting section
2. Search for similar issues in the project's issue tracker
3. Create a new issue with detailed information about your problem 