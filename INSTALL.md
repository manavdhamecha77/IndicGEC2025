# Installation Instructions for IndicGEC2025

This project contains Hindi Grammatical Error Correction models and notebooks for the IndicGEC2025 shared task.

## Requirements Files

We provide two requirements files:

### 1. `requirements.txt` (Recommended)
- Contains minimum version requirements
- More flexible and future-compatible
- Use this for general installation

```bash
pip install -r requirements.txt
```

### 2. `requirements-exact.txt` (For exact reproduction)
- Contains exact pinned versions from the working environment
- Use this if you need to reproduce the exact same environment

```bash
pip install -r requirements-exact.txt
```

## Installation Methods

### Method 1: Using Conda (Recommended)

```bash
# Create a new conda environment
conda create -n gec python=3.10

# Activate the environment
conda activate gec

# Install PyTorch with CUDA support (if you have a GPU)
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

# Install other requirements
pip install -r requirements.txt
```

### Method 2: Using pip only

```bash
# Create virtual environment
python -m venv gec_env

# Activate virtual environment
# On Windows:
gec_env\\Scripts\\activate
# On Linux/Mac:
source gec_env/bin/activate

# Install requirements
pip install -r requirements.txt
```

## GPU Support

This project requires CUDA-capable GPU for training. Make sure you have:
- CUDA 12.4 or compatible version
- PyTorch with CUDA support installed

## Verification

To verify your installation, run:

```python
import torch
import transformers
import pandas as pd

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Transformers version: {transformers.__version__}")
print(f"Pandas version: {pd.__version__}")
print("✅ Installation successful!")
```

## Project Structure

- `Hindi/mt5-hindi-gec-model/` - Trained MT5 model files
- `Hindi/dev.csv` - Development dataset
- `Hindi/train.csv` - Training dataset (if available)
- Training notebooks and scripts for MT5 model

## Branches

- `main` - Latest code and notebooks
- `mt5-model` - Contains trained MT5 model with Git LFS

## Support

If you encounter any installation issues, please check:
1. Python version compatibility (3.8-3.11 recommended)
2. CUDA version compatibility
3. Available disk space (models are large)
4. Internet connection for downloading models