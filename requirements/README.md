# Dependency installation

The project separates platform-neutral dependencies from PyTorch because the correct PyTorch and torchvision packages depend on the Python version, operating system, accelerator, and CUDA environment of the machine running the code.

Do not copy a CUDA-specific PyTorch command from one machine to another.

## Development tools

```powershell
python -m pip install -r requirements/dev.txt
```

## Runtime packages

After installing the PyTorch build appropriate for the target system:

```powershell
python -m pip install -r requirements/base.txt
```

The future `scripts/install_torch.py` command will make the machine-specific installation explicit. A completed training run will save an environment freeze for reproducibility; that freeze describes the original environment and is not intended to force an incompatible CUDA build onto another device.
