# FOD YOLO26n Training Pipeline

This repository is the implementation workspace for Phase 1 of the FOD detection project. It will provide a reproducible pipeline for converting the Kaggle FOD-A dataset to a single-class YOLO dataset, training and evaluating YOLO26n, promoting a validation-selected champion, publishing verified model artifacts, and running image or video inference.

The implementation contract is defined in [`FOD_YOLO26n_Phase1_Technical_Specification.md`](FOD_YOLO26n_Phase1_Technical_Specification.md).

## Current status

Part 1 establishes the repository and packaging foundation. Dataset processing, training, evaluation, model promotion, publication, and inference are not implemented yet.

No model-accuracy claims are made until a real training and evaluation run produces metrics.

## Python compatibility policy

- Local development currently uses Python 3.14.
- Project code targets Python 3.11 or newer so a compatible Python installation on the training system can be used.
- Python, PyTorch, CUDA, cuDNN, Ultralytics, and dependency versions will be recorded for each training run.
- PyTorch and torchvision are installed separately for the target machine because their packages depend on the operating system, Python version, accelerator, and CUDA build.

## Development environment

On the current Windows development machine:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

Development and runtime dependencies will be installed as their corresponding project parts are implemented. See [`requirements/README.md`](requirements/README.md) for the dependency policy.

## Package layout

Business logic lives under `src/fod_yolo`. The top-level `scripts` directory will contain thin command-line wrappers only. Generated datasets, runs, reports, model artifacts, and secrets are excluded from ordinary Git commits.
