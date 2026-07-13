# FOD YOLO26n Training Pipeline

This repository is the implementation workspace for Phase 1 of the FOD detection project. It will provide a reproducible pipeline for converting the Kaggle FOD-A dataset to a single-class YOLO dataset, training and evaluating YOLO26n, promoting a validation-selected champion, publishing verified model artifacts, and running image or video inference.

The implementation contract is defined in [`FOD_YOLO26n_Phase1_Technical_Specification.md`](FOD_YOLO26n_Phase1_Technical_Specification.md).

## Current status

The repository now includes packaging, configuration, portable paths, hashing, atomic artifact writes, logging, environment diagnostics, and the complete dataset download/preparation/validation pipeline. Training, evaluation, model promotion, publication, and inference are not implemented yet.

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

## Configuration foundation

The five version-controlled YAML files under `configs/` define dataset preparation, training, evaluation, champion promotion, and video inference defaults. `fod_yolo.config` loads them with safe YAML parsing and supports validated dotted overrides such as:

```text
training.batch=4
training.device=0
```

`fod_yolo.paths.ProjectPaths` resolves repository-relative defaults. `FOD_DATA_ROOT`, `FOD_RUNS_ROOT`, and `FOD_ARTIFACTS_ROOT` can relocate generated files on another machine without changing source code or committed configuration.

## Environment and PyTorch setup

Use the active Python interpreter on each machine. Select the correct PyTorch wheel profile from the official PyTorch installation guidance for that machine's operating system and CUDA environment:

```powershell
python scripts/install_torch.py --profile <official-profile> --dry-run
python scripts/install_torch.py --profile <official-profile>
python -m pip install -r requirements/base.txt
python scripts/check_environment.py --require-cuda
```

`install_torch.py` accepts only official `download.pytorch.org` wheel indexes and installs into `sys.executable`, so it does not assume the development machine's Python version. `check_environment.py` writes `reports/environment_report.json` with Python, package, Git, CUDA, cuDNN, GPU, and `nvidia-smi` details. Use `--skip-model-check` only for diagnostics that intentionally do not resolve `yolo26n.pt`.

Project logs use ISO-8601 UTC timestamps and automatically redact `KAGGLE_KEY` and `GH_TOKEN`. Long-running commands can opt into rotating file logs through `fod_yolo.logging_utils.configure_logging`.

## Dataset pipeline

Install the official Kaggle CLI into the active interpreter and configure either the `KAGGLE_USERNAME`/`KAGGLE_KEY` pair or a local `kaggle.json`. Do not put credentials in the repository.

```powershell
python -m pip install kaggle
python scripts/download_dataset.py --config configs/dataset.yaml
python scripts/prepare_dataset.py --config configs/dataset.yaml
python scripts/validate_dataset.py --data data/processed/fod_a_single_class_yolo/fod_a.yaml --strict
```

The downloader hashes the source archive, writes `source_manifest.json`, rejects unsafe ZIP paths, extracts through a staging directory, and reuses a valid cached archive unless `--force` is supplied. Preparation recursively discovers the Pascal VOC root, preserves the official test split, deterministically divides `trainval` with seed 42, and writes final split lists and hashes.

Every valid source object is explicitly stored as class `0` (`FOD`) in YOLO labels. Boxes are clipped and validated, rejected objects and image-dimension mismatches are recorded, and images without valid objects retain empty label files. The processed directory includes:

- `fod_a.yaml`
- `dataset_manifest.json` with source/split provenance and a dataset fingerprint
- `dataset_statistics.json` with split, class, dimension, and box-area statistics
- `validation_report.json`
- paired `images/<split>` and `labels/<split>` directories

`FOD_DATA_ROOT` may relocate raw and processed data on the training machine while the committed `data/...` configuration remains unchanged. Use the training machine's active compatible Python interpreter; no Python 3.11 executable path is hardcoded.

For a local fixture-only smoke check that does not contact Kaggle:

```powershell
python -m pytest -q -m smoke
```

Implementation notes for maintainers and the suggested commit message are recorded in [`IMPLEMENTATION_SUMMARY.md`](IMPLEMENTATION_SUMMARY.md).
