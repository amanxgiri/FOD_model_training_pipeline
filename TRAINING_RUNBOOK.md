# Ubuntu Training Runbook

This runbook covers dataset download, preparation, validation, and a 50-epoch
YOLO26n training run on Ubuntu 24.04 with the training device's Python 3.12.3.
The repository does not require Python 3.11 and does not copy the development
machine's virtual environment to the training device.

## 1. Check the training device

From a terminal on the Ubuntu device:

```bash
python3 --version
nvidia-smi
```

Python 3.12.3 is compatible with this project's `>=3.11` policy. `nvidia-smi`
must detect the NVIDIA GPU and driver before CUDA training can work.

## 2. Clone or update the repository

For a new checkout:

```bash
git clone <repository-url> FOD_model_training_pipeline
cd FOD_model_training_pipeline
```

For an existing checkout:

```bash
cd ~/Documents/FOD_model_training_pipeline
git pull
```

## 3. Create a device-local virtual environment

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip
python3 -m venv .venv
source .venv/bin/activate
python --version
python -m pip install --upgrade pip
```

Once `.venv` is active, use `python` for all commands below. It resolves to the
training device's virtual-environment interpreter rather than a Python version
from another machine.

## 4. Install PyTorch for this GPU

Use the official PyTorch installation selector to choose the CUDA wheel profile
supported by the current PyTorch release and this machine. Do not infer the
profile only from the CUDA version printed by `nvidia-smi`.

Preview the selected installation first. For example, if the official selector
provides CUDA 12.8 for this device:

```bash
python scripts/install_torch.py --profile cu128 --dry-run
python scripts/install_torch.py --profile cu128
```

Replace `cu128` with the profile selected for the training device. Then install
the remaining project packages into the same environment:

```bash
python -m pip install -r requirements/base.txt
python -m pip install --editable .
```

## 5. Create Kaggle credentials on the training device

The repository's `.env` is ignored by Git to protect secrets. A `.env` created
on the development machine therefore will not appear on this Ubuntu device.
Create `~/Documents/FOD_model_training_pipeline/.env` locally:

```dotenv
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
```

Do not add spaces around `=`, do not include `<` or `>` placeholders, and do
not commit this file. The project CLI commands automatically load this file.
Existing shell environment variables take precedence over `.env` values.

Only `KAGGLE_USERNAME` and `KAGGLE_KEY` are required for the configured Kaggle
download. `KAGGLE_CONFIG_DIR` is optional and is needed only when using a
`kaggle.json` file instead. `GH_TOKEN` and `GITHUB_REPOSITORY` are unrelated to
dataset download.

Check that credentials are visible without printing their values:

```bash
python -c 'import os; from pathlib import Path; from fod_yolo.bootstrap import load_project_environment; load_project_environment(Path.cwd()); print("username:", bool(os.getenv("KAGGLE_USERNAME")), "key:", bool(os.getenv("KAGGLE_KEY")))'
```

Both results must be `True`.

## 6. Download, prepare, and validate the dataset

Run the three stages in order:

```bash
python scripts/download_dataset.py --config configs/dataset.yaml
python scripts/prepare_dataset.py --config configs/dataset.yaml
python scripts/validate_dataset.py \
  --data data/processed/fod_a_single_class_yolo/fod_a.yaml \
  --strict
```

The published FOD-A split files contain a small number of repeated and
overlapping IDs. Preparation retains each ID once and gives the supplied
official `test` list precedence, removing those IDs from `trainval` before the
seeded train/validation division. The affected IDs and both repairs are stored
under `split_warnings` in `dataset_manifest.json`, so no final split leaks test
images into training or validation.

If an earlier failed validation created only a partial processed directory,
rerun the preparation command normally. The pipeline detects incomplete output
and replaces it atomically; manual deletion and `--force` are not required.

Preparation indexes the `JPEGImages` directory once using case-insensitive IDs
and extensions, which avoids repeated full-directory scans on Linux when the
archive uses uppercase image extensions. It logs the resolved split counts and
conversion progress every 1,000 images. Runtime is primarily determined by CPU
and storage speed rather than the GPU.

By default, raw and processed data are under `data/`. To place generated files
on another disk, add paths such as these to `.env` before downloading:

```dotenv
FOD_DATA_ROOT=/mnt/training-data/fod
FOD_RUNS_ROOT=/mnt/training-runs/fod
FOD_ARTIFACTS_ROOT=/mnt/training-artifacts/fod
```

The same commands continue to work because the project resolves these roots at
runtime.

## 7. Verify CUDA and the training environment

```bash
python scripts/check_environment.py --require-cuda
```

This must succeed before production training. It writes the reproducibility
report to `reports/environment_report.json`.

## 8. Train for 50 epochs

The committed configuration defaults to 100 epochs. Override it for this run
without editing the shared YAML:

```bash
python scripts/train.py \
  --config configs/train_yolo26n_1280.yaml \
  --set training.epochs=50
```

Training outputs are written under `runs/train/<run-id>/`, and verified
candidate checkpoints are copied under `artifacts/candidates/<run-id>/`.

Monitor the run in a second terminal after activating the same environment:

```bash
source .venv/bin/activate
tensorboard --logdir runs/train --port 6006
```

To resume an interrupted run, use its latest checkpoint:

```bash
python scripts/train.py --resume runs/train/<run-id>/weights/last.pt
```

The resume operation retains the original run configuration, including the
50-epoch target recorded for that run.
