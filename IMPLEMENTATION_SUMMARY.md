# Implementation Summary

This file records completed project parts as a maintainer handoff. Generated datasets and reports are intentionally excluded; this repository stores the reproducible code and configuration used to create them.

## Major Part 3: Dataset pipeline

### What this part implements

The dataset milestone covers the complete path from a Kaggle archive to a strictly validated, explicit single-class YOLO dataset:

1. `scripts/download_dataset.py` loads `configs/dataset.yaml`, validates Kaggle credentials, downloads with the official CLI, hashes the archive, writes source provenance, and safely installs the extracted directory.
2. `fod_yolo.dataset.discover` locates a Pascal VOC root even when the archive contains wrapper directories. Ambiguous layouts fail instead of selecting an arbitrary directory.
3. `fod_yolo.dataset.voc` parses XML into immutable typed records and rejects unsafe or malformed input.
4. `fod_yolo.dataset.split` preserves the official test set and creates a seeded 80/20 train/validation split from `trainval`. If official lists are unavailable, it records a deterministic 70/15/15 fallback warning.
5. `fod_yolo.dataset.convert` reads actual image dimensions, clips and validates boxes, remaps every valid original category to `0 FOD`, records rejected annotations, and retains negative images with empty labels.
6. `fod_yolo.dataset.statistics` builds split, source-category, image-size, box-area, and annotation-repair statistics.
7. `fod_yolo.dataset.pipeline` builds through staging, writes immutable split lists, a dataset YAML, statistics, and a provenance manifest/fingerprint, validates the staging output, then installs it atomically.
8. `scripts/validate_dataset.py` verifies YAML class metadata, image/label pairing, image readability, label syntax and coordinates, split separation, optional cross-split duplicate content, and manifest membership/counts.

### Maintainer model

The three CLI scripts are thin wrappers. Dataset rules live in `src/fod_yolo/dataset`, which keeps parsing, conversion, splitting, provenance, and validation independently testable. `prepare_dataset()` is the orchestrator: it will reuse an already valid processed dataset unless forced, and a failed rebuild cannot replace the last completed dataset.

The dataset fingerprint is derived from the source archive SHA-256, conversion configuration, and final split-list hashes. This lets later training and publication stages identify the exact data preparation inputs without committing the dataset itself.

Paths beginning with `data/` are remapped under `FOD_DATA_ROOT`. This preserves one committed configuration across devices while allowing each training system to choose its own storage location and compatible active Python interpreter.

### Validation completed

- Python: 3.14.3 from `.venv`
- Ruff formatting and lint: pass
- Mypy strict source check: pass
- Pytest: 59 tests and 6 subtests pass, including fixture-backed end-to-end coverage with no real Kaggle download

The tiny Pascal VOC fixture deliberately includes multiple original categories, a clipped box, an empty image, a source/XML dimension mismatch, and a degenerate box. It proves explicit class-0 labels, negative-image retention, diagnostic recording, deterministic splitting, and strict validation.

### Operational boundary

No production dataset was downloaded in this milestone because that requires the maintainer's Kaggle credentials and network access. The dataset code is complete, but no model metrics exist until an operator performs training and evaluation.

### Suggested descriptive commit message

```text
Implement reproducible VOC-to-YOLO dataset pipeline
```

## Major Part 4: Training orchestration and recovery

### What this part implements

This milestone implements specification Sections 17 and 18 as one connected training system:

1. `fod_yolo.training.config` validates the fixed Phase 1 baseline, resolves data/run paths for the current machine, and preserves only explicit configuration values for Ultralytics.
2. `fod_yolo.training.run_metadata` creates UTC/Git-based run IDs and stable initializing, running, success, and failed metadata states.
3. `fod_yolo.training.trainer` writes the resolved configuration before preflight, runs strict dataset and environment validation, invokes `YOLO(...).train(...)`, preserves the native Ultralytics run directory, verifies both checkpoints, and installs a stable candidate directory.
4. `fod_yolo.training.resume` verifies `weights/last.pt`, the original run identity/configuration, completion status, and dataset fingerprint before allowing an in-place resume.
5. `scripts/train.py` exposes new-run and resume workflows with the specification-defined exit codes for configuration, dataset, environment/CUDA, and training failures.

### Maintainer model

`run_training()` is the transaction boundary. A unique run directory and its immutable `resolved_config.yaml` exist before validation or model loading, so a later failure still leaves enough evidence to diagnose or resume the run. `run_metadata.json` is updated atomically at each state transition and includes Git state, dataset identity, timestamps, resume history, checkpoint hashes, and structured failure information.

The Ultralytics API is loaded lazily only after strict preflight. New training calls receive the complete resolved arguments, while resume loads `YOLO(last.pt)` and calls `train(resume=True)` so optimizer, scheduler, and epoch state come from the checkpoint. The original run ID is retained rather than creating an unrelated run.

Successful checkpoints have two roles: Ultralytics owns the complete `runs/train/<run-id>` directory and the project copies verified `best.pt`/`last.pt` files plus provenance into `artifacts/candidates/<run-id>`. Candidate creation uses staging and an atomic directory install, so incomplete candidate artifacts are not presented as successful.

CUDA remains mandatory by default. `--allow-cpu` is an explicit escape hatch for controlled debugging. An out-of-memory failure records `imgsz` and `batch` and recommends lowering the batch only; the fixed 1280 image size is never changed automatically.

### Validation completed

- Local interpreter: Python 3.14.3 from `.venv`
- No Python executable version is hardcoded into training or resume logic
- Ruff formatting and lint: pass
- Mypy strict source check: pass
- Pytest: 70 tests and 6 subtests pass, covering configuration invariants, portable run/artifact roots, successful fake training, CUDA preflight rejection, OOM metadata, checkpoint hashing, dataset-change rejection, in-place resume, and transient Windows rename recovery
- No model was downloaded and no production training was started

### Suggested descriptive commit message

```text
Add reproducible YOLO26n training and resume orchestration
```
