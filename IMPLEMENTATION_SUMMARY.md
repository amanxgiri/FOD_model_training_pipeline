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

## Major Part 5: Quantitative evaluation and threshold analysis

### What this part implements

1. `fod_yolo.evaluation.matcher` provides independently tested, confidence-ordered, one-to-one IoU matching with explicit single-class validation.
2. `fod_yolo.evaluation.threshold_sweep` calculates TP/FP/FN, precision, recall, F1, false-negative rate, false positives per image, images with false negatives, and normalized-area small-object metrics at every configured confidence threshold.
3. Reference selection deterministically reports best F1, the highest threshold retaining maximum recall, and the lowest-false-positive threshold within one percentage point of maximum recall.
4. `fod_yolo.evaluation.ultralytics_eval` runs native Ultralytics validation, extracts mAP/precision/recall, performs minimum-confidence predictions, normalizes `Results.boxes.xyxyn`, and captures stage latency plus peak GPU memory when available.
5. `fod_yolo.evaluation.runner` validates the dataset, binds model and dataset hashes, writes stable JSON/CSV/config/environment artifacts through staging, and preserves all framework output.
6. `scripts/evaluate.py` supports validation analysis and locked final-test evaluation with the required non-zero failure codes.

### Maintainer model

Ultralytics remains the source of standard mAP values and its native plots. Project code owns threshold-specific safety metrics so the matching rules are explicit and unit tested. Predictions are collected once at the minimum sweep confidence, then filtered and rematched at every threshold without rerunning inference.

Validation and test have different authority. Validation may sweep thresholds and generate analytical references. Test requires `--locked-threshold`, evaluates only that value, and sets `threshold_selection_allowed=false` in `metrics.json`; this prevents test results from leaking into model or operating-threshold selection.

The evaluation directory is installed atomically only after framework evaluation, matching, metric serialization, and provenance capture all succeed. Existing evaluation directories are never silently overwritten.

### Operational boundary

This part implements quantitative evaluation, threshold tables, and runtime metrics. Qualitative annotated examples, static project plots, and the self-contained HTML report remain for the reporting milestone. No real model metrics were generated locally.

### Validation completed

- Python 3.14.3 local interpreter
- Ruff and strict Mypy checks pass
- 78 tests and 6 subtests pass
- Hand-calculated IoU/matching, threshold references, small-object metrics, fake Ultralytics validation, latency, atomic output, and locked-test behavior are covered
- Evaluation CLI help passes without loading or downloading a model

### Suggested descriptive commit message

```text
Implement safety-focused model evaluation and threshold analysis
```

## Operational readiness update: Local environment loading and Ubuntu runbook

### What this update implements

1. `fod_yolo.bootstrap.load_project_environment` reads an optional repository-root `.env` before CLI configuration, logging, path resolution, or Kaggle authentication.
2. Existing process environment values take precedence, so explicit device or job-scheduler configuration is never silently overwritten.
3. Parser errors identify the affected line without echoing credential values, and loaded values are never logged or returned.
4. Dataset download/preparation/validation, training, evaluation, and environment diagnostics share the same bootstrap behavior.
5. `TRAINING_RUNBOOK.md` documents the Ubuntu 24.04 and Python 3.12.3 workflow, device-specific PyTorch installation, local Kaggle credential creation, dataset stages, CUDA validation, 50-epoch training, monitoring, and resume.
6. Official source split lists are duplicate-aware: repeated IDs are retained once, and overlap is resolved by preserving official test membership and removing those IDs from `trainval` before seeded shuffling. Both repairs are recorded in `split_warnings`.
7. A partial processed directory created by an earlier failed validation is recognized as incomplete and rebuilt atomically without manual deletion or `--force`.
8. Source images are indexed once by case-insensitive ID and extension, eliminating a quadratic Linux fallback scan caused by uppercase archive extensions. Conversion logs progress every 1,000 images.

### Maintainer model

`.env` is a local convenience layer, not a reproducibility artifact. It remains ignored by Git and must be created independently on each device. The project records non-secret resolved configuration and environment metadata in generated reports, while credentials remain outside committed files and generated logs.

The loader is dependency-free so credentials and portable path overrides are available before optional runtime packages are imported. A missing `.env` is valid; malformed content fails immediately with exit code 2 instead of allowing a command to run with partial or surprising configuration.

### Validation completed

- Ruff formatting and lint: pass
- Mypy strict source check: pass across 32 source files
- Pytest: 85 tests and 6 subtests pass
- Credential parser tests use temporary files and do not access or print the maintainer's real `.env`

### Suggested descriptive commit message

```text
Resolve FOD-A split leakage and Linux preparation slowdown
```

## Major Part 6: Streaming annotated video inference

### What this part implements

1. `fod_yolo.inference.detector.FODDetector` lazily loads an Ultralytics checkpoint and normalizes each frame result into stable original-frame pixel coordinates, class labels, confidence, and stage timings.
2. `fod_yolo.inference.video.run_video_inference` validates source metadata, iterates without retaining the full video, applies optional time bounds and frame stride, draws boxes/labels, and writes an annotated MP4.
3. Every prediction is written to the specification-defined `detections.csv`; `frame_metrics.csv` records detection counts and per-frame preprocessing, inference, postprocessing, and end-to-end time.
4. `video_summary.json` reports source/model identity, frame/detection counts, confidence statistics, mean/P95 inference latency, processing FPS, real-time ratio, and completion status.
5. Interruptions retain flushed CSV data and a partial summary marked `incomplete`. Unlabelled videos are explicitly marked `inference-only`, with accuracy metrics unavailable.
6. `scripts/infer_video.py` exposes model/video paths, image size, confidence, device, stride, time bounds, output controls, detection-frame export, and config overrides while printing final statistics to the terminal.

### Maintainer model

The CLI owns user-facing path resolution and exit codes. Typed configuration defines the reproducible contract, `FODDetector` isolates Ultralytics-specific result shapes, and the video runner owns OpenCV streaming and stable artifacts. This separation lets the runner be tested with fake frames and predictions without loading a model or video codec.

The annotated output contains only processed frames and uses `source_fps / frame_stride`, preserving playback timing when frames are intentionally skipped. Bounding-box coordinates in CSV always refer to the original source frame dimensions; model resizing remains internal to Ultralytics.

The summary cannot claim detection accuracy because the uploaded video is unlabelled. Precision, recall, mAP, and false-negative rate require ground-truth video annotations and remain outside this inference-only workflow.

### Validation completed

- Ruff formatting and lint: pass
- Mypy strict source check: pass across 35 source files
- Pytest: 88 tests and 6 subtests pass
- Video output, frame stride, detection schema, confidence/latency statistics, annotation drawing, model-result normalization, and interrupted-run recovery have fixture-backed coverage
- No real checkpoint or video was required for automated tests

### Suggested descriptive commit message

```text
Implement streaming annotated video inference and statistics
```
