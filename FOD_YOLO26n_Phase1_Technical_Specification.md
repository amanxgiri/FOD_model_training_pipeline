---
title: "FOD Detection: YOLO26n Phase 1 Training Pipeline"
subtitle: "Implementation-Grade Technical Specification for Codex"
author: "FOD Detection Project"
date: "14 July 2026"
version: "1.0"
status: "Approved for implementation"
---

# Document Control

| Field | Value |
|---|---|
| Document | FOD Detection: YOLO26n Phase 1 Training Pipeline |
| Version | 1.0 |
| Date | 14 July 2026 |
| Status | Approved for implementation |
| Primary implementation language | Python 3.11 |
| Target model | Ultralytics YOLO26n object detector (`yolo26n.pt`) |
| Training input size | 1280 pixels |
| Phase | Phase 1: broad single-class FOD pretraining/fine-tuning |
| Intended implementation agent | Codex |

# 1. Purpose

This document defines the complete implementation contract for a reproducible Python repository that downloads the FOD-A dataset from Kaggle, converts its Pascal VOC annotations into a single-class Ultralytics YOLO dataset, trains a YOLO26n detector at an input size of 1280 pixels, evaluates the model, promotes the current validation-best project champion, publishes that champion as a downloadable GitHub Release asset, and supports later verification on video files.

The specification is intentionally implementation-grade. Codex should be able to create the repository without needing to infer directory names, script responsibilities, configuration contracts, output schemas, validation behavior, or the execution order.

# 2. Project Requirements and Fixed Decisions

The following are fixed project requirements and must not be changed unless the project owner explicitly changes them:

1. The model family is Ultralytics YOLO26, nano detection variant.
2. The pretrained checkpoint identifier is `yolo26n.pt`.
3. The task is object detection, not classification or segmentation.
4. The Phase 1 dataset is the Kaggle FOD-A dataset identified by the slug:

   `kilogrand/foreign-object-debris-in-airports-fod-a-dataset`

5. The dataset must be downloaded at runtime through the official Kaggle CLI/API. Dataset files must not be committed to GitHub.
6. All original FOD categories in the supplied dataset must be remapped to one broad class:

   `0: FOD`

7. Every valid original bounding box must be retained. Only the class identity changes.
8. Training and evaluation image size must default to `1280`.
9. NVIDIA CUDA GPU training must be supported and verified before a full training run starts.
10. The pipeline must preserve both the latest resumable checkpoint and the best checkpoint from each training run.
11. A project-level champion model must be selected using validation results, not test results.
12. The current champion must be publishable to GitHub as a downloadable model artifact.
13. The repository must support future inference on a runway video file and save an annotated video plus machine-readable detections.
14. Evaluation results must be available as JSON/CSV, static plots, a generated HTML report, and TensorBoard-compatible logs.

# 3. Scope

## 3.1 In Scope

- Python environment setup.
- NVIDIA GPU and CUDA verification.
- Installation of PyTorch, torchvision, Ultralytics, Kaggle CLI, and supporting libraries.
- Secure Kaggle authentication.
- Dataset download, extraction, discovery, integrity checks, and manifest creation.
- Pascal VOC XML parsing.
- Deterministic train/validation/test split construction.
- Mapping all original categories to one `FOD` class.
- Pascal VOC to YOLO bounding-box conversion.
- Dataset validation and statistics.
- YOLO26n transfer learning at `imgsz=1280`.
- Resume support and periodic checkpoints.
- Validation and test evaluation.
- Confidence-threshold analysis.
- Small-object recall analysis.
- False-positive and false-negative visualization.
- Model champion promotion.
- GitHub Release publishing and model download.
- Video inference and video result export.
- Unit tests, smoke tests, linting, and CPU-only continuous integration.

## 3.2 Out of Scope for Phase 1

- Training on the project’s actual runway dataset.
- Multi-camera image stitching or synchronization.
- A runway-feature expert model.
- Anomaly detection for runway damage, water, oil, tire marks, or other runway features.
- Ensemble inference with RT-DETR or another detector.
- Production alerting logic.
- Airfield deployment, camera calibration, geolocation, or FOD distance estimation.
- A final operational confidence threshold for a live runway. Phase 1 will produce threshold trade-off data; the operational threshold must be finalized using the actual runway validation set.
- Model export to TensorRT, ONNX, or an edge device. The design must leave room for this, but implementation is not required in Phase 1.

# 4. Source and Framework Basis

The supplied Kaggle dataset is described as containing common foreign object debris with runway or taxiway backgrounds [R1]. Ultralytics documents `yolo26n.pt` as a YOLO26 detection checkpoint and provides Python and CLI workflows for training, validation, prediction, export, tracking, and benchmarking [R2][R3]. Ultralytics validation exposes mAP metrics and supports saved plots and JSON outputs [R4]. Predict mode accepts images, videos, streams, and generator-style processing for long inputs [R5].

The implementation must use the official PyTorch installation selector for the CUDA build appropriate to the desktop’s NVIDIA driver and platform rather than assuming that one CUDA wheel is correct for every machine [R6]. The official Kaggle CLI must be used to authenticate and download datasets [R7]. GitHub Releases are the preferred distribution mechanism for the binary model and associated reports [R8]. Git LFS may be supported as an optional alternative, but it is not the default [R9].

# 5. High-Level Architecture

```text
Kaggle FOD-A Dataset
        |
        v
Dataset Download + Archive Manifest
        |
        v
Pascal VOC Discovery and Validation
        |
        v
Deterministic Split Resolver
        |
        v
Single-Class Remapping (all classes -> FOD)
        |
        v
YOLO Dataset Conversion + Dataset Manifest
        |
        v
YOLO26n Training at 1280 px
        |
        +----------------------+
        |                      |
        v                      v
Per-Run best.pt            Per-Run last.pt
        |
        v
Validation Evaluation + Threshold Analysis
        |
        v
Project Champion Promotion
        |
        v
Locked Test Evaluation
        |
        v
Model Card + Metrics + SHA-256
        |
        v
GitHub Release Publication
        |
        +----------------------+
        |                      |
        v                      v
Image Inference          Video Inference
```

# 6. Repository Layout

Codex must create the following repository structure. Empty generated-data directories may be represented by `.gitkeep` files only where needed.

```text
fod-yolo26n/
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements/
│   ├── base.txt
│   ├── dev.txt
│   └── README.md
├── configs/
│   ├── dataset.yaml
│   ├── train_yolo26n_1280.yaml
│   ├── evaluate.yaml
│   ├── promotion.yaml
│   └── video_inference.yaml
├── scripts/
│   ├── install_torch.py
│   ├── check_environment.py
│   ├── download_dataset.py
│   ├── prepare_dataset.py
│   ├── validate_dataset.py
│   ├── train.py
│   ├── evaluate.py
│   ├── build_report.py
│   ├── promote_model.py
│   ├── publish_model.py
│   ├── download_model.py
│   ├── infer_image.py
│   └── infer_video.py
├── src/
│   └── fod_yolo/
│       ├── __init__.py
│       ├── config.py
│       ├── logging_utils.py
│       ├── hashing.py
│       ├── environment.py
│       ├── paths.py
│       ├── dataset/
│       │   ├── __init__.py
│       │   ├── kaggle_client.py
│       │   ├── discover.py
│       │   ├── voc.py
│       │   ├── split.py
│       │   ├── convert.py
│       │   ├── validate.py
│       │   └── statistics.py
│       ├── training/
│       │   ├── __init__.py
│       │   ├── trainer.py
│       │   ├── run_metadata.py
│       │   └── resume.py
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── ultralytics_eval.py
│       │   ├── matcher.py
│       │   ├── threshold_sweep.py
│       │   ├── small_objects.py
│       │   ├── latency.py
│       │   └── qualitative.py
│       ├── registry/
│       │   ├── __init__.py
│       │   ├── champion.py
│       │   ├── model_card.py
│       │   ├── github_release.py
│       │   └── downloader.py
│       ├── inference/
│       │   ├── __init__.py
│       │   ├── detector.py
│       │   ├── image.py
│       │   └── video.py
│       └── reporting/
│           ├── __init__.py
│           ├── plots.py
│           ├── html_report.py
│           └── templates/
│               └── evaluation_report.html.j2
├── tests/
│   ├── fixtures/
│   │   └── tiny_voc/
│   ├── test_voc_parser.py
│   ├── test_voc_to_yolo.py
│   ├── test_single_class_mapping.py
│   ├── test_split_reproducibility.py
│   ├── test_dataset_validation.py
│   ├── test_detection_matching.py
│   ├── test_threshold_metrics.py
│   ├── test_champion_promotion.py
│   └── test_video_schema.py
├── models/
│   ├── registry.json
│   └── README.md
├── data/
│   └── .gitkeep
├── runs/
│   └── .gitkeep
├── reports/
│   └── .gitkeep
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── publish-model.yml
├── .env.example
├── .gitignore
└── Makefile
```

# 7. Generated Local Directory Layout

The following directories are generated locally and must not be committed:

```text
data/
├── raw/
│   └── fod_a/
│       ├── downloads/
│       ├── extracted/
│       └── source_manifest.json
├── processed/
│   └── fod_a_single_class_yolo/
│       ├── images/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       ├── labels/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       ├── fod_a.yaml
│       ├── dataset_manifest.json
│       ├── dataset_statistics.json
│       └── validation_report.json
└── cache/

runs/
├── train/
├── val/
├── test/
└── inference/

artifacts/
├── candidates/
└── champion/
    ├── fod_yolo26n_best.pt
    ├── model_metadata.json
    ├── model_card.md
    ├── evaluation_metrics.json
    ├── evaluation_metrics.csv
    ├── training_config.yaml
    ├── dataset_manifest.json
    └── SHA256SUMS

reports/
└── <run_id>/
    ├── index.html
    ├── summary.md
    ├── metrics.json
    ├── metrics.csv
    ├── plots/
    ├── false_positives/
    ├── false_negatives/
    └── ground_truth_vs_prediction/
```

# 8. Environment and Dependencies

## 8.1 Supported Runtime

- Python: `3.11.x`.
- Primary training platforms: Ubuntu Linux or Windows with an NVIDIA CUDA-capable GPU.
- Development may be performed on a different machine, but full training must run only after GPU validation succeeds.
- CPU execution must remain available for unit tests, dataset conversion, report generation, and lightweight smoke tests.

## 8.2 System Prerequisites

The setup documentation must require:

- Git.
- Python 3.11.
- A current NVIDIA driver compatible with the selected PyTorch CUDA wheel.
- `nvidia-smi` available for GPU diagnostics.
- GitHub CLI (`gh`) for publishing, or a Python GitHub Releases API fallback.
- FFmpeg available on `PATH` for robust video probing and optional transcoding.
- Git LFS only when the optional LFS workflow is selected.

## 8.3 Dependency Strategy

Do not place `torch` and `torchvision` in the normal platform-neutral dependency file. Their installation differs by operating system and CUDA build. Instead:

1. `scripts/install_torch.py` accepts an explicit PyTorch index URL or a named compute profile.
2. The script installs `torch` and `torchvision` from the official PyTorch wheel index.
3. The script prints and stores the installed versions.
4. `requirements/base.txt` installs the remaining packages.
5. A successful environment may be frozen to `requirements-lock.txt` for exact reproduction on the same platform.

## 8.4 Initial Dependency Set

`requirements/base.txt` must include compatible pinned or bounded versions of:

```text
ultralytics==8.4.68
kaggle
opencv-python
Pillow
PyYAML
numpy
pandas
matplotlib
Jinja2
tensorboard
tqdm
requests
psutil
```

`requirements/dev.txt` must include:

```text
pytest
pytest-cov
ruff
mypy
pre-commit
types-PyYAML
types-requests
```

The project may update the Ultralytics version later, but the exact version used for every run must be recorded in run metadata. The initial pinned version above matches the project’s currently tested Ultralytics environment.

## 8.5 Environment Variables

`.env.example` must document, but never contain, secrets:

```text
# Kaggle authentication: either these two variables or a local kaggle.json
KAGGLE_USERNAME=
KAGGLE_KEY=
KAGGLE_CONFIG_DIR=

# GitHub publication
GH_TOKEN=
GITHUB_REPOSITORY=owner/repository

# Optional path overrides
FOD_DATA_ROOT=./data
FOD_RUNS_ROOT=./runs
FOD_ARTIFACTS_ROOT=./artifacts

# Optional GPU selection
CUDA_VISIBLE_DEVICES=0
```

The code must never print `KAGGLE_KEY` or `GH_TOKEN`.

# 9. Configuration Files

All principal behavior must be controlled through YAML configuration files. Command-line arguments may override YAML values.

## 9.1 `configs/dataset.yaml`

```yaml
dataset:
  kaggle_slug: kilogrand/foreign-object-debris-in-airports-fod-a-dataset
  kaggle_version: null
  raw_root: data/raw/fod_a
  processed_root: data/processed/fod_a_single_class_yolo
  archive_name: fod_a.zip
  force_download: false
  force_extract: false

conversion:
  target_class_id: 0
  target_class_name: FOD
  image_transfer_mode: copy  # copy | hardlink | symlink
  keep_empty_images: true
  clip_boxes: true
  reject_degenerate_boxes: true
  verify_image_dimensions: true

split:
  seed: 42
  validation_fraction_of_trainval: 0.20
  preserve_official_test: true
  source_trainval_file: ImageSets/Main/trainval.txt
  source_test_file: ImageSets/Main/test.txt
```

## 9.2 `configs/train_yolo26n_1280.yaml`

```yaml
model: yolo26n.pt
data: data/processed/fod_a_single_class_yolo/fod_a.yaml

training:
  epochs: 100
  imgsz: 1280
  batch: -1
  device: 0
  workers: 8
  pretrained: true
  optimizer: auto
  patience: 30
  seed: 42
  deterministic: true
  amp: true
  cache: false
  rect: false
  multi_scale: 0.0
  close_mosaic: 10
  save: true
  save_period: 10
  val: true
  plots: true
  project: runs/train
  name: yolo26n_fod_phase1_1280
  exist_ok: false

metadata:
  experiment_description: "Phase 1 single-class FOD training on Kaggle FOD-A"
  tags:
    - phase1
    - fod-a
    - single-class
    - yolo26n
    - imgsz1280
```

The baseline uses `batch=-1`, which asks Ultralytics to select a batch size using approximately 60 percent of available GPU memory [R3]. If auto-batch fails or is unstable at 1280 pixels, the user must be able to override it with an integer such as `8`, `4`, `2`, or `1` without changing code.

## 9.3 `configs/evaluate.yaml`

```yaml
evaluation:
  imgsz: 1280
  device: 0
  batch: 8
  workers: 8
  split_for_selection: val
  final_split: test
  plots: true
  save_json: true
  save_txt: true
  save_conf: true
  max_det: 300

matching:
  iou_threshold: 0.50
  class_agnostic: true

threshold_sweep:
  start: 0.05
  stop: 0.95
  step: 0.05
  include_default_confidence: 0.25

small_object:
  definition: normalized_area_ratio
  max_area_ratio: 0.01

qualitative:
  max_false_positive_examples: 50
  max_false_negative_examples: 50
  max_side_by_side_examples: 50
```

## 9.4 `configs/promotion.yaml`

```yaml
promotion:
  source_split: val
  metric_tolerance: 0.002
  ranking:
    - metric: small_object_recall
      direction: maximize
    - metric: recall
      direction: maximize
    - metric: map50_95
      direction: maximize
    - metric: false_positives_per_image
      direction: minimize
    - metric: p95_inference_latency_ms
      direction: minimize
  require_complete_metrics: true
  require_dataset_fingerprint_match: true
  require_clean_git_commit: false
```

## 9.5 `configs/video_inference.yaml`

```yaml
video:
  model: artifacts/champion/fod_yolo26n_best.pt
  source: null
  imgsz: 1280
  confidence: 0.25
  device: 0
  frame_stride: 1
  max_detections: 300
  save_annotated_video: true
  save_detection_csv: true
  save_summary_json: true
  save_detection_frames: false
  stream: true
  output_root: runs/inference/video
```

The `0.25` confidence value is only a smoke-test default. It must not be described as the final runway operating threshold. The evaluated threshold report must be used when choosing a threshold for a specific test or deployment context.

# 10. Environment Verification

## 10.1 `scripts/check_environment.py`

The script must:

1. Print Python version and executable path.
2. Import and print versions for torch, torchvision, Ultralytics, OpenCV, and CUDA runtime information.
3. Execute or parse `nvidia-smi` when available.
4. Report:
   - `torch.cuda.is_available()`
   - visible GPU count
   - each GPU name
   - total VRAM
   - current device
   - CUDA version reported by PyTorch
   - cuDNN version
5. Perform a small CUDA tensor operation when CUDA is expected.
6. Instantiate `YOLO("yolo26n.pt")` to confirm that the checkpoint can be resolved.
7. Write `reports/environment_report.json`.
8. Exit non-zero when `--require-cuda` is set and CUDA is unavailable.

Example:

```bash
python scripts/check_environment.py --require-cuda
```

# 11. Dataset Download Pipeline

## 11.1 Authentication

The downloader must support both official Kaggle authentication methods:

- `KAGGLE_USERNAME` plus `KAGGLE_KEY` environment variables.
- A local `kaggle.json` file in the configured Kaggle directory.

If credentials are unavailable, the script must terminate with a clear message and must not attempt anonymous scraping.

## 11.2 `scripts/download_dataset.py`

Required command:

```bash
python scripts/download_dataset.py --config configs/dataset.yaml
```

Required behavior:

1. Resolve the Kaggle dataset slug from configuration.
2. Authenticate with the Kaggle CLI/API.
3. Create the raw download directory.
4. Download the dataset archive.
5. Support an optional specific Kaggle dataset version.
6. Calculate SHA-256 for the downloaded archive.
7. Store a source manifest containing:
   - Kaggle slug
   - requested version
   - resolved version when available
   - download timestamp in UTC
   - archive filename
   - archive byte size
   - SHA-256
   - Kaggle CLI version
8. Extract into a temporary directory.
9. Atomically rename the completed extraction directory.
10. Be idempotent: skip a valid existing download unless `--force` is supplied.
11. Never commit or stage the archive.

Example source manifest:

```json
{
  "dataset_slug": "kilogrand/foreign-object-debris-in-airports-fod-a-dataset",
  "requested_version": null,
  "resolved_version": null,
  "downloaded_at_utc": "2026-07-14T00:00:00Z",
  "archive_path": "data/raw/fod_a/downloads/fod_a.zip",
  "archive_size_bytes": 0,
  "archive_sha256": "<sha256>",
  "kaggle_cli_version": "<version>"
}
```

# 12. Dataset Discovery and Pascal VOC Parsing

The extracted archive may contain one or more wrapper directories. The code must not assume one fixed absolute extraction path. It must recursively locate a Pascal VOC root containing, at minimum:

```text
Annotations/
JPEGImages/
ImageSets/Main/
```

If multiple candidates are found, the script must choose only when one candidate unambiguously contains the configured split files. Otherwise it must terminate and print all candidates.

## 12.1 VOC Parser Contract

`src/fod_yolo/dataset/voc.py` must expose typed data models equivalent to:

```python
@dataclass(frozen=True)
class BoundingBox:
    xmin: float
    ymin: float
    xmax: float
    ymax: float

@dataclass(frozen=True)
class VocObject:
    original_class_name: str
    bbox: BoundingBox
    difficult: bool
    truncated: bool

@dataclass(frozen=True)
class VocAnnotation:
    image_id: str
    filename: str
    width: int
    height: int
    depth: int | None
    objects: list[VocObject]
    xml_path: Path
```

The parser must reject malformed numeric values, missing required elements, non-positive image dimensions, and unreadable XML.

# 13. Split Strategy

The pipeline must preserve the supplied official test split when it exists. The official test split must not be used for hyperparameter selection, confidence-threshold selection, early stopping, or champion promotion.

The configured `trainval` list must be divided deterministically into:

- Training: 80 percent of `trainval`.
- Validation: 20 percent of `trainval`.
- Test: the supplied official test list.

The split algorithm must:

1. Normalize IDs and remove blank lines.
2. Detect duplicate IDs.
3. Verify every ID has an image and annotation.
4. Use `seed=42`.
5. Sort the source IDs before seeded shuffling so results do not depend on filesystem order.
6. Save final ID files and their SHA-256 hashes.
7. Guarantee that train, validation, and test sets are disjoint.
8. Store the exact split membership in the dataset manifest.

If the dataset version does not provide the configured split files, the script may create a deterministic 70/15/15 image-level split, but it must emit a warning and record that fallback in the manifest.

# 14. Single-Class Conversion

## 14.1 Class Mapping

Every valid VOC object, regardless of its original category, must be written with class ID `0`.

The dataset YAML must contain exactly:

```yaml
path: <absolute-or-resolved-processed-root>
train: images/train
val: images/val
test: images/test
names:
  0: FOD
```

Do not rely only on Ultralytics `single_cls=True`. The stored labels themselves must already contain only class ID `0`. This makes the converted dataset explicit, inspectable, portable, and independent of a training flag.

## 14.2 Bounding-Box Conversion

For an image of width `W` and height `H`, a valid VOC box `(xmin, ymin, xmax, ymax)` must be converted to normalized YOLO format:

```text
x_center = ((xmin + xmax) / 2) / W
y_center = ((ymin + ymax) / 2) / H
box_width = (xmax - xmin) / W
box_height = (ymax - ymin) / H
```

Each output line must be:

```text
0 x_center y_center box_width box_height
```

Values must be written with sufficient precision, for example six decimal places.

## 14.3 Box Validation Rules

Before conversion:

1. Optionally clip coordinates to image boundaries.
2. Reject a box when `xmax <= xmin` or `ymax <= ymin` after clipping.
3. Reject NaN or infinite coordinates.
4. Reject boxes that normalize outside `[0, 1]` after clipping.
5. Record every rejected object with its image ID, original class, XML path, and reason.
6. Do not silently discard invalid objects.

## 14.4 Image Dimension Verification

When `verify_image_dimensions=true`, Pillow must read the actual image dimensions. If they differ from the XML dimensions:

- Use the actual image dimensions for conversion.
- Record the discrepancy.
- Fail only when the discrepancy makes the annotation impossible to validate safely.

## 14.5 Empty Images

When an image contains no valid retained object, create an empty `.txt` label file when `keep_empty_images=true`. This preserves negative examples and maintains one image-to-label mapping.

# 15. Dataset Validation

## 15.1 `scripts/validate_dataset.py`

Required command:

```bash
python scripts/validate_dataset.py \
  --data data/processed/fod_a_single_class_yolo/fod_a.yaml \
  --strict
```

Required checks:

- Dataset YAML exists and parses.
- Exactly one class exists and is named `FOD`.
- Every label class ID is `0`.
- Each image has a corresponding label file.
- Each label file has a corresponding image.
- All normalized coordinates are finite and within valid limits.
- Width and height are greater than zero.
- No split overlap exists.
- No duplicate image content exists across splits when duplicate hashing is enabled.
- Images are readable.
- Labels are parseable.
- No missing IDs exist relative to the manifest.
- Split counts match the manifest.

Required output:

```json
{
  "status": "pass",
  "errors": [],
  "warnings": [],
  "counts": {
    "train_images": 0,
    "val_images": 0,
    "test_images": 0,
    "train_objects": 0,
    "val_objects": 0,
    "test_objects": 0
  },
  "class_ids_observed": [0],
  "class_names": {"0": "FOD"}
}
```

Training must refuse to start when strict dataset validation fails.

# 16. Dataset Statistics and Manifest

`dataset_statistics.json` must include:

- Image count by split.
- Object count by split.
- Empty-image count by split.
- Original category names and original object counts before remapping.
- Final class distribution.
- Image width and height distributions.
- Bounding-box normalized area distribution.
- Counts of small, medium, and large boxes using configurable project definitions.
- Minimum, median, mean, P95, and maximum box area ratio.
- Invalid or repaired annotation counts.

`dataset_manifest.json` must include:

- Source manifest.
- VOC root path relative to the project data root.
- Conversion code version or Git commit.
- Class mapping.
- Split configuration.
- Exact split IDs or paths to immutable split files.
- File counts.
- Hashes of split lists.
- Hash of the final dataset YAML.
- A dataset fingerprint computed from the source archive hash, conversion configuration, and split-list hashes.

# 17. Training Pipeline

## 17.1 `scripts/train.py`

Required command:

```bash
python scripts/train.py --config configs/train_yolo26n_1280.yaml
```

Optional overrides:

```bash
python scripts/train.py \
  --config configs/train_yolo26n_1280.yaml \
  --set training.batch=4 \
  --set training.device=0 \
  --set training.epochs=150 \
  --set training.name=yolo26n_fod_phase1_1280_run02
```

Required behavior:

1. Load and validate configuration.
2. Run strict dataset validation.
3. Run environment validation.
4. Require CUDA unless `--allow-cpu` is explicitly passed.
5. Load `YOLO("yolo26n.pt")`.
6. Call `model.train(...)` with the resolved configuration.
7. Save the fully resolved configuration before training starts.
8. Save environment metadata.
9. Save Git commit and dirty-state information.
10. Save the dataset fingerprint.
11. Use a unique run ID and directory.
12. Preserve all Ultralytics-generated logs and plots.
13. Copy or link `weights/best.pt` and `weights/last.pt` into a stable candidate artifact directory after successful training.
14. Calculate SHA-256 hashes for checkpoints.
15. Generate `run_metadata.json` even when training fails, with status and error information.

## 17.2 Run ID

Use a stable pattern:

```text
yolo26n_fod_phase1_1280_<UTC timestamp>_<short git sha>
```

Example:

```text
yolo26n_fod_phase1_1280_20260714T021500Z_a1b2c3d
```

## 17.3 Initial Training Parameters

The initial baseline must use:

| Parameter | Value |
|---|---:|
| Model | `yolo26n.pt` |
| Task | Detect |
| Classes | 1 (`FOD`) |
| Image size | 1280 |
| Epochs | 100, configurable |
| Batch | `-1` auto, configurable |
| Device | GPU 0 |
| Pretrained | true |
| Optimizer | auto |
| AMP | true |
| Deterministic | true |
| Seed | 42 |
| Early-stopping patience | 30 |
| Checkpoint period | every 10 epochs |
| Mosaic close | final 10 epochs |
| Validation during training | true |
| Plots | true |

The initial run should retain Ultralytics defaults for unlisted augmentation and optimizer parameters. Hyperparameters must not be changed merely because they appear tunable; every later change must be made through a versioned configuration file and a separate run.

# 18. Resume and Failure Recovery

The pipeline must support:

```bash
python scripts/train.py --resume runs/train/<run-name>/weights/last.pt
```

Resume behavior must:

- Verify that the checkpoint exists.
- Confirm that the dataset fingerprint matches the original run unless `--allow-dataset-change` is explicitly supplied.
- Reuse the original run configuration where required by Ultralytics resume behavior.
- Record the resume source and resume timestamp.
- Never overwrite a different run silently.

If an out-of-memory error occurs:

1. Preserve logs.
2. Print the current image size and batch size.
3. Recommend lowering batch size only.
4. Do not automatically lower `imgsz` because 1280 is a fixed Phase 1 requirement.
5. Allow gradient accumulation only if it is explicitly introduced and recorded in a later configuration.

# 19. Evaluation Design

Evaluation has two layers:

1. **Ultralytics evaluation** for standard mAP, precision-recall curves, confusion matrix, and framework outputs.
2. **Project-controlled evaluation** for explicit TP/FP/FN matching, false-negative rate, small-object recall, threshold sweeps, latency, and qualitative failure analysis.

## 19.1 Validation vs Test Use

- Validation split: model comparison, checkpoint promotion, confidence-threshold analysis, and experiment decisions.
- Test split: one locked final evaluation after a candidate has been promoted to champion.
- Test metrics must not be used to decide which candidate becomes champion.

## 19.2 `scripts/evaluate.py`

Validation example:

```bash
python scripts/evaluate.py \
  --model runs/train/<run>/weights/best.pt \
  --data data/processed/fod_a_single_class_yolo/fod_a.yaml \
  --split val \
  --config configs/evaluate.yaml
```

Final test example:

```bash
python scripts/evaluate.py \
  --model artifacts/champion/fod_yolo26n_best.pt \
  --data data/processed/fod_a_single_class_yolo/fod_a.yaml \
  --split test \
  --config configs/evaluate.yaml \
  --locked-threshold <selected-validation-threshold>
```

# 20. Required Metrics

## 20.1 Standard Detection Metrics

The evaluation output must include:

- Precision.
- Recall.
- F1 score.
- mAP@0.50.
- mAP@0.75.
- mAP@0.50:0.95.
- True positives.
- False positives.
- False negatives.
- Number of ground-truth objects.
- Number of predictions.

Ultralytics validation supports mAP metrics and generated validation plots [R4].

## 20.2 Safety-Relevant Project Metrics

The evaluation output must additionally include:

```text
false_negative_rate = FN / (TP + FN)
false_positives_per_image = FP / evaluated_images
images_with_false_negatives
fraction_of_images_with_false_negatives
small_object_recall
small_object_false_negative_rate
small_object_ground_truth_count
small_object_true_positives
small_object_false_negatives
```

## 20.3 Small-Object Definition

For this project, a ground-truth object is considered small when:

```text
(box_width * box_height) / (image_width * image_height) <= 0.01
```

In normalized YOLO coordinates this is equivalent to:

```text
normalized_width * normalized_height <= 0.01
```

The threshold must be configurable. The report must state the exact threshold used and the count of ground-truth boxes in the small-object group.

## 20.4 Runtime Metrics

Measure and report:

- Mean preprocessing latency per image.
- Mean model inference latency per image.
- Mean postprocessing latency per image.
- Mean end-to-end latency per image.
- Median end-to-end latency.
- P95 end-to-end latency.
- Throughput FPS.
- Peak GPU memory allocated when available.
- Model file size.

Latency runs must include a warm-up phase and must record the hardware, batch size, precision, and image size.

# 21. Project-Controlled Matching Algorithm

For threshold-specific metrics at IoU `0.50`, use deterministic one-to-one matching per image:

1. Filter predictions below the selected confidence threshold.
2. Sort predictions by descending confidence.
3. Compute IoU against unmatched ground-truth boxes.
4. Match a prediction to the unmatched ground-truth box with the highest IoU when IoU is at least the configured threshold.
5. Count each ground-truth box at most once.
6. Unmatched predictions are false positives.
7. Unmatched ground-truth boxes are false negatives.
8. Because the dataset is single-class, class matching is trivially class `0`, but class ID must still be validated.

This matching implementation must have independent unit tests with hand-calculated examples.

# 22. Confidence-Threshold Analysis

The evaluator must sweep confidence thresholds from `0.05` through `0.95` in configurable increments.

For every threshold, save:

- TP.
- FP.
- FN.
- Precision.
- Recall.
- F1.
- False-negative rate.
- False positives per image.
- Small-object recall.
- Small-object false-negative rate.

The report must identify three reference thresholds:

1. `best_f1_threshold`: threshold with the highest F1 score.
2. `max_recall_threshold`: highest threshold that still achieves the maximum observed recall within numeric tolerance.
3. `balanced_high_recall_threshold`: among thresholds whose recall is within one percentage point of maximum recall, choose the one with the lowest false positives per image.

These are analytical references, not automatic production settings. The final runway threshold must be selected using the future runway validation set and operational false-alarm tolerance.

# 23. Qualitative Evaluation

The evaluator must generate visual examples for:

- Ground truth beside model prediction.
- True positives.
- Highest-confidence false positives.
- False negatives.
- Small-object false negatives.
- Images with multiple detections.
- Images with no detections.

For side-by-side images:

- Left panel: ground-truth boxes.
- Right panel: predicted boxes at the chosen evaluation threshold.
- Each source image gets one row.
- Include image ID, confidence values, and IoU for matched detections where practical.

# 24. Metrics Storage Schema

`metrics.json` must use a stable schema:

```json
{
  "schema_version": "1.0",
  "run_id": "<run-id>",
  "model_sha256": "<sha256>",
  "dataset_fingerprint": "<fingerprint>",
  "split": "val",
  "imgsz": 1280,
  "confidence_threshold": 0.25,
  "matching_iou_threshold": 0.5,
  "counts": {
    "images": 0,
    "ground_truth": 0,
    "predictions": 0,
    "true_positives": 0,
    "false_positives": 0,
    "false_negatives": 0
  },
  "metrics": {
    "precision": 0.0,
    "recall": 0.0,
    "f1": 0.0,
    "false_negative_rate": 0.0,
    "false_positives_per_image": 0.0,
    "map50": 0.0,
    "map75": 0.0,
    "map50_95": 0.0,
    "small_object_recall": 0.0,
    "small_object_false_negative_rate": 0.0
  },
  "latency": {
    "mean_inference_ms": 0.0,
    "mean_end_to_end_ms": 0.0,
    "median_end_to_end_ms": 0.0,
    "p95_end_to_end_ms": 0.0,
    "fps": 0.0
  },
  "threshold_references": {
    "best_f1_threshold": 0.0,
    "max_recall_threshold": 0.0,
    "balanced_high_recall_threshold": 0.0
  }
}
```

# 25. Metrics Display and Reporting

## 25.1 TensorBoard

Ultralytics supports TensorBoard logging for training metrics [R10]. The README must include:

```bash
tensorboard --logdir runs/train --port 6006
```

The user must be able to view:

- Training losses by epoch.
- Validation losses by epoch.
- Precision.
- Recall.
- mAP@0.50.
- mAP@0.50:0.95.
- Learning-rate progression.
- Sample training and validation images where logged.

When accessed over VS Code Remote SSH, the README must mention forwarding port `6006` through VS Code’s Ports panel.

## 25.2 Static Plots

The generated report must include:

- Precision-recall curve.
- Precision vs confidence.
- Recall vs confidence.
- F1 vs confidence.
- False-negative rate vs confidence.
- False positives per image vs confidence.
- Small-object recall vs confidence.
- Confusion matrix.
- Normalized confusion matrix.
- Box-area distribution.
- Recall by box-area bucket.
- Latency distribution.
- Training metric curves copied from or derived from the run.

## 25.3 HTML Report

`scripts/build_report.py` must create a self-contained `index.html` with:

- Run identity and model checksum.
- Dataset identity and fingerprint.
- Training configuration.
- Hardware and dependency versions.
- Core metric cards.
- Threshold table.
- Plots.
- Qualitative examples.
- Links to JSON and CSV files.
- Champion status.
- Limitations and warnings.

Required command:

```bash
python scripts/build_report.py --evaluation-dir reports/<run-id>
```

# 26. Definition of “Best Model”

The implementation must explicitly distinguish:

## 26.1 Per-Run Best Checkpoint

`runs/train/<run>/weights/best.pt` is the checkpoint saved by Ultralytics as the best checkpoint within one training run.

## 26.2 Per-Run Latest Checkpoint

`runs/train/<run>/weights/last.pt` is the latest checkpoint and is used for resuming interrupted training.

## 26.3 Project Champion

`artifacts/champion/fod_yolo26n_best.pt` is the best eligible candidate across completed runs according to the project’s validation-based promotion policy.

Only the project champion is published as the active model release.

# 27. Champion Promotion Policy

## 27.1 `scripts/promote_model.py`

Example:

```bash
python scripts/promote_model.py \
  --candidate-run runs/train/<run-id> \
  --candidate-evaluation reports/<run-id>/metrics.json \
  --config configs/promotion.yaml
```

A candidate is eligible only when:

- Training completed successfully.
- `best.pt` exists and is readable.
- Validation evaluation completed.
- Required metrics are finite and complete.
- Dataset fingerprint is present.
- Model SHA-256 is present.
- Evaluation split is `val`.
- The candidate is not compared using test metrics.

## 27.2 Ranking

Candidates are compared lexicographically, using the order in `promotion.yaml`:

1. Higher small-object recall.
2. Higher overall recall.
3. Higher mAP@0.50:0.95.
4. Lower false positives per image.
5. Lower P95 inference latency.

A configurable tolerance prevents meaningless replacement due only to tiny floating-point differences.

If no champion exists, the first eligible candidate becomes champion.

## 27.3 Promotion Outputs

Promotion must atomically update:

```text
artifacts/champion/fod_yolo26n_best.pt
artifacts/champion/model_metadata.json
artifacts/champion/model_card.md
artifacts/champion/evaluation_metrics.json
artifacts/champion/evaluation_metrics.csv
artifacts/champion/training_config.yaml
artifacts/champion/dataset_manifest.json
artifacts/champion/SHA256SUMS
models/registry.json
```

The previous champion metadata must be archived under `artifacts/candidates/history/` or referenced in registry history.

# 28. Final Test Evaluation

After promotion, run one locked test evaluation using the threshold selected from validation. Store the result separately from validation metrics.

The final model card must clearly show:

- Validation metrics used for promotion.
- Test metrics used only for final reporting.
- Locked confidence threshold.
- Matching IoU threshold.
- Small-object definition.
- Dataset fingerprint.

# 29. Model Metadata and Model Card

`model_metadata.json` must include:

- Model display name.
- Model architecture.
- Source checkpoint.
- Model SHA-256.
- File size.
- Run ID.
- Training start and end timestamps.
- Git commit.
- Python version.
- PyTorch version.
- torchvision version.
- CUDA version.
- cuDNN version.
- Ultralytics version.
- GPU model.
- Image size.
- Epochs completed.
- Batch setting and resolved batch size when available.
- Dataset fingerprint.
- Split counts.
- Class mapping.
- Validation metrics.
- Test metrics.
- Selected confidence threshold.
- Known limitations.

`model_card.md` must be human-readable and contain:

- Intended use.
- Out-of-scope use.
- Training data summary.
- Single-class mapping explanation.
- Training configuration.
- Evaluation metrics.
- Threshold guidance.
- Failure modes.
- Video usage example.
- Checksum and release information.

# 30. GitHub Publication

## 30.1 Default Distribution Method: GitHub Release

GitHub Releases support attaching binary files to a tagged release [R8]. The default publication must create a release such as:

```text
Tag: fod-yolo26n-v1.0.0
Title: FOD YOLO26n Champion v1.0.0
```

Release assets:

```text
fod_yolo26n_best.pt
model_metadata.json
model_card.md
evaluation_metrics.json
evaluation_metrics.csv
training_config.yaml
dataset_manifest.json
SHA256SUMS
```

## 30.2 `scripts/publish_model.py`

Example:

```bash
python scripts/publish_model.py \
  --repo owner/repository \
  --tag fod-yolo26n-v1.0.0 \
  --champion-dir artifacts/champion \
  --update-registry \
  --push-registry
```

Required behavior:

1. Require explicit invocation. Training must not publish automatically.
2. Verify all checksums.
3. Verify the asset is the current local champion.
4. Require `GH_TOKEN` or an authenticated `gh` session.
5. Create a draft release first.
6. Upload all required assets.
7. Verify uploaded asset names and sizes.
8. Publish the release only after all assets are present.
9. Update `models/registry.json` with the active release tag and checksum.
10. Optionally commit and push only the registry and model-card metadata.
11. Never commit secrets or dataset files.

## 30.3 Optional Git LFS Alternative

Git LFS may be supported for teams that require the `.pt` file inside the repository history. GitHub documents Git LFS as the mechanism for files beyond normal repository limits [R9]. It is not the default because repeated binary model versions can consume repository and LFS storage.

# 31. Model Registry

`models/registry.json` must be small and committed to Git:

```json
{
  "schema_version": "1.0",
  "active_model": {
    "name": "fod_yolo26n_best",
    "release_tag": "fod-yolo26n-v1.0.0",
    "asset_name": "fod_yolo26n_best.pt",
    "sha256": "<sha256>",
    "imgsz": 1280,
    "class_names": ["FOD"],
    "selected_confidence_threshold": 0.25,
    "published_at_utc": "<timestamp>"
  },
  "history": []
}
```

# 32. Model Download

## 32.1 `scripts/download_model.py`

Examples:

```bash
python scripts/download_model.py --repo owner/repository --active
```

```bash
python scripts/download_model.py \
  --repo owner/repository \
  --tag fod-yolo26n-v1.0.0 \
  --output models/downloaded/fod_yolo26n_best.pt
```

Required behavior:

- Read `models/registry.json` when `--active` is used.
- Download the matching GitHub Release asset.
- Support public and private repositories.
- Use `GH_TOKEN` only when needed.
- Verify SHA-256.
- Delete a corrupted partial file.
- Use a temporary file followed by atomic rename.
- Print the final local model path.

# 33. Image Inference

`scripts/infer_image.py` must accept one image or a directory.

Example:

```bash
python scripts/infer_image.py \
  --model artifacts/champion/fod_yolo26n_best.pt \
  --source data/samples \
  --imgsz 1280 \
  --conf 0.25 \
  --device 0 \
  --save
```

Required outputs:

- Annotated images.
- Per-image JSON or CSV detections.
- Summary metrics such as images processed, detections, mean confidence, and latency.

# 34. Video Inference

Ultralytics predict mode supports video sources and streaming results [R5]. The repository must provide a project-controlled wrapper so outputs and metadata remain stable across Ultralytics versions.

## 34.1 `scripts/infer_video.py`

Example:

```bash
python scripts/infer_video.py \
  --model artifacts/champion/fod_yolo26n_best.pt \
  --source data/test_videos/runway_test.mp4 \
  --imgsz 1280 \
  --conf 0.25 \
  --device 0 \
  --frame-stride 1 \
  --save-video \
  --save-csv
```

## 34.2 Required Behavior

1. Open and validate the video.
2. Read source width, height, FPS, frame count, codec when available, and duration.
3. Process frames using a memory-efficient iterator.
4. Preserve original source frames; resizing is internal to inference.
5. Draw boxes and confidence labels on the annotated output.
6. Save a detection record for every prediction.
7. Record per-frame inference time.
8. Support frame stride.
9. Support an optional start time and end time.
10. Save only detection frames when requested.
11. Handle an interrupted run by retaining a partial summary marked incomplete.
12. Avoid loading the entire video into memory.

## 34.3 Video Output Layout

```text
runs/inference/video/<run-id>/
├── annotated_video.mp4
├── detections.csv
├── frame_metrics.csv
├── video_summary.json
├── inference_config.yaml
├── source_metadata.json
└── detection_frames/
```

## 34.4 Detection CSV Schema

```text
video_name
frame_index
timestamp_seconds
class_id
class_name
confidence
x1
x2
y1
y2
box_width_pixels
box_height_pixels
box_area_pixels
box_area_ratio
preprocess_ms
inference_ms
postprocess_ms
```

Use zero-based `frame_index`. Coordinates must refer to the original video frame.

## 34.5 Video Summary Schema

The summary must include:

- Source video metadata.
- Total frames in source.
- Frames processed.
- Frame stride.
- Frames with one or more detections.
- Total detections.
- Mean, median, and maximum confidence.
- Mean and P95 inference latency.
- Processing FPS.
- Source FPS.
- Real-time ratio: `processing_fps / source_fps`.
- Completion status.
- Model checksum.
- Confidence threshold.
- Image size.

# 35. Limitation of Unlabelled Video Evaluation

An unlabelled video can show detections, confidence distributions, persistence, latency, and throughput. It cannot provide true recall, precision, mAP, or false-negative rate because the ground truth is unknown.

The video report must explicitly label such a run as `inference-only` rather than `accuracy evaluation`.

A later labelled-video evaluator may calculate frame-level TP, FP, FN, recall, and false alarms per minute after a video annotation format is selected. This future extension must reuse the same detection CSV schema.

# 36. Logging

All scripts must use Python logging with:

- ISO-8601 UTC timestamps.
- Log level.
- Module name.
- Human-readable console output.
- Rotating file logs for long-running training or video inference.
- No secret values.

Each top-level script must return a non-zero exit code on failure.

# 37. Reproducibility Requirements

Every run must record:

- Resolved configuration.
- Random seed.
- Dataset fingerprint.
- Model source checkpoint.
- Model source checksum when available.
- Git commit and dirty state.
- Python and package versions.
- CUDA and cuDNN versions.
- GPU name and VRAM.
- Operating system.
- Start and end timestamps.
- Training command.
- Resume source if applicable.

`pip freeze` must be saved as `environment_freeze.txt` inside each completed run.

# 38. `.gitignore` Requirements

At minimum:

```gitignore
# Secrets
.env
kaggle.json

# Python
.venv/
__pycache__/
*.py[cod]
.pytest_cache/
.mypy_cache/
.ruff_cache/

# Data
data/raw/
data/processed/
data/cache/
*.zip

# Training and reports
runs/
artifacts/
reports/

# Model binaries
*.pt
*.onnx
*.engine

# IDE and OS
.vscode/
.idea/
.DS_Store
Thumbs.db
```

`models/registry.json` and documentation under `models/` must remain tracked.

# 39. Makefile Targets

The repository must provide:

```text
make setup
make check-env
make download-data
make prepare-data
make validate-data
make train
make evaluate-val
make promote
make evaluate-test
make report
make publish-model
make download-model
make infer-video VIDEO=<path>
make test
make lint
```

Targets must call the Python scripts rather than duplicating business logic in shell code.

# 40. Testing Requirements

## 40.1 Unit Tests

Tests must cover:

- Correct VOC XML parsing.
- Correct conversion of known boxes to YOLO coordinates.
- All output class IDs are zero.
- Clipping behavior.
- Rejection of degenerate boxes.
- Dimension mismatch handling.
- Deterministic split membership.
- Split disjointness.
- Dataset validator failures.
- IoU calculation.
- One-to-one detection matching.
- TP/FP/FN metric calculations.
- Small-object classification.
- Threshold ranking.
- Champion ranking and tie tolerance.
- Registry serialization.
- SHA-256 verification.
- Video CSV schema.

## 40.2 Smoke Test

A tiny fixture dataset must support:

```bash
pytest -m smoke
```

The smoke test must:

- Convert a few VOC images.
- Validate the resulting YOLO dataset.
- Load YOLO26n.
- Optionally run one CPU prediction when network/model availability permits.

The full Kaggle dataset must never be downloaded by normal CI.

# 41. Continuous Integration

`.github/workflows/ci.yml` must run on pull requests and pushes:

- Python 3.11.
- Install CPU-compatible dependencies.
- Ruff format check.
- Ruff lint.
- Mypy for project modules.
- Pytest with coverage.
- No Kaggle download.
- No GPU training.
- No GitHub model publication.

`.github/workflows/publish-model.yml` should be manually triggered and must require protected secrets. It may publish already prepared champion assets, but it must not train the model inside standard GitHub-hosted CI.

# 42. Error Handling

The implementation must use explicit exceptions and actionable errors for:

- Missing Kaggle credentials.
- Download failure.
- Corrupt archive.
- Missing VOC directories.
- Ambiguous VOC roots.
- Invalid XML.
- Missing image.
- Invalid bounding box.
- Split overlap.
- CUDA unavailable.
- GPU out of memory.
- Missing checkpoint.
- Mismatched dataset fingerprint on resume.
- Incomplete metrics on promotion.
- GitHub authentication failure.
- Release asset upload failure.
- Checksum mismatch.
- Unreadable video.
- Video writer initialization failure.

Never catch a broad exception and continue as though a stage succeeded.

# 43. Security Requirements

- Never commit Kaggle or GitHub credentials.
- Never echo secret environment variables.
- Do not execute arbitrary shell commands derived from YAML values.
- Validate paths and create outputs only under configured project roots unless explicitly overridden.
- Download to a temporary file and verify before replacing a model.
- Record checksums for archives and model artifacts.
- Treat externally downloaded archives as untrusted input and prevent path traversal during extraction.
- Do not automatically publish a model after training.

# 44. Implementation Sequence for Codex

Codex must implement in this order:

1. Repository skeleton and packaging.
2. Configuration loading and path utilities.
3. Hashing, logging, and environment reporting.
4. Kaggle authentication and dataset download.
5. VOC discovery and parser.
6. Deterministic split logic.
7. Single-class conversion.
8. Dataset validation and statistics.
9. Unit tests for data pipeline.
10. Training wrapper and run metadata.
11. Evaluation wrapper and standard metrics.
12. Project-controlled matching and threshold sweep.
13. Qualitative plots and HTML report.
14. Champion promotion.
15. GitHub Release publication and download.
16. Image inference.
17. Video inference.
18. CI, Makefile, and final README.

A pull request should not implement all modules as one monolithic script.

# 45. End-to-End Commands

## 45.1 Linux/macOS Shell

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Use a CUDA index selected from the official PyTorch installation page.
python scripts/install_torch.py --index-url <official-pytorch-cuda-index>
pip install -r requirements/base.txt
pip install -r requirements/dev.txt

python scripts/check_environment.py --require-cuda
python scripts/download_dataset.py --config configs/dataset.yaml
python scripts/prepare_dataset.py --config configs/dataset.yaml
python scripts/validate_dataset.py \
  --data data/processed/fod_a_single_class_yolo/fod_a.yaml \
  --strict

python scripts/train.py --config configs/train_yolo26n_1280.yaml

python scripts/evaluate.py \
  --model runs/train/<run-id>/weights/best.pt \
  --split val \
  --config configs/evaluate.yaml

python scripts/promote_model.py \
  --candidate-run runs/train/<run-id> \
  --candidate-evaluation reports/<run-id>/metrics.json \
  --config configs/promotion.yaml

python scripts/evaluate.py \
  --model artifacts/champion/fod_yolo26n_best.pt \
  --split test \
  --config configs/evaluate.yaml \
  --locked-threshold <validation-selected-threshold>

python scripts/build_report.py --evaluation-dir reports/<run-id>

python scripts/publish_model.py \
  --repo owner/repository \
  --tag fod-yolo26n-v1.0.0 \
  --champion-dir artifacts/champion \
  --update-registry \
  --push-registry
```

## 45.2 Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

python scripts/install_torch.py --index-url <official-pytorch-cuda-index>
pip install -r requirements/base.txt
pip install -r requirements/dev.txt

python scripts/check_environment.py --require-cuda
python scripts/download_dataset.py --config configs/dataset.yaml
python scripts/prepare_dataset.py --config configs/dataset.yaml
python scripts/validate_dataset.py --data data/processed/fod_a_single_class_yolo/fod_a.yaml --strict
python scripts/train.py --config configs/train_yolo26n_1280.yaml
```

# 46. Acceptance Criteria

Phase 1 implementation is complete only when all of the following are true:

## 46.1 Environment

- A clean environment can install dependencies using documented commands.
- CUDA verification succeeds on the desktop GPU.
- The exact environment is recorded.

## 46.2 Data

- The dataset downloads through Kaggle without being committed.
- Extraction is idempotent and safe.
- The VOC root is discovered automatically.
- Train, validation, and test splits are deterministic and disjoint.
- Every retained label uses class ID `0`.
- The dataset YAML contains only `0: FOD`.
- Strict validation passes.
- A dataset manifest and fingerprint are generated.

## 46.3 Training

- YOLO26n trains at `imgsz=1280` on CUDA.
- `best.pt` and `last.pt` are saved.
- Periodic checkpoints are saved at the configured interval.
- Interrupted training can resume from `last.pt`.
- Run configuration and environment metadata are preserved.

## 46.4 Evaluation

- Standard mAP metrics are generated.
- TP, FP, FN, recall, FNR, and false positives per image are generated.
- Small-object recall is generated.
- Confidence-threshold analysis is generated.
- Static plots, JSON, CSV, and HTML report are generated.
- Ground-truth vs prediction examples and FP/FN examples are generated.
- TensorBoard can display training metrics.

## 46.5 Champion and Publishing

- A candidate can be promoted using validation metrics only.
- The champion artifacts are atomically created.
- SHA-256 verification succeeds.
- A GitHub Release can be created with the model and metadata.
- `models/registry.json` identifies the active model.
- A fresh clone can download and verify the active model.

## 46.6 Video

- A video file can be processed using the champion model.
- An annotated video is saved.
- Detection CSV and summary JSON are saved.
- Frame indexes, timestamps, bounding boxes, confidence, and latency are correct.
- Unlabelled video output is correctly described as inference-only.

## 46.7 Quality

- Unit tests pass.
- Linting passes.
- CI does not require the full dataset or a GPU.
- Secrets, data, runs, and model binaries are excluded from ordinary Git commits.

# 47. Future Extension Points

The implementation must keep these interfaces stable for later phases:

1. A different dataset YAML can be supplied without changing training code.
2. Phase 2 runway data can reuse the same conversion and validation contracts.
3. The detector can be loaded through one `FODDetector` interface for image and video inference.
4. A second detector or runway-feature expert can be added without modifying dataset download logic.
5. A tracker can later attach temporal IDs to video detections.
6. A labelled-video evaluator can consume the existing detection CSV.
7. ONNX or TensorRT export can be added after the PyTorch champion is validated.
8. A future model registry may move from GitHub Releases to a dedicated MLOps platform without changing model metadata format.

# 48. Deliverables Expected from Codex

Codex must produce:

- The complete repository structure.
- All Python source files.
- YAML configurations.
- Requirements and packaging files.
- Unit tests and fixtures.
- README with setup and execution instructions.
- Makefile.
- GitHub Actions workflows.
- `.env.example` and `.gitignore`.
- No dataset files.
- No secret files.
- No fabricated model metrics.

Codex must not claim the model is accurate until a real training and evaluation run has produced metrics.

# 49. References

- **[R1] Kaggle FOD-A dataset:** https://www.kaggle.com/datasets/kilogrand/foreign-object-debris-in-airports-fod-a-dataset
- **[R2] Ultralytics YOLO26 model documentation:** https://docs.ultralytics.com/models/yolo26/
- **[R3] Ultralytics configuration and training arguments:** https://docs.ultralytics.com/usage/cfg/
- **[R4] Ultralytics validation mode:** https://docs.ultralytics.com/modes/val/
- **[R5] Ultralytics prediction mode:** https://docs.ultralytics.com/modes/predict/
- **[R6] PyTorch local installation selector:** https://pytorch.org/get-started/locally/
- **[R7] Official Kaggle CLI:** https://github.com/Kaggle/kaggle-cli
- **[R8] GitHub Releases documentation:** https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository
- **[R9] GitHub Large File Storage documentation:** https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage
- **[R10] Ultralytics TensorBoard integration:** https://docs.ultralytics.com/integrations/tensorboard/

# Appendix A. Required Top-Level Script Interfaces

| Script | Required principal arguments | Principal outputs |
|---|---|---|
| `install_torch.py` | `--index-url`, optional version pins | Installed torch environment and console verification |
| `check_environment.py` | `--require-cuda` | `reports/environment_report.json` |
| `download_dataset.py` | `--config`, `--force` | archive, extracted dataset, source manifest |
| `prepare_dataset.py` | `--config` | converted YOLO dataset, manifest, statistics |
| `validate_dataset.py` | `--data`, `--strict` | validation report |
| `train.py` | `--config`, overrides, `--resume` | Ultralytics run, best/last checkpoints, metadata |
| `evaluate.py` | `--model`, `--split`, `--config` | metrics, predictions, plots, examples |
| `build_report.py` | evaluation directory | HTML and Markdown report |
| `promote_model.py` | candidate run, metrics, config | champion directory and registry update |
| `publish_model.py` | repo, tag, champion directory | GitHub Release |
| `download_model.py` | repo, active/tag, output | verified local `.pt` file |
| `infer_image.py` | model, source, threshold | annotated images and detections |
| `infer_video.py` | model, video, threshold | annotated video, CSV, JSON |

# Appendix B. Required Exit Codes

| Code | Meaning |
|---:|---|
| 0 | Success |
| 2 | Invalid command-line arguments or configuration |
| 3 | Missing credentials or authentication failure |
| 4 | Dataset download or extraction failure |
| 5 | Dataset validation failure |
| 6 | Environment or CUDA validation failure |
| 7 | Training failure |
| 8 | Evaluation failure |
| 9 | Promotion failure |
| 10 | Publication or download failure |
| 11 | Video input/output failure |

# Appendix C. Implementation Notes for Codex

1. Use `pathlib.Path` for all paths.
2. Use dataclasses or typed Pydantic-style models, but avoid adding a heavy dependency unless needed.
3. Use atomic writes for JSON, YAML, registry, model downloads, and champion replacement.
4. Use UTF-8 for all text files.
5. Sort JSON keys where practical for stable diffs.
6. Keep business logic in `src/fod_yolo`; top-level scripts should be thin CLI wrappers.
7. Add docstrings and type hints to public functions.
8. Do not hard-code local usernames, drive letters, SSH hostnames, or absolute desktop paths.
9. Do not hard-code dataset counts as correctness requirements; calculate them from the downloaded dataset version and record them.
10. Do not fabricate evaluation values in examples or documentation.
