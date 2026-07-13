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

No production dataset was downloaded in this milestone because that requires the maintainer's Kaggle credentials and network access. No training, evaluation, or model inference behavior is included yet.

### Suggested descriptive commit message

```text
Implement reproducible VOC-to-YOLO dataset pipeline
```
