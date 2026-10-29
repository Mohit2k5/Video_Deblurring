# Data Loading & Dataset Management (`src/data/`)

This module provides model-independent dataset discovery, indexing, splitting, and PyTorch dataset loading utilities for the REDS dataset.

---

## Key Modules

### 1. `src/data/dataset.py` (`REDSVideoDeblurDataset`)
A model-independent PyTorch-compatible dataset class supporting:
- Partition loading (`'train'`, `'val'`, `'held_out_test'`, `'inference_test'`).
- Arbitrary odd temporal window sizes (default $T=5$ for EDVR).
- Boundary padding policies (`'replicate'`, `'reflect'`, `'truncate'`, `'zero_pad'`).
- Deterministic temporal indexing that never pairs frames across sequence boundaries.
- Flexible outputs (returns 5 blur PIL images/tensors and 1 center sharp ground-truth image).

### 2. `src/data/indexing.py`
Scans raw sequence folders, indexes frame files, and validates filename ordering (`00000000.png` ...).

### 3. `src/data/splitting.py`
Generates deterministic 90:10 sequence-level splits with fixed random seed 42 and writes manifests to `data/manifests/` and `splits/`.

### 4. `src/data/config_utils.py`
Helper utilities for resolving the project root directory and loading YAML configuration files.

### 5. `src/data/dataset_inventory.py`, `image_statistics.py`, `report_generation.py`
Exploratory data analysis utilities verifying image dimensions ($720 \times 1280$), channel integrity (RGB), and dataset metadata.
