# REDS Video Deblurring: Shared Dataset Pipeline Documentation

## 1. Overview
The shared dataset pipeline provides a model-agnostic, reproducible data loading and preprocessing framework for video motion deblurring on the REDS dataset. It supports multi-frame video models (e.g., EDVR, VRT) and single-frame architectures (e.g., MIMO-UNet, BSSTNet).

## 2. Dataset Root Configuration
Configure dataset directories in [`configs/data.yaml`](../../configs/data.yaml) or via `project_root`:
```yaml
paths:
  raw_dir: data/raw
  manifests_dir: data/manifests
  splits_dir: splits
  reports_dir: reports/data_pipeline
```
All paths resolve dynamically relative to the project root using `get_project_root()`.

## 3. Reuse of Existing EDA Outputs
The pipeline reuses the verified EDA reports to avoid repetitive directory scanning:
- [`reports/eda/frame_pairing_summary.csv`](../eda/frame_pairing_summary.csv): Source sequence index with 100% 1-to-1 pairing guarantee.
- [`reports/eda/dataset_inventory.csv`](../eda/dataset_inventory.csv): Discovered dataset structures and frame counts.
- [`reports/eda/image_validation_results.csv`](../eda/image_validation_results.csv): Integrity confirmation across all 54,000 images.

## 4. 90:10 Sequence-Level Split
Run the deterministic sequence split generator:
```python
from src.data.splitting import generate_sequence_split
df_split, summary = generate_sequence_split(train_ratio=0.90, test_ratio=0.10, random_seed=42)
```
- **Train:** 216 sequences (21,600 frames) from `train_blur` / `train_sharp`.
- **Held-Out Test:** 24 sequences (2,400 frames) from `train_blur` / `train_sharp` (ground-truth available for final benchmarking; never used during training/validation).
- **Validation:** 30 sequences (3,000 frames) from official `val_blur` / `val_sharp`.
- **Inference Test:** 30 sequences (3,000 frames) from official `test_blur` (unpaired).

Manifests generated:
- [`data/manifests/sequence_split.csv`](../../data/manifests/sequence_split.csv)
- [`data/manifests/split_summary.json`](../../data/manifests/split_summary.json)
- [`splits/sequence_split.csv`](../../splits/sequence_split.csv)
- [`splits/split_summary.json`](../../splits/split_summary.json)
- [`reports/data_pipeline/split_report.md`](split_report.md)

## 5. Usage & Dataset Loading
```python
from src.data.dataset import REDSVideoDeblurDataset
from src.preprocessing.paired_transforms import PairedTransformPipeline

# Configurable paired transformations
transform = PairedTransformPipeline(
    crop_size=(256, 256),
    flip_h=True,
    flip_v=True,
    temporal_reverse=True,
    seed=42
)

# Load training dataset (T=5 consecutive frames, center frame target)
train_dataset = REDSVideoDeblurDataset(
    partition="train",
    window_size=5,
    boundary_policy="replicate", # 'replicate', 'reflect', 'truncate', or 'zero_pad'
    transform=transform
)

sample = train_dataset[0]
blur_frames = sample["blur_frames"]  # List of 5 cropped PIL Images
sharp_frame = sample["sharp_frame"]  # Center target cropped PIL Image
```

## 6. How Future Model Adapters Should Interface
- **EDVR / VRT (Multi-frame models):** Set `window_size=5` (or model-specific odd window $T$), convert PIL images to PyTorch tensors `(B, T, C, H, W)` in adapter DataLoader collate function.
- **MIMO-UNet / Single-frame models:** Set `window_size=1`, returning single input frame `(B, C, H, W)` and corresponding ground truth.
- **Inference Challenge:** Set `partition="inference_test"` to stream `test_blur` without ground truth.

## 7. EDVR Baseline Architecture & Hardware Validation
The official EDVR (Enhanced Deformable Video Restoration) baseline for REDS motion deblurring has been audited and faithfully implemented:
- Architecture details, resolution pathway, and official configuration comparison: [`reports/data_pipeline/edvr_architecture_and_validation_report.md`](edvr_architecture_and_validation_report.md)
- GPU Memory Validation Report: [`reports/data_pipeline/gpu_memory_validation_report.json`](gpu_memory_validation_report.json)
- Tiny-Subset Overfitting Experiment Report: [`reports/data_pipeline/overfitting_experiment_report.json`](overfitting_experiment_report.json)
- Verification Test Suite (39 passed): `pytest tests/ -v`
- Readiness Decision: **READY FOR FULL EDVR TRAINING**.

