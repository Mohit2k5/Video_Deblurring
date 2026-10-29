# Pipeline Audit Report: REDS Video Deblurring

## 1. Executive Summary
An end-to-end audit was conducted across the `reds-video-deblurring` codebase, EDA reports, manifests, split configurations, and Python dependencies. 

The primary findings confirm that:
1. All 57,000 REDS frames (240 `train_blur`, 240 `train_sharp`, 30 `val_blur`, 30 `val_sharp`, 30 `test_blur`) are intact, uncorrupted, and uniform $1280 \times 720$ RGB PNGs.
2. The 90:10 sequence-level split partitions the 240 training sequences into exactly 216 `train` and 24 `held_out_test` sequences, completely preserving the 30 `val` sequences and 30 `test_blur` inference sequences.
3. Upstream PyTorch CUDA execution environment (`.venv_torch`) has been verified on Python 3.12 with PyTorch 2.11.0 + CUDA 12.8, detecting the local **NVIDIA GeForce RTX 5060 Laptop GPU (8 GB VRAM)**.

## 2. Split Protocol & Identity Audit
- **Composite Identity Requirement:** Sequence IDs (`000` to `029`) appear in multiple distinct source directories (e.g. `data/raw/train_blur/000` vs `data/raw/val_blur/000` vs `data/raw/test_blur/000`). To prevent namespace collision or cross-partition confusion, each sequence is tracked via a composite identity `(source_dataset, sequence_id)`.
- **Partition Independence:**
  - `train` (216 sequences, 21,600 frames) $\subset$ `data/raw/train_blur` & `train_sharp`.
  - `held_out_test` (24 sequences, 2,400 frames) $\subset$ `data/raw/train_blur` & `train_sharp`.
  - `val` (30 sequences, 3,000 frames) = `data/raw/val_blur` & `val_sharp`.
  - `inference_test` (30 sequences, 3,000 frames) = `data/raw/test_blur` (unpaired).
  - The intersection between `train` and `held_out_test` sequence sets is empty ($\emptyset$).

## 3. Confirmed Issues & Actions Taken
1. **Paired Crop Verification Rigor:** The original crop test only asserted output tensor dimensions (`(32, 32)`). It did not verify that pixel coordinate $(x, y)$ in the blurred frame maps to identical spatial coordinate $(x, y)$ in the sharp target. (Addressed in Phase 2 with coordinate-coded gradient patterns).
2. **Dual Manifest Synchronization:** Manifests currently exist in both `data/manifests/` and `splits/`. A canonical source of truth (`data/manifests/`) with an automated parity validation check is established.
3. **Type Annotation Fix:** Added missing `Any` import to `src/preprocessing/paired_transforms.py`.
