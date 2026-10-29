# REDS Dataset Storage (`data/`)

This directory houses the REDS (REalistic and Dynamic Scenes) dataset assets and canonical split manifests.

---

## 1. `data/raw/` (Raw Dataset Layout)

The raw REDS dataset is organized by sequence folders, where each folder contains 100 high-definition frames ($720 \times 1280$ resolution in PNG format, named `00000000.png` through `00000099.png`):

```text
data/raw/
├── train_blur/        # 240 sequences (000–239), 24,000 blurry frames
├── train_sharp/       # 240 sequences (000–239), 24,000 ground-truth sharp frames
├── val_blur/          # 30 sequences (000–029), 3,000 blurry frames
├── val_sharp/         # 30 sequences (000–029), 3,000 ground-truth sharp frames
└── test_blur/         # 30 sequences (000–029), 3,000 blurry frames (blind test, no GT)
```

**Total Dataset Size**: 300 sequences (30,000 frames total).

---

## 2. `data/manifests/` (Canonical Split Manifests)

- **`sequence_split.csv`**:
  Canonical manifest mapping every sequence to its partition, source dataset, ground-truth availability, and frame count:
  - `train`: 216 sequences (21,600 frames) from `train_blur` / `train_sharp`.
  - `held_out_test`: 24 sequences (2,400 frames) from `train_blur` / `train_sharp` (strictly held out from training).
  - `val`: 30 sequences (3,000 frames) from `val_blur` / `val_sharp` (used for validation during training).
  - `inference_test`: 30 sequences (3,000 frames) from `test_blur` (blind evaluation only).
- **`split_summary.json`**:
  JSON summary documenting sequence counts, random seed (42), split ratios (90:10 on train sequences), and explicit sequence ID lists.

*Note: Raw images are strictly immutable. Split manifests are mirrored in `splits/` for portability.*
