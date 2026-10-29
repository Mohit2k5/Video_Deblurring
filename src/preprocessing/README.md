# Preprocessing & Transformations (`src/preprocessing/`)

This module implements temporal window generation and paired spatial transformations for multi-frame video deblurring.

---

## Key Modules

### 1. `src/preprocessing/temporal_windows.py`
Builds deterministic temporal sliding-window indices for video sequence frames:
- Supports arbitrary odd window size $T$ (e.g. $T=5$).
- Guarantees that temporal windows never cross sequence boundaries.
- Boundary handling policies:
  - `replicate`: Repeats the boundary frame (e.g. $[0, 0, 0, 1, 2]$ for index 0 with $T=5$). Matches official EDVR/BasicSR inference.
  - `reflect`: Reflects frames about the boundary.
  - `truncate`: Discards boundary windows that cannot form a full symmetric window.
  - `zero_pad`: Replaces out-of-bound indices with $-1$ (zero padding).

### 2. `src/preprocessing/paired_transforms.py`
Implements synchronous data augmentation across all $T$ blur frames and the center sharp ground-truth frame:
- `paired_random_crop`: Synchronously extracts matching spatial patches (e.g. $256 \times 256$) from all blur frames and the ground truth.
- `paired_horizontal_flip`: Random horizontal reflection applied identically across all frames.
- `paired_vertical_flip`: Random vertical reflection applied identically across all frames.
- `paired_temporal_reverse`: Reverses the temporal order of blur frames $[t_0, t_1, t_2, t_3, t_4] \rightarrow [t_4, t_3, t_2, t_1, t_0]$ while keeping the center ground truth frame invariant.
- `PairedTransformPipeline`: Composable pipeline combining cropping and augmentations with optional random seeding.
