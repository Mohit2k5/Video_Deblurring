# Configuration Files (`configs/`)

This directory contains the central configuration files controlling dataset loading, preprocessing, model architecture, training hyperparameters, and hardware adaptation.

---

## 1. `configs/edvr.yaml`
Controls the active EDVR model architecture and training hyperparameters.
- **Model Section (`model`)**:
  - `num_feat: 64`: Feature channel dimension for EDVR-M (active hardware-adapted configuration matching official BasicSR EDVR-M).
  - `num_frame: 5`: Temporal input sliding window ($T=5$).
  - `num_extract_block: 5`: Residual blocks in PreDeblur feature extraction.
  - `num_reconstruct_block: 10`: Residual blocks in reconstruction trunk.
  - `center_frame_idx: 2`: 0-indexed center frame to restore.
  - `hr_in: true`: Multi-scale PreDeblur pyramid with spatial downsampling for motion blur.
  - `with_predeblur: true`: PreDeblur module enabled.
  - `with_tsa: true`: Temporal and Spatial Attention fusion enabled.
  - `deformable_groups: 8`: Deformable convolution groups in PCD alignment.
- **Training Section (`train`)**:
  - `patch_size: 256`: Spatial crop size ($256 \times 256$).
  - `batch_size: 2`: Fitted for 8 GB VRAM.
  - `loss: CharbonnierLoss`: Differentiable Charbonnier loss with $\epsilon = 1 \times 10^{-12}$.
  - `lr: 2.0e-4`: Initial learning rate.
  - `lr_schedule`: Cosine annealing with $\eta_{\min} = 1 \times 10^{-7}$.
  - `use_amp: true`: Automatic Mixed Precision (FP16).

---

## 2. `configs/data.yaml`
Controls the shared, model-independent dataset pipeline:
- **Paths**: `data/raw`, `data/manifests`, `splits`, `reports/data_pipeline`.
- **Dataset Names**: `train_blur`, `train_sharp`, `val_blur`, `val_sharp`, `test_blur`.
- **Split Configuration**: 90:10 random sequence split with fixed random seed 42.
- **Loader Settings**: Temporal window size ($T=5$), boundary policy (`replicate`), supported extensions (`.png`, `.jpg`).
- **Preprocessing Augmentations**: Random horizontal flip, vertical flip, temporal reverse.

---

## 3. `configs/eda_config.yaml`
Controls dataset inventory and exploratory data analysis:
- Sequence directory verification, image pairing checks, statistical sampling, and visualization parameters.
