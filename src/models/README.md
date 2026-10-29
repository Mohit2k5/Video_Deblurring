# Model Architecture & Adapter (`src/models/`)

This directory contains the verified PyTorch implementation of the **EDVR (Enhanced Deformable Video Restoration)** model and its training/inference adapter.

---

## 1. `src/models/edvr_arch.py` (`EDVR`)
Official EDVR architecture implementation aligned with upstream BasicSR commit `b02e63a`:
- **PreDeblur Module (`PredeblurModule`)**:
  - Downsamples input frames by $\times 4$ using strided convolutions and 5 residual blocks to enlarge receptive fields for large motion blurs.
- **PCD Alignment (`PCDAlignment`)**:
  - 3-level Pyramid, Cascading, Deformable alignment module.
  - Generates multi-scale feature pyramids ($L1, L2, L3$) and computes deformable convolution offsets and modulation masks.
  - Utilizes `torchvision.ops.deform_conv2d` via `DCNv2Pack` (numerically equivalent to standard convolution with zero offsets).
- **TSA Fusion (`TSAFusion`)**:
  - Temporal and Spatial Attention module.
  - Computes temporal attention weights across $T$ frames and cross-frame spatial attention maps ($320 \rightarrow 64$ channels) to fuse aligned features into a single reference feature map.
- **Reconstruction Trunk**:
  - 10 Residual Blocks with LeakyReLU ($\alpha = 0.1$).
- **PixelShuffle Reconstruction**:
  - $\times 4$ upsampling via two PixelShuffle layers followed by residual addition of the center blurry input frame.
- **Parameter Count**: Exactly **4,042,851** parameters in EDVR-M (active config).

---

## 2. `src/models/edvr_adapter.py` (`EDVRAdapter`)
High-level PyTorch wrapper bridging dataset loaders, loss functions, and checkpoint management:
- **`collate_fn`**: Collates dataset sample dicts into batch tensors `(B, T, C, H, W)` and `(B, C, H, W)`.
- **`prepare_batch`**: Normalizes inputs to float32 on the target device in range $[0.0, 1.0]$.
- **`criterion`**: Differentiable Charbonnier Loss $\sqrt{(pred - target)^2 + \epsilon^2}$.
- **`save_checkpoint` / `load_checkpoint`**:
  - Checkpoints save model weights, optimizer state, scheduler state, GradScaler state, step, epoch, RNG states, and a 9-point architecture signature.
  - Enforces strict architecture compatibility checks upon reload, preventing silent weight mismatches.
