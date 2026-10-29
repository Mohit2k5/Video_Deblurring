# Stage 5 GPU Memory Reconciliation and Benchmark Report

## 1. Executive Summary & Resolution of Memory Discrepancies
Previous reports contained two sharply conflicting memory measurements:
1. `~2,694.5 MB` peak VRAM for Batch Size 1 ($256 \times 256$ crops) reported in `gpu_smoke_test_report.json`.
2. `~644 MB` peak VRAM for Batch Size 2 ($256 \times 256$ crops, AMP) reported in `gpu_memory_validation_report.json`.

### Technical Root Cause of the Discrepancy
The discrepancy was caused by an architectural difference in the feature resolution pathway:
- **Old Un-downsampled Draft (2,694.5 MB peak VRAM)**:
  - In the early draft, PreDeblur did not downsample the input by $x4$.
  - Consequently, all 5 feature extraction blocks, 3 levels of PCD alignment, TSA attention, and 10 reconstruction blocks processed activations at the full input resolution ($256 \times 256$).
  - Because intermediate spatial tensors at $256 \times 256$ have $16\times$ the element count of $64 \times 64$ tensors, caching backward-pass gradient activations for 15 residual blocks and multi-scale deformable convolutions consumed **2,694.5 MB** of VRAM for a single sample ($B=1$).
- **Corrected Official EDVR (367.9 MB / 673.1 MB peak allocated VRAM)**:
  - Official EDVR performs $x4$ downsampling inside PreDeblur via two stride-2 convolutions (`stride_conv_hr1`, `stride_conv_hr2`).
  - Alignment and reconstruction modules operate on $64 \times 64$ feature maps, reducing activation memory by a factor of 16.
  - Native resolution ($256 \times 256$) is restored at the very end via two `PixelShuffle(2)` operations.
  - As a result, peak allocated memory dropped to **367.9 MB** for $B=1$ (AMP) and **673.1 MB** for $B=2$ (AMP).

---

## 2. Rigorous GPU Benchmark Matrix (NVIDIA RTX 5060 Laptop GPU - 8,150.56 MB VRAM)
Measured on complete training steps (Forward Pass + Loss + Backward Pass + Optimizer Step) with CUDA synchronization and warmup iterations:

| Architecture Profile | Batch Size | Spatial Crop | Precision Mode | Peak Allocated VRAM | Peak Reserved VRAM | Peak Reserved % of VRAM | Fits in 8 GB | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Active EDVR** (`num_feat=64`, 10 blocks) | 1 | $256 \times 256$ | AMP (`fp16`) | **367.9 MB** | **532.0 MB** | 6.53% | Yes | **PASSED** |
| **Active EDVR** (`num_feat=64`, 10 blocks) | 2 | $256 \times 256$ | AMP (`fp16`) | **673.1 MB** | **848.0 MB** | 10.40% | Yes | **PASSED** |
| **Active EDVR** (`num_feat=64`, 10 blocks) | 1 | $256 \times 256$ | FP32 | **569.7 MB** | **768.0 MB** | 9.42% | Yes | **PASSED** |
| **Active EDVR** (`num_feat=64`, 10 blocks) | 2 | $256 \times 256$ | FP32 | **1,076.0 MB** | **1,428.0 MB** | 17.52% | Yes | **PASSED** |
| **Official EDVR-L** (`num_feat=128`, 40 blocks) | 1 | $256 \times 256$ | AMP (`fp16`) | **897.4 MB** | **1,178.0 MB** | 14.45% | Yes | **PASSED** |
| **Official EDVR-L** (`num_feat=128`, 40 blocks) | 2 | $256 \times 256$ | AMP (`fp16`) | **1,445.6 MB** | **2,102.0 MB** | 25.79% | Yes | **PASSED** |

---

## 3. Allocated vs Reserved VRAM Distinctions
- **Allocated Memory**: Exact memory occupied by active model parameters, gradients, and forward-pass activation tensors (e.g. 673.1 MB for $B=2$ AMP).
- **Reserved Memory**: Total memory held by PyTorch's caching memory allocator to avoid system-level `cudaMalloc` overhead (e.g. 848.0 MB for $B=2$ AMP).
- **VRAM Headroom**: On the 8,150.56 MB RTX 5060 Laptop GPU, even under full FP32 without AMP, Batch Size 2 reserves only 1,428.0 MB (leaving 6,722 MB free). Under AMP, reserved memory is only 848.0 MB (leaving over 7.3 GB free).

---

## 4. Stage 5 Gate Assessment
- **Status**: **PASSED**.
- The historical memory discrepancy is conclusively explained.
- Both Batch Size 1 and Batch Size 2 fit within GPU VRAM limits with over 80% headroom.
- Proceeding to Stage 6 is safe.
