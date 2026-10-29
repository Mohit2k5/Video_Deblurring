# Stage 8 Controlled Training Pilot Metrics and Verification Report

## 1. Executive Summary
- **Execution Script**: `src/training/train.py`
- **Hardware Platform**: NVIDIA GeForce RTX 5060 Laptop GPU (8,150.56 MB VRAM)
- **Dataset**: REDS Real Training Partition (`train`: 216 sequences) with Real Validation Partition (`val`: 30 sequences)
- **Total Training Steps**: **100 optimizer steps** (Batch Size 2, $256 \times 256$ spatial crops)
- **Total Execution Duration**: **89.3 seconds** (Average: $0.893\text{s}$ per step including validation)
- **Precision Mode**: PyTorch AMP (`torch.amp.autocast('cuda')` + `GradScaler`)
- **Optimizer**: Adam ($\text{lr} = 2.0 \times 10^{-4}$ with `CosineAnnealingLR` schedule down to $1.0 \times 10^{-7}$)
- **Loss Function**: Charbonnier Loss ($\epsilon = 1.0 \times 10^{-12}$)

---

## 2. Validation Progression & Generalization Metrics
Periodic evaluation performed on the unseen validation partition (`val`):

| Evaluation Point | Optimizer Step | Validation Loss | Validation PSNR (dB) | Validation PSNR Gain | Peak Allocated VRAM | Peak Reserved VRAM |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Initial (Step 0)** | 0 | 0.057090 | **21.58 dB** | Baseline | — | — |
| **Checkpoint 1** | 25 | 0.042044 | **23.33 dB** | **+1.75 dB** | 665.4 MB | 952.0 MB |
| **Checkpoint 2** | 50 | 0.047134 | **22.67 dB** | **+1.09 dB** | 665.4 MB | 952.0 MB |
| **Checkpoint 3** | 75 | 0.047064 | **22.49 dB** | **+0.91 dB** | 665.4 MB | 952.0 MB |
| **Final (Step 100)**| 100 | 0.044335 | **23.24 dB** | **+1.66 dB** | 665.4 MB | 952.0 MB |

Validation PSNR improved from 21.58 dB to 23.24 dB (+1.66 dB) across unseen validation sequences, confirming true generalizable deblurring optimization rather than mere single-batch memorization.

---

## 3. Stability and Resource Profiling
- **Gradient Finiteness**: Gradients remained strictly finite ($0 < \text{GradNorm} < \infty$) across all 100 steps.
- **Memory Stability**: Peak allocated VRAM stabilized at **665.4 MB** (8.16% of 8 GB), and peak reserved VRAM remained locked at **952.0 MB** (11.68% of 8 GB) with zero unbounded growth or leak.
- **Data Loading**: Real REDS dataset streaming with `num_workers=0` executed smoothly with no I/O errors or dropped frames.
- **Checkpoint Resilience**: Midpoint checkpoint (`pilot_checkpoint_step50.pth`, 48.75 MB) saved at step 50 and verified restored into a fresh model instance with step count, optimizer momentum, scheduler state, and scaler restored.

---

## 4. Stage 8 Gate Assessment
- **Status**: **PASSED**.
- The controlled pilot completed within resource limits, verified numerical stability, and confirmed validation metric improvement on real REDS data.
- Proceeding to Stage 9 and Stage 10 is safe.
