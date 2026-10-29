# REDS Video Motion Deblurring

A reproducible, high-performance research codebase for video motion deblurring on the **REDS (REalistic and Dynamic Scenes)** dataset.

This repository implements the verified **EDVR (Enhanced Deformable Video Restoration)** baseline model with multi-scale PreDeblur, Pyramid Cascading Deformable (PCD) alignment, Temporal-Spatial Attention (TSA) fusion, a model-independent shared dataset pipeline, an automated test suite (62 tests), and a full evaluation harness for unseen benchmark sequences.

---

## Key Features & Achievements

- **Exact Upstream Fidelity**: Architectural and parameter count alignment with official upstream BasicSR EDVR (Wang et al., CVPRW 2019). Active EDVR-M configuration contains exactly **4,042,851 trainable parameters**.
- **Hardware-Adapted Efficiency**: Tailored for single-GPU execution on modern 8 GB mobile GPUs (NVIDIA GeForce RTX 5060 Laptop GPU). Consumes **< 1.0 GB VRAM** during batched training ($256 \times 256$ crops) and **< 1.5 GB VRAM** during full native $720 \times 1280$ resolution inference under AMP FP16.
- **Strict Split Protocol**: Fixed sequence-level split (seed 42) into 216 training sequences, 24 strictly held-out test sequences, 30 official validation sequences, and 30 inference-only sequences (`test_blur`). Zero partition leakage guaranteed by automated integrity checks.
- **Robust AMP Gradient Validation**: Unscales gradients via `torch.amp.GradScaler.unscale_` before finiteness checks and norm clipping, preventing false gradient-norm reporting and unstable updates.
- **Mathematically Formulated Evaluation**: Full metric evaluation using true pixel-weighted aggregate PSNR (accumulated squared pixel error across all evaluated frames), per-frame PSNR, SSIM with circular Gaussian weighting (Wang et al. 2004), and Charbonnier loss.
- **Verified Benchmark Performance**: Evaluated on all 2,400 full-resolution ($720 \times 1280$) held-out frames after 1,000 steps, achieving **$100\%$ positive frame PSNR gain** ($+0.1034\text{ dB}$ aggregate gain, $+0.1263\text{ dB}$ mean gain).

---

## Directory Structure

```text
Video_Deblurring/
├── configs/                # YAML configuration files (EDVR, dataset, EDA)
├── data/
│   ├── raw/                # REDS raw dataset (train_blur, train_sharp, val_blur, val_sharp, test_blur)
│   └── manifests/          # Canonical sequence split manifests and summaries
├── docs/                   # Documentation and architectural guides
├── experiments/            # Full-training run outputs, logs, and checkpoints
├── notebooks/              # Jupyter notebooks for visual inspection and EDA
├── reports/                # Structured research reports
│   ├── eda/                # Initial dataset analysis reports
│   ├── data_pipeline/      # Pipeline and architecture audit reports
│   ├── validation/         # Pilot metrics, readiness gates, and memory benchmarks
│   └── evaluation/         # Held-out benchmark metrics, CSVs, and visual comparisons
├── splits/                 # Sequence-level split manifest mirrors (sequence_split.csv)
├── src/                    # Modular source code
│   ├── data/               # Dataset indexing, discovery, and PyTorch dataset loader
│   ├── evaluation/         # Metric definitions (PSNR, SSIM) and held-out evaluation runner
│   ├── models/             # EDVR architecture, DCNv2, PreDeblur, TSA, and adapter
│   ├── preprocessing/      # Multi-frame temporal windows and paired spatial transforms
│   ├── training/           # Controlled pilot, full training harness, and AMP validation
│   └── utils/              # Project path and configuration utilities
└── tests/                  # Complete pytest suite (62 automated unit/integration tests)
```

---

## Environment Setup

### 1. Requirements
- Python 3.10+ (tested on Python 3.12.10)
- NVIDIA GPU with CUDA 12.8 support (e.g. RTX 5060, RTX 40-series, RTX 30-series)
- PyTorch 2.0+ with CUDA acceleration

### 2. Installation
```powershell
# Create virtual environment
python -m venv .venv_torch
.venv_torch\Scripts\activate

# Install PyTorch with CUDA 12.8 support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install dependencies
pip install -r requirements.txt
```

---

## Quickstart Guide

### 1. Run Automated Test Suite
Run the 62-test regression suite covering model construction, DCNv2 numerical equivalence, TSA fusion, gradient unscaling, checkpoint resumption, split integrity, and evaluation metrics:
```powershell
.venv_torch\Scripts\python.exe -m pytest tests/ -v
```

### 2. Run Controlled Training Pilot (100 Steps)
Execute a rapid hardware smoke test verifying forward pass, backward pass, AMP scaling, and checkpoint saving/resumption:
```powershell
.venv_torch\Scripts\python.exe -m src.training.train --mode pilot --num_pilot_steps 100 --pilot_val_freq 25
```

### 3. Launch Full Baseline Training
Launch configurable baseline training on the 216 REDS training sequences:
```powershell
.venv_torch\Scripts\python.exe -m src.training.train --mode train --total_steps 50000 --val_freq 2500 --ckpt_freq 5000 --batch_size 2 --lr 0.0002
```

To resume training from an existing checkpoint:
```powershell
.venv_torch\Scripts\python.exe -m src.training.train --mode resume --resume_from experiments/edvr_run_<timestamp>/checkpoints/latest_checkpoint.pth --total_steps 50000
```

### 4. Evaluate Trained Checkpoint on Held-Out Sequences
Run full-frame $720 \times 1280$ evaluation across all 24 unseen held-out test sequences (2,400 frames), generating per-frame metrics and 4-panel visual comparisons:
```powershell
.venv_torch\Scripts\python.exe -m src.evaluation.evaluate_held_out --checkpoint experiments/edvr_run_<timestamp>/checkpoints/best_checkpoint.pth --visual_interval 25
```

---

## Benchmark Results on Held-Out REDS Test Sequences

Evaluated on all 24 held-out test sequences (2,400 full $720 \times 1280$ frames) from the verified `best_checkpoint.pth` (1,000 steps):

| Metric | Blurry Input Baseline | EDVR-M Restored (1,000 steps) | Absolute Gain |
| :--- | :---: | :---: | :---: |
| **Pixel-Weighted Aggregate PSNR** | 22.8853 dB | **22.9887 dB** | **+0.1034 dB** |
| **Mean Per-Frame PSNR** | 23.9030 dB | **24.0293 dB** | **+0.1263 dB** |
| **Mean Per-Frame SSIM** | 0.7199 | **0.7208** | **+0.0009** |
| **Mean Charbonnier Loss** | 0.038291 | **0.037867** | **-0.000424** |
| **Positive Frame PSNR Ratio** | — | **2,400 / 2,400 (100.0%)** | Full partition improved |
| **Inference Speed (720x1280)** | — | **2.05 fps** | 1.48 GB VRAM consumed |

Detailed results and per-sequence tables are available in [`reports/evaluation/edvr_held_out_evaluation_report.md`](reports/evaluation/edvr_held_out_evaluation_report.md).
