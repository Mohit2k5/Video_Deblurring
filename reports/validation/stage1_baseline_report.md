# Stage 1 Baseline Inspection Report: REDS Video Deblurring (EDVR)

## 1. Executive Summary & Environment
- **Project Directory**: `D:\reds-video-deblurring` (accessed via junction `D:\Video_Deblurring`)
- **Git Repository Status**: Not a git repository (project directory initialized as standalone research workspace). Active modifications tracked directly against project baseline.
- **Python Environment**: `Python 3.12.10`, executable: `D:\reds-video-deblurring\.venv_torch\Scripts\python.exe`
- **PyTorch Version**: `2.11.0+cu128`
- **Torchvision Version**: `0.26.0+cu128`
- **CUDA Device**: `NVIDIA GeForce RTX 5060 Laptop GPU`
- **Total Dedicated VRAM**: `8,150.56 MB`
- **CUDA Version**: `12.8`

---

## 2. Active Model Configuration & Parameter Baseline
- **Configuration File**: [`configs/edvr.yaml`](../../configs/edvr.yaml)
- **Active Model**: Hardware-adapted EDVR for REDS Video Deblurring (`num_feat: 64`, `num_extract_block: 5`, `num_reconstruct_block: 10`, `deformable_groups: 8`, `num_frame: 5`, `hr_in: true`, `with_predeblur: true`, `with_tsa: true`)
- **Total Parameters**: **4,042,851**
- **Trainable Parameters**: **4,042,851**

### Module-Level Parameter Breakdown
| Submodule | Parameter Count | Architectural Purpose |
| :--- | :--- | :--- |
| `predeblur` | 740,352 | 3-level pyramid with stride-2 downsampling ($x4$) + 5 residual blocks |
| `conv_1x1` | 4,160 | 1x1 feature bottleneck projection |
| `feature_extraction` | 369,280 | 5 ResidualBlockNoBN layers (64 channels) |
| `conv_l2_1`, `conv_l2_2` | 73,856 | 2-stage pyramid downsampling to Level 2 ($H/8 \times W/8$) |
| `conv_l3_1`, `conv_l3_2` | 73,856 | 2-stage pyramid downsampling to Level 3 ($H/16 \times W/16$) |
| `pcd_align` | 1,384,288 | 3-level PCD offset prediction, deformable convs, cascading alignment |
| `fusion` (TSAFusion) | 324,416 | Temporal correlation + multi-scale spatial pooling attention |
| `reconstruction` | 738,560 | 10 ResidualBlockNoBN layers at downsampled scale |
| `upconv1` | 147,712 | 3x3 conv expanding 64 to 256 channels for first PixelShuffle(2) |
| `upconv2` | 147,712 | 3x3 conv expanding 64 to 256 channels for second PixelShuffle(2) |
| `pixel_shuffle` | 0 | Parameterless spatial shuffle |
| `conv_hr` | 36,928 | Post-upsampling feature refinement |
| `conv_last` | 1,731 | Output projection to 3-channel RGB |
| **Total** | **4,042,851** | **Matches official BasicSR EDVR-M configuration exactly** |

---

## 3. Existing Test & Benchmark Baseline
- **Pytest Suite**: 39 tests passed in 10.27s (`tests/test_data.py`, `tests/test_pairing_and_validation.py`, `tests/test_shared_pipeline.py`, `tests/test_edvr.py`).
- **GPU Benchmarks Recorded**:
  - `gpu_smoke_test_report.json` (old smoke test model with 2,996,451 params): 2,694.5 MB peak VRAM on un-downsampled features.
  - `gpu_memory_validation_report.json` (corrected model with 4,042,851 params): 323.5 MB peak ($B=1$, AMP) and 644.1 MB peak ($B=2$, AMP).
- **Available Checkpoint**:
  - `reports/data_pipeline/checkpoints/overfit_checkpoint_step20.pth` (48.75 MB, step 20 of tiny-subset overfitting test).

---

## 4. Dataset Partition & Manifest Baseline
- **Canonical Manifest**: `data/manifests/sequence_split.csv` (300 sequences total: 216 train, 30 val, 24 held-out test, 30 inference test).
- **Mirrored Manifest**: `splits/sequence_split.csv` (100% hash parity with canonical manifest).
- **Partition Isolation**: Verified zero overlap between 216 training sequences and 24 held-out test sequences.

---

## 5. Stage 1 Gate Assessment
- **Status**: **PASSED**.
- All baseline parameters, configurations, test outputs, manifests, and file paths have been recorded and cross-verified.
- Proceeding to Stage 2 is safe.
