# Research Reports & Metrics (`reports/`)

This directory houses all experimental reports, verification artifacts, and evaluation benchmarks organized chronologically and functionally.

---

## Directory Structure

### 1. `reports/evaluation/` (Held-Out Benchmark Evaluation)
- **`edvr_held_out_evaluation_report.md`**:
  Comprehensive benchmark report evaluating the 1,000-step EDVR-M baseline across all 24 held-out REDS test sequences (2,400 frames) at full $720 \times 1280$ resolution.
- **`held_out_summary_metrics.json`**:
  Dataset-wide summary containing aggregate PSNR ($22.9887\text{ dB}$, $+0.1034\text{ dB}$ gain), mean PSNR ($24.0293\text{ dB}$), mean SSIM ($0.7208$), and Charbonnier loss.
- **`held_out_per_sequence_metrics.csv` / `.json`**:
  Per-sequence metrics breakdown across all 24 sequences.
- **`held_out_per_frame_metrics.csv`**:
  Detailed per-frame table (2,400 rows) documenting blurry vs restored metrics.
- **`visuals/`**:
  96 4-panel visual comparison figures (Blurry Input, EDVR Restored, Sharp Ground Truth, Absolute Error $\times 5$) and 3 side-by-side MP4 verification clips.

### 2. `reports/validation/` (Training Verification & Pilot Gates)
- **`final_edvr_readiness_report.md`**:
  Final readiness gate report certifying that EDVR-M passed all correctness checks for full baseline training.
- **`final_controlled_pilot_metrics.json`**:
  Metrics from the 100-step controlled pilot verifying AMP unscaled gradient validation and checkpoint resumption.
- **`stage8_controlled_pilot_metrics.json` / `.md`**:
  Historical Stage 8 pilot baseline metrics.
- **`stage9_readiness_and_continuation_report.md`**:
  Continuation audit resolving parameter counts, vacuous assertions, and memory benchmarking.
- **`pilot_checkpoints/`**:
  Saved step-50 pilot checkpoint (`pilot_checkpoint_step50.pth`).

### 3. `reports/data_pipeline/` (Pipeline & Architecture Audits)
- **`edvr_architecture_and_validation_report.md`**:
  Deep architectural comparison with upstream BasicSR EDVR, parameter count reconciliation (4,042,851 params), and GPU memory profile.
- **`README.md`**:
  Data pipeline specifications and test documentation.

### 4. `reports/eda/` (Exploratory Data Analysis)
- Dataset inventory CSVs, file extension distributions, sequence statistics, and frame pairing verification summaries.
