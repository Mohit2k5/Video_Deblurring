# Evaluation & Benchmarking (`src/evaluation/`)

This directory provides standardized evaluation metrics and the benchmark evaluation harness for measuring video deblurring performance on unseen REDS sequences.

---

## Key Modules

### 1. `src/evaluation/metrics.py`
Provides mathematically formulated quality assessment metrics for image and video deblurring:
- **`calculate_psnr(pred, gt)`**:
  Computes Peak Signal-to-Noise Ratio (dB) on tensors clamped to $[0.0, 1.0]$. Returns `float("inf")` for exact reconstruction ($\text{MSE} = 0.0$).
- **`calculate_ssim(pred, gt)`**:
  Implementation of the Structural Similarity Index Measure (Wang et al., IEEE TIP 2004). Uses an $11 \times 11$ circular Gaussian window ($\sigma = 1.5$) with $K_1 = 0.01, K_2 = 0.03$, computed per RGB channel and averaged over spatial and channel dimensions.
- **`calculate_charbonnier_loss(pred, gt)`**:
  Computes differentiable Charbonnier distance $\sqrt{(pred - gt)^2 + \epsilon^2}$.
- **`AggregateMetricsTracker`**:
  Accumulates squared Euclidean pixel error across all evaluated frames to compute true pixel-weighted aggregate PSNR:
  $$\text{MSE}_{\text{agg}} = \frac{\sum_{\text{frames}} \sum_{\text{pixels}} (pred - gt)^2}{N_{\text{total\_pixels}}},\quad \text{PSNR}_{\text{agg}} = -10 \log_{10}(\text{MSE}_{\text{agg}})$$

### 2. `src/evaluation/evaluate_held_out.py`
Automated benchmark evaluation harness:
- Loads verified checkpoints (`best_checkpoint.pth` or specified path).
- Evaluates the 24 held-out REDS sequences (2,400 frames total at full native $720 \times 1280$ resolution).
- Compares restored outputs against the blurry input baseline.
- Generates 4-panel visual comparisons (Blurry Input, EDVR Restored, Sharp Ground Truth, and Absolute Error $\times 5$) under `reports/evaluation/visuals/`.
- Renders side-by-side MP4 verification clips for temporal continuity inspection.
- Generates structured CSVs and JSONs (`held_out_per_frame_metrics.csv`, `held_out_per_sequence_metrics.csv`, `held_out_summary_metrics.json`).

---

## CLI Usage

```powershell
# Run full held-out evaluation with visual panels every 25 frames
python -m src.evaluation.evaluate_held_out --checkpoint experiments/edvr_run_20261016_080811/checkpoints/best_checkpoint.pth --visual_interval 25
```
