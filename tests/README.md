# Automated Test Suite (`tests/`)

This directory houses the comprehensive regression and unit test suite for the `reds-video-deblurring` project.

---

## Test Coverage (62 Total Tests)

All tests are automated and execute via `pytest`. Current status: **62 passed (100% pass rate, 0 failed, 0 skipped, 0 warnings)**.

### 1. `tests/test_edvr.py` (22 Tests)
- Architecture construction from `configs/edvr.yaml` with PreDeblur and without PreDeblur.
- Parameter count regression check (asserting exactly 4,042,851 parameters for EDVR-M).
- Forward pass tensor shapes for batch sizes 1 and 2.
- Input/output channel and spatial dimension preservation.
- Center-frame selection ($t=2$) and additive residual connection.
- Backward pass finite gradient propagation across all trainable parameters.
- PCD multi-level pyramid alignment tensor shapes.
- TSA temporal attention and multi-scale spatial attention numerical stability.
- Paired real-data loading integration with actual REDS samples.
- Checkpoint saving, architecture signature mismatch rejection, and reload.
- AMP execution under `torch.amp.autocast`.
- Mathematical equivalence of `DCNv2Pack` with zero offsets to $0.5 \times \text{Conv2d}$.
- Unscaled AMP gradient validation and NaN/Inf gradient detection.
- Exact PSNR behavior returning `float("inf")` on zero MSE.
- Validation mode preservation (train and eval mode immutability, exception safety).
- Empty/invalid validation DataLoader rejection.
- Separate synthetic checkpoint unit test (Test A) and real REDS pilot checkpoint resumption test with genuine optimizer step (Test B).
- Aggregate pixel MSE PSNR vs. batch-mean PSNR mathematical properties.

### 2. `tests/test_evaluation.py` (8 Tests)
- PSNR mathematical formulation on identical tensors and known values.
- SSIM calculation (identical images return 1.0; random images bound in $[-1, 1]$).
- Charbonnier loss calculation.
- AggregateMetricsTracker mathematical aggregation vs mean averaging.
- Held-out test split verification (24 sequences, zero leakage).
- Checkpoint loading verification.

### 3. `tests/test_shared_pipeline.py` (11 Tests)
- Sequence split count generation (90:10 ratio).
- Split reproducibility across identical seeds.
- Zero sequence overlap between training and held-out partitions.
- Temporal window boundary guarantees (`replicate`, `truncate`, `zero_pad`).
- Paired random crop spatial alignment across all frames.
- Paired geometric flips preserving pixel-to-pixel correspondences.
- Paired transforms reproducibility.
- Synthetic dataset loader indexing.
- Fixed REDS split integrity with composite-key mutual exclusivity across all 4 partitions.

### 4. `tests/test_pairing_and_validation.py` (15 Tests)
- Correctly matched blur-sharp frame pairing.
- Missing frame handling and detection.
- Unequal frame counts and sequence missing detection.
- Image dimension mismatch detection.
- Corrupted image handling.
- Config validation and consecutive frame selection.

### 5. `tests/test_data.py` (2 Tests)
- Empty dataset discovery handling.
- Sequence analysis and metadata extraction.

---

## Running the Tests

```powershell
# Run all 62 tests with verbose output
.venv_torch\Scripts\python.exe -m pytest tests/ -v

# Run only model and training tests
.venv_torch\Scripts\python.exe -m pytest tests/test_edvr.py -v

# Run only evaluation metric tests
.venv_torch\Scripts\python.exe -m pytest tests/test_evaluation.py -v
```
