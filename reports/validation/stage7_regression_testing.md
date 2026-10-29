# Stage 7 Regression Testing Report

## 1. Test Suite Execution Summary
- **Execution Command**: `.venv_torch\Scripts\pytest tests/ -v`
- **Total Tests Collected**: **42**
- **Passed Tests**: **42**
- **Failed Tests**: **0**
- **Warnings**: **0**
- **Execution Duration**: **18.71 seconds**
- **Test Framework**: `pytest 9.1.1` on `Python 3.12.10` (Windows 11)

---

## 2. Test Coverage & Module Summary
| Test Module | Tests Count | Status | Key Coverage Areas |
| :--- | :---: | :---: | :--- |
| `tests/test_edvr.py` | 15 | **PASSED** | Model construction from YAML, parameter count regression ($4,042,851$ for EDVR-M, $23,602,019$ for EDVR-L), forward pass $B=1$ and $B=2$, resolution transitions, center-frame residual skip, finite backward gradients, 3-level PCD shapes, TSA attention numerical stability, real REDS sample integration, checkpoint save/restore, AMP execution, adapter-model config agreement, DCNv2 zero-offset convolution equivalence, and checkpoint architecture mismatch rejection. |
| `tests/test_shared_pipeline.py` | 9 | **PASSED** | Deterministic 90:10 split generation, split reproducibility, zero partition leakage, temporal window boundary replicate & truncate policies, paired crop coordinate-coded alignment, geometric flips, transform reproducibility, and synthetic dataset loading. |
| `tests/test_pairing_and_validation.py` | 16 | **PASSED** | Paired frame matching, missing frame detection, dimension mismatch detection, corrupted file detection, empty sequence handling, reproducible sampling, issue logging, consecutive frame sequencing, and config validation. |
| `tests/test_data.py` | 2 | **PASSED** | Empty dataset discovery and sequence inventory indexing. |
| **Total** | **42** | **100% PASS** | |

---

## 3. Strengthened Behavioral Tests
1. **`test_dcnv2_zero_offset_convolution_equivalence`**:
   - Numerically validates that when `conv_offset` weights and biases are zero, `DCNv2Pack` produces output identical to `0.5 * Conv2d` with max absolute difference $< 10^{-6}$ (actual: `0.00000000e+00`).
2. **`test_checkpoint_architecture_mismatch_rejection`**:
   - Asserts that attempting to load a checkpoint with mismatched hyperparameters (e.g. `num_feat=128` into a model with `num_feat=64`) strictly raises a descriptive `ValueError`.
3. **`test_parameter_count_regression`**:
   - Enforces exact active parameter count ($4,042,851$) and individual submodule totals.

---

## 4. Stage 7 Gate Assessment
- **Status**: **PASSED**.
- All 42 unit, integration, and regression tests pass with zero warnings.
- Proceeding to Stage 8 (Controlled Training Pilot) is safe.
