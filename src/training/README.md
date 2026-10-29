# Training Pipeline & Harness (`src/training/`)

This directory contains the training engine, validation evaluation loop, automatic mixed precision (AMP) gradient validation, and checkpoint management for EDVR.

---

## Key Modules & Functions

### `src/training/train.py`

#### 1. `evaluate_validation()`
- Evaluates the model on validation sequences.
- **Mode Preservation**: Safely captures `was_training = model.training`, runs evaluation in a `try...finally: model.train(was_training)` block, guaranteeing that evaluation mode is never corrupted.
- **Empty-Batch Rejection**: Raises `RuntimeError` if an empty or un-paired batch occurs, preventing silent zero-metric reporting.
- **Aggregate Pixel PSNR**: Accumulates squared Euclidean pixel error across all batches:
  $$\text{MSE}_{\text{agg}} = \frac{\sum (pred - gt)^2}{N_{\text{pixels}}},\quad \text{PSNR}_{\text{agg}} = -10 \log_{10}(\text{MSE}_{\text{agg}})$$

#### 2. `run_training_pilot()`
- Controlled 100-step technical smoke test.
- Profiles peak allocated and reserved VRAM with CUDA synchronization.
- Saves midpoint checkpoint at step 50 and executes a genuine post-resume optimizer step (step 51) verifying state restoration and disk file immutability.
- Saves run logs to `reports/validation/final_controlled_pilot_metrics.json`.

#### 3. `run_full_training()`
- Configurable baseline training workflow:
  - Supports arbitrary total steps (e.g. 50,000 steps).
  - Validation cadence and periodic checkpoint saving.
  - Multi-tier checkpoint policy:
    - `latest_checkpoint.pth`: For resuming interrupted runs.
    - `best_checkpoint.pth`: Preserves the best model according to validation aggregate PSNR.
    - `final_checkpoint.pth`: Final completed run state.
    - `checkpoint_step_{step}.pth`: Periodic snapshots.
  - Logs training history and metrics to `experiments/edvr_run_<timestamp>/training_summary.json`.

#### 4. AMP Gradient Validation
- Enforces strict unscaling order:
  1. `scaler.scale(loss).backward()`
  2. `scaler.unscale_(optimizer)`
  3. Strict finiteness check across all parameter gradients.
  4. Unscaled `torch.nn.utils.clip_grad_norm_`.
  5. `scaler.step(optimizer)`, `scaler.update()`, `scheduler.step()`.

---

## CLI Usage

```powershell
# Run 100-step pilot smoke test
python -m src.training.train --mode pilot --num_pilot_steps 100 --pilot_val_freq 25

# Run full baseline training (50,000 steps)
python -m src.training.train --mode train --total_steps 50000 --val_freq 2500 --ckpt_freq 5000 --batch_size 2 --lr 2e-4

# Resume from existing checkpoint
python -m src.training.train --mode resume --resume_from experiments/edvr_run_20261016_080811/checkpoints/latest_checkpoint.pth --total_steps 50000
```
