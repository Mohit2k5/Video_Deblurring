# Stage 4 Configuration, AMP, and Checkpoint Compatibility Report

## 1. Configuration Audit (`configs/edvr.yaml`)
- **Temporal Window Length**: `num_frame = 5`, matched to pipeline window size $T=5$.
- **Center Frame Index**: `center_frame_idx = 2` ($0$-indexed frame 2 of 5), matched to temporal index $T // 2$.
- **Channels**: `num_in_ch = 3`, `num_out_ch = 3` (RGB).
- **Active Feature Channels**: `num_feat = 64`.
- **Active Residual Blocks**: `num_extract_block = 5`, `num_reconstruct_block = 10`.
- **PreDeblur & TSA**: Both explicitly enabled (`with_predeblur: true`, `with_tsa: true`, `hr_in: true`).
- **Deformable Groups**: `deformable_groups: 8`.
- **Loss**: CharbonnierLoss ($\epsilon = 1.0 \times 10^{-12}$).
- **Optimizer**: Adam ($\text{lr} = 2.0 \times 10^{-4}$, $\beta_1 = 0.9, \beta_2 = 0.99$, weight decay = 0.0).
- **Learning Rate Scheduler**: `CosineAnnealingLR` ($T_{\max} = 600,000, \eta_{\min} = 1.0 \times 10^{-7}$).
- **Precision**: PyTorch AMP enabled (`use_amp: true`).

---

## 2. Checkpoint Save, Restore, and Resumption Verification
- **Saved State Components**:
  - `model_state_dict`: Full model weights.
  - `optimizer_state_dict`: Adam momentum and variance buffers.
  - `scheduler_state_dict`: Learning rate schedule step and state.
  - `scaler_state_dict`: `torch.amp.GradScaler` scale and growth tracker.
  - `step`: Current optimization step.
  - `epoch`: Current training epoch.
  - `arch_signature`: Verification dictionary (`num_feat`, `num_frame`, `num_extract_block`, `num_reconstruct_block`, `center_frame_idx`, `hr_in`, `with_predeblur`, `with_tsa`, `deformable_groups`).
- **Incompatible Signature Rejection**:
  - Verified that loading a checkpoint with mismatched hyperparameters (e.g. `num_feat: 128` into a model with `num_feat: 64`) immediately raises a descriptive `ValueError`:
    `Incompatible checkpoint architecture parameters: num_feat (current: 64, saved: 128)`
- **Resumption Test**:
  - Verified that training state resumes seamlessly with exact parameter and scheduler state equality.

---

## 3. Stage 4 Gate Assessment
- **Status**: **PASSED**.
- Configuration, AMP, and checkpoint compatibility are verified and regression-tested.
- Proceeding to Stage 5 is safe.
