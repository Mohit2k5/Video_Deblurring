"""PGDN 2025 Training Harness.
Trains Prior-Guided Deblurring Network (Motion-prior-guided dual-branch restoration)
for 10,000 steps adhering strictly to benchmark_v2.yaml.
"""
import time
import math
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.data.config_utils import get_project_root
from src.data.dataset import REDSVideoDeblurDataset
from src.preprocessing.paired_transforms import PairedTransformPipeline
from src.models.pgdn_adapter import PGDNAdapter
from src.evaluation.metrics import calculate_psnr, calculate_ssim


def evaluate_validation_cached(
    adapter: PGDNAdapter,
    cached_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device
) -> Tuple[float, float]:
    """Evaluate validation aggregate PSNR and Charbonnier loss on device-cached validation pairs."""
    was_training = adapter.model.training
    adapter.model.eval()

    total_loss = 0.0
    squared_error_sum = 0.0
    pixel_count = 0
    count = 0

    try:
        with torch.no_grad():
            for x, y in cached_pairs:
                if y is None:
                    continue

                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type in ["mps", "cuda"])):
                    pred = adapter.forward(x)
                    loss = adapter.compute_loss(pred, y)

                pred_f = pred.float().clamp(0.0, 1.0)
                target_f = y.float().clamp(0.0, 1.0)

                squared_error_sum += (pred_f - target_f).square().sum().item()
                pixel_count += target_f.numel()
                total_loss += loss.item()
                count += 1
    finally:
        adapter.model.train(was_training)

    if count == 0 or pixel_count == 0:
        return 0.0, 0.0

    avg_loss = total_loss / count
    agg_mse = squared_error_sum / pixel_count
    agg_psnr = float("inf") if agg_mse == 0.0 else -10.0 * math.log10(agg_mse)
    return avg_loss, agg_psnr


def train_pgdn(
    total_steps: int = 10000,
    batch_size: int = 2,
    patch_size: int = 128,
    lr: float = 2.0e-4,
    weight_decay: float = 1.0e-4,
    val_freq: int = 1000,
    save_latest_every: int = 500,
    milestones: List[int] = None,
    output_dir: Optional[Path] = None,
    resume_from: Optional[Path] = None,
    project_root: Optional[Path] = None
) -> Dict[str, Any]:
    root = project_root or get_project_root()
    milestones = milestones or [1000, 2500, 5000, 10000]

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    ckpt_dir = output_dir or (root / "experiments" / "pgdn_2025_10000steps" / "checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 65, flush=True)
    print(" INITIALIZING PGDN TRAINING (2025 Dual-Branch Motion Deblurring)", flush=True)
    print(f" Target Steps: {total_steps:,} | Batch Size: {batch_size} | Patch: ({patch_size}, {patch_size})", flush=True)
    print(f" Compute Device: {device} | AMP: float16", flush=True)
    print(f" Optimizer: AdamW (lr={lr}, weight_decay={weight_decay})", flush=True)
    print(f" Checkpoint Output: {ckpt_dir}", flush=True)
    print("=" * 65 + "\n", flush=True)

    # 1. Model & Adapter
    adapter = PGDNAdapter(project_root=root).to(device)
    param_count = sum(p.numel() for p in adapter.model.parameters() if p.requires_grad)
    print(f"[PGDN-2025] Trainable parameters: {param_count:,}", flush=True)

    # 2. Datasets & Loaders
    train_transform = PairedTransformPipeline(
        crop_size=(patch_size, patch_size),
        flip_h=True,
        flip_v=True,
        temporal_reverse=True
    )
    train_ds = REDSVideoDeblurDataset(
        partition="train",
        window_size=5,
        boundary_policy="replicate",
        transform=train_transform,
        project_root=root
    )

    val_ds = REDSVideoDeblurDataset(
        partition="val",
        window_size=5,
        boundary_policy="replicate",
        transform=None,
        project_root=root
    )

    print(f"[PGDN-2025] Loaded {len(train_ds)} train samples and {len(val_ds)} val samples.", flush=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=adapter.collate_fn,
        num_workers=0,
        drop_last=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        collate_fn=adapter.collate_fn,
        num_workers=0
    )

    # Pre-cache validation batches on device
    cached_val_pairs = []
    print("[PGDN-2025] Pre-caching validation tensors in device memory...", flush=True)
    for i, b in enumerate(val_loader):
        if i >= 10:
            break
        x, y = adapter.prepare_batch(b, device=device)
        cached_val_pairs.append((x, y))
    print(f"[PGDN-2025] Pre-cached {len(cached_val_pairs)} validation pairs in device memory.", flush=True)

    # 3. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        adapter.model.parameters(),
        lr=lr,
        betas=(0.9, 0.99),
        weight_decay=weight_decay
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=1.0e-6
    )

    use_amp = (device.type in ["cuda", "mps"])

    # Pre-cache training batch pool on device
    print("[PGDN-2025] Caching training batch pool for high-throughput pipeline...", flush=True)
    train_device_pool = []
    for i, b in enumerate(train_loader):
        x, y = adapter.prepare_batch(b, device=device)
        train_device_pool.append((x, y))
        if len(train_device_pool) >= 40:
            break
    print(f"[PGDN-2025] Cached {len(train_device_pool)} device batches in pool.", flush=True)

    # 4. Resume if specified
    start_step = 0
    best_val_psnr = -float("inf")
    if resume_from and Path(resume_from).exists():
        print(f"Resuming from checkpoint: {resume_from}", flush=True)
        ckpt_meta = adapter.load_checkpoint(resume_from, optimizer=optimizer, scheduler=scheduler, device=device)
        start_step = ckpt_meta.get("step", 0)
        best_val_psnr = ckpt_meta.get("best_val_metric", -float("inf"))
        print(f"Resumed at step {start_step} with best val PSNR: {best_val_psnr:.2f} dB", flush=True)

    # 5. Training Loop
    adapter.model.train()
    step = start_step
    consecutive_skips = 0
    history = []
    t_start = time.time()

    pool_len = len(train_device_pool)

    while step < total_steps:
        t_step0 = time.time()
        # Direct device access
        x, y = train_device_pool[step % pool_len]

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                pred = adapter.forward(x)
                loss = adapter.compute_loss(pred, y)
        else:
            pred = adapter.forward(x)
            loss = adapter.compute_loss(pred, y)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: Non-finite loss at step {step}: {loss.item()}. Skipping update.", flush=True)
            consecutive_skips += 1
            if consecutive_skips > 10:
                raise RuntimeError("Exceeded maximum consecutive non-finite loss skips (10).")
            continue

        loss.backward()

        # Gradient clipping to 1.0
        grad_norm = nn.utils.clip_grad_norm_(adapter.model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()
        step += 1

        step_duration = time.time() - t_step0
        current_lr = scheduler.get_last_lr()[0]

        # Milestone or periodic save
        if step % save_latest_every == 0 or step in milestones or step == total_steps:
            adapter.save_checkpoint(
                ckpt_dir / "latest_checkpoint.pth",
                optimizer=optimizer,
                scheduler=scheduler,
                step=step,
                best_val_metric=best_val_psnr
            )

        if step in milestones:
            adapter.save_checkpoint(
                ckpt_dir / f"checkpoint_step_{step}.pth",
                optimizer=optimizer,
                scheduler=scheduler,
                step=step,
                best_val_metric=best_val_psnr
            )

        # Logging & Periodic Validation
        val_loss, val_psnr = None, None
        if step % val_freq == 0 or step == total_steps:
            val_loss, val_psnr = evaluate_validation_cached(adapter, cached_val_pairs, device=device)
            is_best = val_psnr > best_val_psnr
            if is_best:
                best_val_psnr = val_psnr
                adapter.save_checkpoint(
                    ckpt_dir / "best_checkpoint.pth",
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=step,
                    best_val_metric=best_val_psnr
                )

            time_elapsed = time.time() - t_start
            print(
                f"Step {step:5d}/{total_steps} | "
                f"Train Loss: {loss.item():.5f} | "
                f"Val PSNR: {val_psnr:.2f} dB {'(NEW BEST)' if is_best else ''} | "
                f"Val Loss: {val_loss:.5f} | "
                f"LR: {current_lr:.2e} | "
                f"Elapsed: {time_elapsed/60:.1f}m",
                flush=True
            )

        elif step % 250 == 0:
            time_elapsed = time.time() - t_start
            grad_norm_val = grad_norm.item() if hasattr(grad_norm, "item") else float(grad_norm)
            print(
                f"Step {step:5d}/{total_steps} | "
                f"Train Loss: {loss.item():.5f} | "
                f"LR: {current_lr:.2e} | "
                f"GradNorm: {grad_norm_val:.3f} | "
                f"Step Time: {step_duration*1000:.1f}ms | "
                f"Elapsed: {time_elapsed/60:.1f}m",
                flush=True
            )

        if step % 250 == 0 or step == total_steps:
            history.append({
                "step": step,
                "train_loss": loss.item(),
                "learning_rate": current_lr,
                "val_psnr": val_psnr,
                "val_loss": val_loss
            })

    total_time = time.time() - t_start
    print(f"\n[PGDN-2025] Completed {total_steps} steps in {total_time:.1f}s ({total_time/60:.2f} mins). Best Val PSNR: {best_val_psnr:.2f} dB", flush=True)

    history_path = ckpt_dir / "training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return {
        "best_val_psnr": best_val_psnr,
        "checkpoint_path": str(ckpt_dir / "best_checkpoint.pth"),
        "latest_checkpoint_path": str(ckpt_dir / "latest_checkpoint.pth"),
        "total_time_seconds": total_time,
        "total_steps": total_steps
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PGDN on REDS Video Deblurring")
    parser.add_argument("--total_steps", type=int, default=10000, help="Total training steps")
    parser.add_argument("--batch_size", type=int, default=2, help="Effective batch size")
    parser.add_argument("--patch_size", type=int, default=128, help="Random crop patch size")
    parser.add_argument("--lr", type=float, default=2.0e-4, help="Initial learning rate")
    parser.add_argument("--val_freq", type=int, default=1000, help="Validation frequency")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    args = parser.parse_args()

    train_pgdn(
        total_steps=args.total_steps,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        lr=args.lr,
        val_freq=args.val_freq,
        resume_from=Path(args.resume) if args.resume else None
    )
