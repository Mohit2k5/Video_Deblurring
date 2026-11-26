"""Training script for MSVSSM (Motion-aware Visual State Space Model).
Supports training MSVSSM-Base, MSVSSM-FFT, and MSVSSM-TSFF for 1000 steps on REDS dataset.
"""
import time
import math
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.data.config_utils import get_project_root
from src.data.dataset import REDSVideoDeblurDataset
from src.preprocessing.paired_transforms import PairedTransformPipeline
from src.models.msvssm_adapter import MSVSSMAdapter
from src.evaluation.metrics import calculate_psnr, calculate_ssim


def evaluate_validation_cached(
    adapter: MSVSSMAdapter,
    cached_batches: List[Any],
    device: torch.device
) -> Tuple[float, float]:
    """Evaluate validation loss and aggregate PSNR on cached validation samples."""
    was_training = adapter.model.training
    adapter.model.eval()

    total_loss = 0.0
    squared_error_sum = 0.0
    pixel_count = 0
    count = 0

    try:
        with torch.no_grad():
            for batch in cached_batches:
                x, y = adapter.prepare_batch(batch, device=device)
                if y is None:
                    continue

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


def train_msvssm(
    algo: str = "base",
    total_steps: int = 1000,
    val_freq: int = 250,
    batch_size: int = 2,
    lr: float = 2e-4,
    crop_size: Tuple[int, int] = (192, 192),
    save_dir: Optional[Path] = None,
    project_root: Optional[Path] = None
) -> Dict[str, Any]:
    """Train MSVSSM model for specified steps."""
    root = project_root or get_project_root()
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    algo_name = f"MSVSSM-{algo.upper()}"
    run_dir = save_dir or (root / "experiments" / f"msvssm_{algo}_1000steps")
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n========================================================", flush=True)
    print(f" Initializing Training: {algo_name}", flush=True)
    print(f" Target Steps: {total_steps} | Batch Size: {batch_size} | Crop: {crop_size}", flush=True)
    print(f" Compute Device: {device}", flush=True)
    print(f" Checkpoint Output: {ckpt_dir}", flush=True)
    print(f"========================================================\n", flush=True)

    # 1. Initialize Model Adapter
    adapter = MSVSSMAdapter(algo=algo, project_root=root).to(device)
    param_count = sum(p.numel() for p in adapter.model.parameters() if p.requires_grad)
    print(f"[{algo_name}] Trainable parameters: {param_count:,}", flush=True)

    # 2. Setup DataLoaders
    train_transform = PairedTransformPipeline(
        crop_size=crop_size,
        flip_h=True,
        flip_v=True,
        temporal_reverse=True
    )
    val_transform = PairedTransformPipeline(
        crop_size=(256, 256),
        flip_h=False,
        flip_v=False,
        temporal_reverse=False
    )

    train_ds = REDSVideoDeblurDataset(partition="train", transform=train_transform, project_root=root)
    val_ds = REDSVideoDeblurDataset(partition="val", transform=val_transform, project_root=root)

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

    # Pre-cache 4 validation batches in memory for zero-overhead validation checks
    val_it = iter(val_loader)
    cached_val_batches = [next(val_it) for _ in range(4)]
    print(f"[{algo_name}] Pre-cached 4 validation batches in memory.", flush=True)

    # 3. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        adapter.model.parameters(),
        lr=lr,
        betas=(0.9, 0.99),
        weight_decay=1e-4
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    # 4. Training Loop
    adapter.model.train()
    step = 0
    best_val_psnr = -1.0
    history = []
    start_time = time.time()
    train_iter = iter(train_loader)

    while step < total_steps:
        step += 1
        step_t0 = time.time()

        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        x, y = adapter.prepare_batch(batch, device=device)
        optimizer.zero_grad()

        pred = adapter.forward(x)
        loss = adapter.compute_loss(pred, y)
        loss.backward()

        grad_norm = nn.utils.clip_grad_norm_(adapter.model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        step_duration = time.time() - step_t0
        curr_lr = optimizer.param_groups[0]["lr"]

        # Logging & Periodic Validation
        val_loss, val_psnr = None, None
        if step % val_freq == 0 or step == total_steps:
            val_loss, val_psnr = evaluate_validation_cached(adapter, cached_val_batches, device=device)
            print(f"Step {step:4d}/{total_steps} | Loss: {loss.item():.5f} | Val PSNR: {val_psnr:.2f} dB | Val Loss: {val_loss:.5f} | Step Time: {step_duration:.2f}s", flush=True)

            if val_psnr > best_val_psnr:
                best_val_psnr = val_psnr
                best_ckpt_path = ckpt_dir / "best_checkpoint.pth"
                adapter.save_checkpoint(
                    best_ckpt_path,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=step,
                    extra_meta={"best_val_psnr": best_val_psnr, "algo": algo}
                )
        elif step % 100 == 0 or step == 1:
            print(f"Step {step:4d}/{total_steps} | Train Loss: {loss.item():.5f} | LR: {curr_lr:.2e} | GradNorm: {grad_norm.item():.3f} | Step Time: {step_duration:.2f}s", flush=True)

        history.append({
            "step": step,
            "train_loss": round(loss.item(), 6),
            "val_loss": round(val_loss, 6) if val_loss is not None else None,
            "val_psnr": round(val_psnr, 2) if val_psnr is not None else None,
            "step_time_sec": round(step_duration, 4)
        })

    # Save final checkpoint
    final_ckpt = ckpt_dir / "latest_checkpoint.pth"
    adapter.save_checkpoint(
        final_ckpt,
        optimizer=optimizer,
        scheduler=scheduler,
        step=step,
        extra_meta={"best_val_psnr": best_val_psnr, "algo": algo}
    )

    # Ensure best checkpoint exists
    best_ckpt = ckpt_dir / "best_checkpoint.pth"
    if not best_ckpt.exists():
        adapter.save_checkpoint(best_ckpt, optimizer=optimizer, scheduler=scheduler, step=step, extra_meta={"best_val_psnr": best_val_psnr, "algo": algo})

    total_time = time.time() - start_time
    print(f"\n[{algo_name}] Completed {total_steps} steps in {total_time:.1f}s ({total_time / 60:.2f} mins). Best Val PSNR: {best_val_psnr:.2f} dB\n", flush=True)

    summary = {
        "algo": algo,
        "algo_name": algo_name,
        "total_steps": total_steps,
        "total_time_sec": round(total_time, 2),
        "best_val_psnr_db": round(best_val_psnr, 2),
        "checkpoint_path": str(best_ckpt),
        "history": history
    }
    with open(run_dir / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Train MSVSSM for Video Deblurring")
    parser.add_argument("--algo", type=str, choices=["base", "fft", "tsff"], default="base")
    parser.add_argument("--total_steps", type=int, default=1000)
    parser.add_argument("--val_freq", type=int, default=250)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_msvssm(
        algo=args.algo,
        total_steps=args.total_steps,
        val_freq=args.val_freq,
        batch_size=args.batch_size,
        lr=args.lr
    )
