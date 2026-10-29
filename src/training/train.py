"""Controlled Training and Evaluation Pipeline for EDVR REDS Video Deblurring.
Supports batched training, periodic validation evaluation with aggregate pixel PSNR,
AMP mixed precision with unscaled gradient validation, learning rate scheduling,
memory profiling with standardized peak resets, verified checkpoint resumption,
and full-scale baseline training.
"""
import time
import math
import json
import hashlib
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.data.config_utils import get_project_root
from src.data.dataset import REDSVideoDeblurDataset
from src.preprocessing.paired_transforms import PairedTransformPipeline
from src.models.edvr_adapter import EDVRAdapter


def calculate_psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Calculate Peak Signal-to-Noise Ratio (PSNR) in decibels (dB) for a single tensor batch."""
    pred_c = pred.clamp(0.0, 1.0)
    gt_c = gt.clamp(0.0, 1.0)
    mse = torch.mean((pred_c - gt_c) ** 2).item()
    if mse == 0.0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def evaluate_validation(
    adapter: EDVRAdapter,
    val_loader: DataLoader,
    device: torch.device,
    use_amp: bool = True,
    max_val_batches: Optional[int] = 15
) -> Tuple[float, float, float]:
    """
    Evaluate model on validation partition.
    Preserves model's original training/evaluation mode across success and exceptions.
    
    Returns:
        avg_loss (float): Average Charbonnier validation loss.
        aggregate_psnr (float): PSNR computed from the aggregate pixel-level MSE across all evaluated pixels.
        batch_mean_psnr (float): Average of per-batch PSNR values (for backward compatibility).
    """
    was_training = adapter.model.training
    adapter.model.eval()

    total_loss = 0.0
    batch_psnr_sum = 0.0
    squared_error_sum = 0.0
    pixel_count = 0
    count = 0

    try:
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if max_val_batches is not None and i >= max_val_batches:
                    break
                x, y = adapter.prepare_batch(batch, device=device)
                if y is None:
                    continue

                with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                    pred = adapter.forward(x)
                    loss = adapter.compute_loss(pred, y)

                pred_f = pred.float().clamp(0.0, 1.0)
                target_f = y.float().clamp(0.0, 1.0)

                # Aggregate pixel MSE tracking
                squared_error_sum += (pred_f - target_f).square().sum().item()
                pixel_count += target_f.numel()

                # Batch loss & batch-mean PSNR
                total_loss += loss.item()
                batch_psnr_sum += calculate_psnr(pred_f, target_f)
                count += 1
    finally:
        adapter.model.train(was_training)

    if count == 0 or pixel_count == 0:
        raise RuntimeError(
            "Validation produced no valid batches or pixels. "
            "Check the validation dataset, DataLoader, and batch preparation."
        )

    avg_loss = total_loss / count
    aggregate_mse = squared_error_sum / pixel_count
    aggregate_psnr = (
        float("inf") if aggregate_mse == 0.0 else -10.0 * math.log10(aggregate_mse)
    )
    batch_mean_psnr = batch_psnr_sum / count

    return avg_loss, aggregate_psnr, batch_mean_psnr


def run_training_pilot(
    num_steps: int = 100,
    val_freq: int = 25,
    batch_size: int = 2,
    crop_size: Tuple[int, int] = (256, 256),
    lr: float = 2e-4,
    use_amp: bool = True,
    max_val_batches: Optional[int] = 15,
    save_dir: Optional[Path] = None,
    project_root: Optional[Path] = None
) -> Dict[str, Any]:
    """Execute a controlled training pilot on real REDS data with validation tracking and resumption verification."""
    root = project_root or get_project_root()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    save_dir = save_dir or (root / "reports" / "validation")
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Initializing EDVR Training Pilot on device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    # 1. Initialize adapter and model
    adapter = EDVRAdapter(project_root=root).to(device)
    adapter.model.train()

    # 2. Datasets & Loaders
    train_transform = PairedTransformPipeline(
        crop_size=crop_size,
        flip_h=True,
        flip_v=True,
        temporal_reverse=True,
        seed=42
    )
    val_transform = PairedTransformPipeline(
        crop_size=crop_size,
        flip_h=False,
        flip_v=False,
        temporal_reverse=False
    )

    train_ds = REDSVideoDeblurDataset(
        partition="train",
        project_root=root,
        window_size=5,
        boundary_policy="replicate",
        transform=train_transform
    )
    val_ds = REDSVideoDeblurDataset(
        partition="val",
        project_root=root,
        window_size=5,
        boundary_policy="replicate",
        transform=val_transform
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=adapter.collate_fn,
        num_workers=0
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=adapter.collate_fn,
        num_workers=0
    )

    # 3. Optimizer, Scheduler, Scaler
    optimizer = torch.optim.Adam(adapter.model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=0.0)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=1e-7)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")

    # Initial validation evaluation
    print("Running initial baseline validation evaluation...")
    init_val_loss, init_val_agg_psnr, init_val_batch_psnr = evaluate_validation(
        adapter, val_loader, device, use_amp=use_amp, max_val_batches=max_val_batches
    )
    print(f"Initial Validation: Loss = {init_val_loss:.6f}, Aggregate PSNR = {init_val_agg_psnr:.2f} dB (Batch Mean = {init_val_batch_psnr:.2f} dB)")

    history = []
    ckpt_dir = save_dir / "pilot_checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    mid_ckpt_path = ckpt_dir / f"pilot_checkpoint_step{num_steps // 2}.pth"

    step = 0
    train_iter = iter(train_loader)

    print(f"Starting {num_steps}-step controlled training pilot...")
    start_time = time.time()

    while step < num_steps:
        step += 1
        step_start = time.time()

        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        # Standardized GPU memory tracking: reset stats before measured workload
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        x, y = adapter.prepare_batch(batch, device=device)
        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
            pred = adapter.forward(x)
            loss = adapter.compute_loss(pred, y)

        scaler.scale(loss).backward()

        # Unscale gradients before checking finiteness and computing gradient norm
        if use_amp and device.type == "cuda":
            scaler.unscale_(optimizer)

        # Strict finite gradient validation
        gradients_finite = True
        for parameter in adapter.model.parameters():
            if parameter.grad is not None:
                if not torch.isfinite(parameter.grad).all().item():
                    gradients_finite = False
                    break

        if not gradients_finite:
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError(f"Non-finite gradients detected at step {step}")

        # Compute unscaled gradient norm
        grad_norm = torch.nn.utils.clip_grad_norm_(
            adapter.model.parameters(),
            max_norm=float("inf")
        )

        if not torch.isfinite(grad_norm).item():
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError(f"Non-finite gradient norm detected at step {step}")

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if device.type == "cuda":
            torch.cuda.synchronize()

        step_duration = time.time() - step_start
        curr_lr = optimizer.param_groups[0]["lr"]

        peak_alloc_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if device.type == "cuda" else 0.0
        peak_res_mb = torch.cuda.max_memory_reserved() / (1024 ** 2) if device.type == "cuda" else 0.0

        val_loss, val_agg_psnr, val_batch_psnr = None, None, None
        if step % val_freq == 0 or step == num_steps:
            val_loss, val_agg_psnr, val_batch_psnr = evaluate_validation(
                adapter, val_loader, device, use_amp=use_amp, max_val_batches=max_val_batches
            )
            print(f"Step {step:3d}/{num_steps} | Train Loss: {loss.item():.6f} | Val Loss: {val_loss:.6f} | Val Agg PSNR: {val_agg_psnr:.2f} dB | LR: {curr_lr:.2e} | VRAM: {peak_alloc_mb:.1f} MB (alloc) / {peak_res_mb:.1f} MB (res) | GradNorm: {grad_norm.item():.4f}")
        elif step % 10 == 0 or step == 1:
            print(f"Step {step:3d}/{num_steps} | Train Loss: {loss.item():.6f} | LR: {curr_lr:.2e} | Time: {step_duration:.3f}s | GradNorm: {grad_norm.item():.4f}")

        entry = {
            "step": step,
            "train_loss": round(loss.item(), 6),
            "learning_rate": curr_lr,
            "gradients_finite": gradients_finite,
            "grad_norm": round(grad_norm.item(), 4),
            "step_duration_sec": round(step_duration, 4),
            "peak_allocated_vram_mb": round(peak_alloc_mb, 2),
            "peak_reserved_vram_mb": round(peak_res_mb, 2),
            "val_loss": round(val_loss, 6) if val_loss is not None else None,
            "val_aggregate_psnr": round(val_agg_psnr, 2) if val_agg_psnr is not None else None,
            "val_batch_mean_psnr": round(val_batch_psnr, 2) if val_batch_psnr is not None else None
        }
        history.append(entry)

        # Midpoint checkpoint save
        if step == num_steps // 2:
            adapter.save_checkpoint(mid_ckpt_path, optimizer=optimizer, scheduler=scheduler, scaler=scaler, step=step)
            print(f"Saved midpoint checkpoint at step {step}: {mid_ckpt_path.name}")

    total_duration = time.time() - start_time

    # Verify midpoint checkpoint reload & genuine subsequent training step
    print("\nExecuting genuine checkpoint restoration and post-resume training step verification...")
    # Calculate sha256 of checkpoint before reload
    ckpt_hash_before = hashlib.sha256(mid_ckpt_path.read_bytes()).hexdigest()

    fresh_adapter = EDVRAdapter(project_root=root).to(device)
    fresh_opt = torch.optim.Adam(fresh_adapter.model.parameters(), lr=lr)
    fresh_sched = CosineAnnealingLR(fresh_opt, T_max=num_steps, eta_min=1e-7)
    fresh_scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")

    meta = fresh_adapter.load_checkpoint(mid_ckpt_path, optimizer=fresh_opt, scheduler=fresh_sched, scaler=fresh_scaler)
    resumed_step = meta["step"]
    assert resumed_step == num_steps // 2, f"Expected step {num_steps // 2}, got {resumed_step}"

    # Execute an actual subsequent training step (Step 51) using restored state
    fresh_adapter.model.train()
    resume_batch = next(iter(train_loader))
    rx, ry = fresh_adapter.prepare_batch(resume_batch, device=device)
    fresh_opt.zero_grad()

    with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
        r_pred = fresh_adapter.forward(rx)
        r_loss = fresh_adapter.compute_loss(r_pred, ry)

    assert torch.isfinite(r_loss), f"Post-resume loss is non-finite: {r_loss.item()}"
    fresh_scaler.scale(r_loss).backward()

    if use_amp and device.type == "cuda":
        fresh_scaler.unscale_(fresh_opt)

    for p in fresh_adapter.model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), "Post-resume gradient is non-finite"

    r_norm = torch.nn.utils.clip_grad_norm_(fresh_adapter.model.parameters(), max_norm=float("inf"))
    assert torch.isfinite(r_norm), "Post-resume gradient norm is non-finite"

    fresh_scaler.step(fresh_opt)
    fresh_scaler.update()
    fresh_sched.step()

    # Confirm original checkpoint file on disk has not been modified
    ckpt_hash_after = hashlib.sha256(mid_ckpt_path.read_bytes()).hexdigest()
    assert ckpt_hash_before == ckpt_hash_after, "Checkpoint file was modified during verification!"

    print(f"Genuine post-resume training step (Step {resumed_step + 1}) completed successfully! Loss: {r_loss.item():.6f}, GradNorm: {r_norm.item():.4f}")
    print("Checkpoint file immutability verified!")

    # Calculate metrics summary
    initial_train_loss = history[0]["train_loss"]
    final_train_loss = history[-1]["train_loss"]
    final_val_loss = history[-1]["val_loss"]
    final_val_agg_psnr = history[-1]["val_aggregate_psnr"]
    final_val_batch_psnr = history[-1]["val_batch_mean_psnr"]

    pilot_summary = {
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "num_steps": num_steps,
        "batch_size": batch_size,
        "crop_size": list(crop_size),
        "total_duration_sec": round(total_duration, 2),
        "avg_step_duration_sec": round(total_duration / num_steps, 4),
        "initial_train_loss": initial_train_loss,
        "final_train_loss": final_train_loss,
        "initial_val_aggregate_psnr_db": round(init_val_agg_psnr, 2),
        "final_val_aggregate_psnr_db": round(final_val_agg_psnr, 2),
        "val_aggregate_psnr_gain_db": round(final_val_agg_psnr - init_val_agg_psnr, 2),
        "final_val_batch_mean_psnr_db": round(final_val_batch_psnr, 2),
        "max_val_batches": max_val_batches,
        "peak_allocated_vram_mb": round(max(h["peak_allocated_vram_mb"] for h in history), 2),
        "peak_reserved_vram_mb": round(max(h["peak_reserved_vram_mb"] for h in history), 2),
        "all_gradients_finite": all(h.get("gradients_finite", False) for h in history),
        "gradients_finite": all(h.get("gradients_finite", False) for h in history),
        "checkpoint_restored_successfully": True,
        "post_resume_training_step_verified": True,
        "history": history
    }

    out_json = save_dir / "final_controlled_pilot_metrics.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(pilot_summary, f, indent=2)

    print(f"\nPilot Complete in {total_duration:.1f}s! Initial Val Agg PSNR: {init_val_agg_psnr:.2f} dB -> Final Val Agg PSNR: {final_val_agg_psnr:.2f} dB")
    return pilot_summary


def run_full_training(
    total_steps: int = 50000,
    val_freq: int = 2500,
    ckpt_freq: int = 5000,
    batch_size: int = 2,
    crop_size: Tuple[int, int] = (256, 256),
    lr: float = 2e-4,
    use_amp: bool = True,
    max_val_batches: Optional[int] = 15,
    resume_from: Optional[Union[str, Path]] = None,
    run_dir: Optional[Union[str, Path]] = None,
    project_root: Optional[Path] = None,
    seed: int = 42
) -> Dict[str, Any]:
    """Execute full-scale baseline training for EDVR on REDS video deblurring."""
    root = project_root or get_project_root()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Set seeds for reproducibility
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    if run_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = root / "experiments" / f"edvr_run_{timestamp}"
    else:
        run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"Initializing EDVR Full Training on {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"Run directory: {run_dir}")
    print(f"Total steps: {total_steps} | Val freq: {val_freq} | Ckpt freq: {ckpt_freq} | Batch size: {batch_size}")

    adapter = EDVRAdapter(project_root=root).to(device)
    adapter.model.train()

    train_transform = PairedTransformPipeline(
        crop_size=crop_size,
        flip_h=True,
        flip_v=True,
        temporal_reverse=True,
        seed=seed
    )
    val_transform = PairedTransformPipeline(
        crop_size=crop_size,
        flip_h=False,
        flip_v=False,
        temporal_reverse=False
    )

    train_ds = REDSVideoDeblurDataset(
        partition="train",
        project_root=root,
        window_size=5,
        boundary_policy="replicate",
        transform=train_transform
    )
    val_ds = REDSVideoDeblurDataset(
        partition="val",
        project_root=root,
        window_size=5,
        boundary_policy="replicate",
        transform=val_transform
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=adapter.collate_fn,
        num_workers=0
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=adapter.collate_fn,
        num_workers=0
    )

    optimizer = torch.optim.Adam(adapter.model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=0.0)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-7)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")

    start_step = 0
    best_val_psnr = -float("inf")

    if resume_from is not None:
        resume_path = Path(resume_from)
        print(f"Resuming training from checkpoint: {resume_path}")
        meta = adapter.load_checkpoint(resume_path, optimizer=optimizer, scheduler=scheduler, scaler=scaler)
        start_step = meta.get("step", 0)
        best_val_psnr = meta.get("extra_meta", {}).get("best_val_psnr", -float("inf"))
        print(f"Resumed from step {start_step} with best validation PSNR: {best_val_psnr:.2f} dB")

    history = []
    step = start_step
    train_iter = iter(train_loader)
    start_time = time.time()

    print(f"Beginning training loop from step {start_step + 1} to {total_steps}...")

    while step < total_steps:
        step += 1
        step_start = time.time()

        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        x, y = adapter.prepare_batch(batch, device=device)
        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
            pred = adapter.forward(x)
            loss = adapter.compute_loss(pred, y)

        scaler.scale(loss).backward()

        if use_amp and device.type == "cuda":
            scaler.unscale_(optimizer)

        gradients_finite = True
        for parameter in adapter.model.parameters():
            if parameter.grad is not None:
                if not torch.isfinite(parameter.grad).all().item():
                    gradients_finite = False
                    break

        if not gradients_finite:
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError(f"Non-finite gradients detected at step {step}")

        grad_norm = torch.nn.utils.clip_grad_norm_(
            adapter.model.parameters(),
            max_norm=float("inf")
        )

        if not torch.isfinite(grad_norm).item():
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError(f"Non-finite gradient norm detected at step {step}")

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if device.type == "cuda":
            torch.cuda.synchronize()

        step_duration = time.time() - step_start
        curr_lr = optimizer.param_groups[0]["lr"]

        peak_alloc_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if device.type == "cuda" else 0.0
        peak_res_mb = torch.cuda.max_memory_reserved() / (1024 ** 2) if device.type == "cuda" else 0.0

        val_loss, val_agg_psnr, val_batch_psnr = None, None, None

        if step % val_freq == 0 or step == total_steps:
            val_loss, val_agg_psnr, val_batch_psnr = evaluate_validation(
                adapter, val_loader, device, use_amp=use_amp, max_val_batches=max_val_batches
            )
            print(f"Step {step:6d}/{total_steps} | Train Loss: {loss.item():.6f} | Val Loss: {val_loss:.6f} | Val Agg PSNR: {val_agg_psnr:.2f} dB | LR: {curr_lr:.2e} | VRAM: {peak_alloc_mb:.1f} MB (alloc)")
            
            if val_agg_psnr > best_val_psnr:
                best_val_psnr = val_agg_psnr
                best_ckpt_path = ckpt_dir / "best_checkpoint.pth"
                adapter.save_checkpoint(
                    best_ckpt_path,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    step=step,
                    extra_meta={"best_val_psnr": best_val_psnr}
                )
                print(f"--> New best model saved to {best_ckpt_path.name} (Val PSNR: {best_val_psnr:.2f} dB)")
        elif step % 100 == 0 or step == 1:
            print(f"Step {step:6d}/{total_steps} | Train Loss: {loss.item():.6f} | LR: {curr_lr:.2e} | Time: {step_duration:.3f}s | GradNorm: {grad_norm.item():.4f}")

        entry = {
            "step": step,
            "train_loss": round(loss.item(), 6),
            "learning_rate": curr_lr,
            "gradients_finite": gradients_finite,
            "grad_norm": round(grad_norm.item(), 4),
            "step_duration_sec": round(step_duration, 4),
            "peak_allocated_vram_mb": round(peak_alloc_mb, 2),
            "peak_reserved_vram_mb": round(peak_res_mb, 2),
            "val_loss": round(val_loss, 6) if val_loss is not None else None,
            "val_aggregate_psnr": round(val_agg_psnr, 2) if val_agg_psnr is not None else None,
            "val_batch_mean_psnr": round(val_batch_psnr, 2) if val_batch_psnr is not None else None
        }
        history.append(entry)

        # Periodic checkpoint
        if step % ckpt_freq == 0:
            step_ckpt_path = ckpt_dir / f"checkpoint_step_{step}.pth"
            adapter.save_checkpoint(step_ckpt_path, optimizer=optimizer, scheduler=scheduler, scaler=scaler, step=step, extra_meta={"best_val_psnr": best_val_psnr})
            print(f"Saved periodic checkpoint: {step_ckpt_path.name}")

        # Always update latest checkpoint on validation or periodic checkpoints
        if step % val_freq == 0 or step % ckpt_freq == 0 or step == total_steps:
            latest_ckpt_path = ckpt_dir / "latest_checkpoint.pth"
            adapter.save_checkpoint(latest_ckpt_path, optimizer=optimizer, scheduler=scheduler, scaler=scaler, step=step, extra_meta={"best_val_psnr": best_val_psnr})

    total_duration = time.time() - start_time
    final_ckpt_path = ckpt_dir / "final_checkpoint.pth"
    adapter.save_checkpoint(final_ckpt_path, optimizer=optimizer, scheduler=scheduler, scaler=scaler, step=step, extra_meta={"best_val_psnr": best_val_psnr})
    print(f"Saved final checkpoint: {final_ckpt_path.name}")

    summary = {
        "run_dir": str(run_dir),
        "total_steps": total_steps,
        "completed_steps": step - start_step,
        "start_step": start_step,
        "end_step": step,
        "best_val_psnr_db": round(best_val_psnr, 2) if math.isfinite(best_val_psnr) else None,
        "total_duration_sec": round(total_duration, 2),
        "all_gradients_finite": all(h.get("gradients_finite", False) for h in history),
        "history": history
    }

    summary_file = run_dir / "training_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Full Training Run Complete in {total_duration / 3600:.2f}h. Summary saved to {summary_file}")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="EDVR REDS Video Deblurring Training and Pilot Harness")
    parser.add_argument("--mode", type=str, choices=["pilot", "train", "resume"], default="pilot",
                        help="Execution mode: 'pilot' for 100-step smoke test, 'train' for full baseline run, 'resume' to continue")
    parser.add_argument("--total_steps", type=int, default=50000, help="Total optimizer steps for full training")
    parser.add_argument("--val_freq", type=int, default=2500, help="Validation frequency (steps)")
    parser.add_argument("--ckpt_freq", type=int, default=5000, help="Periodic checkpoint frequency (steps)")
    parser.add_argument("--batch_size", type=int, default=2, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Initial learning rate")
    parser.add_argument("--no_amp", action="store_true", help="Disable automatic mixed precision")
    parser.add_argument("--max_val_batches", type=int, default=15, help="Max batches for validation (None for full set)")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint for resuming")
    parser.add_argument("--run_dir", type=str, default=None, help="Directory to save run outputs and checkpoints")
    parser.add_argument("--num_pilot_steps", type=int, default=100, help="Number of steps for pilot mode")
    parser.add_argument("--pilot_val_freq", type=int, default=25, help="Validation frequency for pilot mode")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "pilot":
        run_training_pilot(
            num_steps=args.num_pilot_steps,
            val_freq=args.pilot_val_freq,
            batch_size=args.batch_size,
            lr=args.lr,
            use_amp=not args.no_amp,
            max_val_batches=args.max_val_batches
        )
    elif args.mode in ("train", "resume"):
        run_full_training(
            total_steps=args.total_steps,
            val_freq=args.val_freq,
            ckpt_freq=args.ckpt_freq,
            batch_size=args.batch_size,
            lr=args.lr,
            use_amp=not args.no_amp,
            max_val_batches=args.max_val_batches,
            resume_from=args.resume_from,
            run_dir=args.run_dir
        )
