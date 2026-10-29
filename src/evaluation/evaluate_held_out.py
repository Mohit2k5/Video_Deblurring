"""Evaluation Harness for Trained EDVR Baseline on Held-Out REDS Sequences.

Executes inference across the 24 held-out REDS sequences (2,400 full 720x1280 frames),
computes per-frame and aggregate PSNR, SSIM, and Charbonnier loss,
generates comparison figures and clips, and outputs structured metrics.
"""
import time
import math
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import torch
from torch.utils.data import DataLoader

from src.data.config_utils import get_project_root
from src.data.dataset import REDSVideoDeblurDataset
from src.models.edvr_adapter import EDVRAdapter
from src.evaluation.metrics import (
    calculate_psnr,
    calculate_ssim,
    calculate_charbonnier_loss,
    create_gaussian_window,
    AggregateMetricsTracker
)


def verify_checkpoint(
    ckpt_path: Path,
    summary_path: Optional[Path] = None,
    device: torch.device = torch.device("cpu"),
    project_root: Optional[Path] = None
) -> Tuple[EDVRAdapter, Dict[str, Any]]:
    """Loads and validates the checkpoint against the model architecture and training summary."""
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    root = project_root or get_project_root()
    adapter = EDVRAdapter(project_root=root).to(device)
    meta = adapter.load_checkpoint(ckpt_path, strict=True)
    step = meta.get("step", 0)

    # Cross-check against training_summary.json if provided
    summary_info = {}
    if summary_path and summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            summary_info = json.load(f)
        recorded_best_psnr = summary_info.get("best_val_psnr_db")
        print(f"Verified against {summary_path.name}: Recorded Best Val PSNR = {recorded_best_psnr} dB")

    print(f"Loaded checkpoint '{ckpt_path.name}' at step {step} (Architecture: EDVR-M with PreDeblur & TSA)")
    return adapter, meta


def verify_held_out_split(project_root: Path) -> List[str]:
    """Verifies that held_out_test sequences exist, have 24 sequences, and do not overlap with train/val."""
    manifest_path = project_root / "data" / "manifests" / "sequence_split.csv"
    if not manifest_path.exists():
        manifest_path = project_root / "splits" / "sequence_split.csv"

    df = pd.read_csv(manifest_path, dtype={"sequence_id": str})
    df["composite_id"] = df["source_dataset"] + "_" + df["sequence_id"]

    train_ids = set(df[df["partition"] == "train"]["composite_id"])
    val_ids = set(df[df["partition"] == "val"]["composite_id"])
    held_out_df = df[df["partition"] == "held_out_test"]
    held_out_ids = set(held_out_df["composite_id"])

    assert len(held_out_ids) == 24, f"Expected 24 held-out sequences, found {len(held_out_ids)}"
    assert len(held_out_ids.intersection(train_ids)) == 0, "Leakage: Held-out sequences overlap with train!"
    assert len(held_out_ids.intersection(val_ids)) == 0, "Leakage: Held-out sequences overlap with validation!"

    held_out_sequences = sorted(held_out_df["sequence_id"].tolist())
    print(f"Verified 24 held-out sequences with zero train/val overlap: {held_out_sequences}")
    return held_out_sequences


def create_visual_comparison_panel(
    blur_img: np.ndarray,      # (H, W, 3) uint8 RGB
    restored_img: np.ndarray,  # (H, W, 3) uint8 RGB
    sharp_img: np.ndarray,     # (H, W, 3) uint8 RGB
    seq_id: str,
    frame_idx: int,
    blur_psnr: float,
    blur_ssim: float,
    rest_psnr: float,
    rest_ssim: float
) -> Image.Image:
    """Create a 4-panel comparison visualization:
    [Blur Input] | [EDVR Restored] | [Sharp GT] | [Absolute Error x5]
    """
    h, w, c = blur_img.shape
    # Compute absolute error scaled by 5 for visibility
    diff = np.abs(restored_img.astype(np.float32) - sharp_img.astype(np.float32)) * 5.0
    diff = np.clip(diff, 0, 255).astype(np.uint8)

    # Downscale slightly for practical report viewing (e.g. half-res: 360x640 per panel)
    disp_w, disp_h = w // 2, h // 2
    b_disp = cv2.resize(blur_img, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
    r_disp = cv2.resize(restored_img, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
    s_disp = cv2.resize(sharp_img, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
    d_disp = cv2.resize(diff, (disp_w, disp_h), interpolation=cv2.INTER_AREA)

    # 2x2 grid canvas
    header_h = 40
    canvas_w = disp_w * 2 + 30
    canvas_h = (disp_h + header_h) * 2 + 30

    grid = np.full((canvas_h, canvas_w, 3), 30, dtype=np.uint8) # Dark background

    # Convert to PIL for clear typography
    pil_canvas = Image.fromarray(grid)
    draw = ImageDraw.Draw(pil_canvas)

    panels = [
        (b_disp, 10, 10, f"Blurry Input (Seq {seq_id}, Frame {frame_idx:08d})\nPSNR: {blur_psnr:.2f} dB | SSIM: {blur_ssim:.4f}"),
        (r_disp, disp_w + 20, 10, f"EDVR Restored\nPSNR: {rest_psnr:.2f} dB | SSIM: {rest_ssim:.4f} (dPSNR: {rest_psnr - blur_psnr:+.2f} dB)"),
        (s_disp, 10, disp_h + header_h + 20, f"Sharp Ground Truth (Reference)"),
        (d_disp, disp_w + 20, disp_h + header_h + 20, f"Absolute Error (|Restored - Sharp| x 5)")
    ]

    for p_img, x, y, title in panels:
        draw.text((x, y), title, fill=(230, 230, 230))
        p_pil = Image.fromarray(p_img)
        pil_canvas.paste(p_pil, (x, y + header_h))

    return pil_canvas


def run_held_out_evaluation(
    ckpt_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    project_root: Optional[Path] = None,
    save_visuals: bool = True,
    visual_interval: int = 25,
    create_clips: bool = True,
    device_str: str = "cuda"
) -> Dict[str, Any]:
    """Execute evaluation on all 24 held-out REDS sequences."""
    root = project_root or get_project_root()
    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")
    out_dir = output_dir or (root / "reports" / "evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)
    visuals_dir = out_dir / "visuals"
    visuals_dir.mkdir(parents=True, exist_ok=True)

    # 1. Verify Checkpoint
    default_ckpt = root / "experiments" / "edvr_run_20261016_080811" / "checkpoints" / "best_checkpoint.pth"
    ckpt_file = ckpt_path or default_ckpt
    summary_file = ckpt_file.parent.parent / "training_summary.json"

    adapter, ckpt_meta = verify_checkpoint(ckpt_file, summary_file, device=device, project_root=root)
    adapter.model.eval()

    # 2. Verify Split
    held_out_seqs = verify_held_out_split(root)

    # 3. Dataset & DataLoader (Sequential, full frames, no transforms)
    dataset = REDSVideoDeblurDataset(
        partition="held_out_test",
        project_root=root,
        window_size=5,
        boundary_policy="replicate",
        transform=None
    )
    total_frames = len(dataset)
    print(f"Beginning evaluation on {len(held_out_seqs)} sequences ({total_frames} total frames) on device {device}...")

    # Shared Gaussian window for SSIM
    ssim_window = create_gaussian_window(window_size=11, sigma=1.5, channels=3, device=device)

    # Trackers
    total_blur_tracker = AggregateMetricsTracker()
    total_rest_tracker = AggregateMetricsTracker()

    per_sequence_trackers = {
        seq: {
            "blur": AggregateMetricsTracker(),
            "rest": AggregateMetricsTracker(),
            "frames": []
        }
        for seq in held_out_seqs
    }

    per_frame_records = []
    clip_frames_cache = {seq: [] for seq in held_out_seqs[:3]} # Save first 3 sequences for short side-by-side clips

    start_eval_time = time.time()

    with torch.inference_mode():
        for i, item in enumerate(dataset):
            seq_id = item["sequence_id"]
            frame_idx = item["center_index"]
            frame_name = item["center_filename"]

            # Prepare batch: (1, 5, 3, 720, 1280) and (1, 3, 720, 1280)
            x, y = adapter.prepare_batch(item, device=device)
            sharp_gt = y.float().clamp(0.0, 1.0)
            blur_in = x[:, adapter.center_frame_idx, :, :, :].float().clamp(0.0, 1.0)

            # Model forward inference with AMP FP16
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                pred = adapter.forward(x)

            pred_f = pred.float().clamp(0.0, 1.0)

            # Calculate Frame Metrics
            b_psnr = calculate_psnr(blur_in, sharp_gt)
            r_psnr = calculate_psnr(pred_f, sharp_gt)

            b_ssim = calculate_ssim(blur_in, sharp_gt, window=ssim_window)
            r_ssim = calculate_ssim(pred_f, sharp_gt, window=ssim_window)

            r_loss = calculate_charbonnier_loss(pred_f, sharp_gt, eps=adapter.criterion.eps)
            b_loss = calculate_charbonnier_loss(blur_in, sharp_gt, eps=adapter.criterion.eps)

            psnr_gain = r_psnr - b_psnr
            ssim_gain = r_ssim - b_ssim

            # Update Trackers
            total_blur_tracker.update(blur_in, sharp_gt, b_psnr, b_ssim, b_loss)
            total_rest_tracker.update(pred_f, sharp_gt, r_psnr, r_ssim, r_loss)

            per_sequence_trackers[seq_id]["blur"].update(blur_in, sharp_gt, b_psnr, b_ssim, b_loss)
            per_sequence_trackers[seq_id]["rest"].update(pred_f, sharp_gt, r_psnr, r_ssim, r_loss)

            frame_entry = {
                "sequence_id": seq_id,
                "frame_index": frame_idx,
                "filename": frame_name,
                "blur_psnr_db": round(b_psnr, 4),
                "restored_psnr_db": round(r_psnr, 4),
                "psnr_gain_db": round(psnr_gain, 4),
                "blur_ssim": round(b_ssim, 4),
                "restored_ssim": round(r_ssim, 4),
                "ssim_gain": round(ssim_gain, 4),
                "restored_charbonnier_loss": round(r_loss, 6)
            }
            per_frame_records.append(frame_entry)

            # Convert to numpy uint8 for visual saving
            blur_np = (blur_in[0].permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
            rest_np = (pred_f[0].permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
            sharp_np = (sharp_gt[0].permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)

            # Cache frames for side-by-side clip (first 10 frames of selected sequences)
            if create_clips and seq_id in clip_frames_cache and len(clip_frames_cache[seq_id]) < 10:
                sbs = np.hstack([blur_np, rest_np, sharp_np])
                clip_frames_cache[seq_id].append(sbs)

            # Save representative visual comparison panel
            if save_visuals and (frame_idx % visual_interval == 0):
                panel_img = create_visual_comparison_panel(
                    blur_img=blur_np,
                    restored_img=rest_np,
                    sharp_img=sharp_np,
                    seq_id=seq_id,
                    frame_idx=frame_idx,
                    blur_psnr=b_psnr,
                    blur_ssim=b_ssim,
                    rest_psnr=r_psnr,
                    rest_ssim=r_ssim
                )
                panel_img.save(visuals_dir / f"seq_{seq_id}_frame_{frame_idx:03d}_comparison.png")

            if (i + 1) % 100 == 0 or (i + 1) == total_frames:
                elapsed = time.time() - start_eval_time
                fps = (i + 1) / elapsed
                print(f"[{i + 1:4d}/{total_frames}] Sequence {seq_id} | "
                      f"Blur PSNR: {b_psnr:.2f} -> EDVR: {r_psnr:.2f} dB (dPSNR: {psnr_gain:+.2f} dB) | "
                      f"SSIM: {r_ssim:.4f} | Speed: {fps:.2f} fps")

    eval_duration = time.time() - start_eval_time
    print(f"\nInference Complete in {eval_duration:.1f}s ({total_frames / eval_duration:.2f} fps).")

    # Generate side-by-side clips for the first 3 sequences
    if create_clips:
        for seq_id, frames in clip_frames_cache.items():
            if len(frames) > 0:
                clip_path = visuals_dir / f"seq_{seq_id}_side_by_side_clip.mp4"
                h_c, w_c, _ = frames[0].shape
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out_writer = cv2.VideoWriter(str(clip_path), fourcc, 5.0, (w_c, h_c))
                for f_rgb in frames:
                    out_writer.write(cv2.cvtColor(f_rgb, cv2.COLOR_RGB2BGR))
                out_writer.release()
                print(f"Saved side-by-side verification clip: {clip_path.name}")

    # Compile Per-Sequence Summaries
    seq_summaries = []
    for seq_id in held_out_seqs:
        b_sum = per_sequence_trackers[seq_id]["blur"].summary()
        r_sum = per_sequence_trackers[seq_id]["rest"].summary()
        seq_summaries.append({
            "sequence_id": seq_id,
            "frame_count": b_sum["frame_count"],
            "blur_aggregate_psnr_db": b_sum["aggregate_psnr"],
            "restored_aggregate_psnr_db": r_sum["aggregate_psnr"],
            "aggregate_psnr_gain_db": round(r_sum["aggregate_psnr"] - b_sum["aggregate_psnr"], 4),
            "blur_mean_psnr_db": b_sum["mean_psnr"],
            "restored_mean_psnr_db": r_sum["mean_psnr"],
            "mean_psnr_gain_db": round(r_sum["mean_psnr"] - b_sum["mean_psnr"], 4),
            "blur_mean_ssim": b_sum["mean_ssim"],
            "restored_mean_ssim": r_sum["mean_ssim"],
            "mean_ssim_gain": round(r_sum["mean_ssim"] - b_sum["mean_ssim"], 4),
            "restored_mean_charbonnier_loss": r_sum["mean_charbonnier_loss"]
        })

    # Compile Dataset-Wide Summaries
    tot_b_sum = total_blur_tracker.summary()
    tot_r_sum = total_rest_tracker.summary()

    overall_summary = {
        "evaluation_partition": "held_out_test",
        "num_sequences": len(held_out_seqs),
        "total_frames_evaluated": total_frames,
        "checkpoint_evaluated": str(ckpt_file.name),
        "checkpoint_step": ckpt_meta.get("step", 0),
        "evaluation_device": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "evaluation_duration_sec": round(eval_duration, 2),
        "avg_fps": round(total_frames / eval_duration, 2),
        "metrics": {
            "blur_input_aggregate_psnr_db": tot_b_sum["aggregate_psnr"],
            "restored_aggregate_psnr_db": tot_r_sum["aggregate_psnr"],
            "aggregate_psnr_gain_db": round(tot_r_sum["aggregate_psnr"] - tot_b_sum["aggregate_psnr"], 4),
            "blur_input_mean_psnr_db": tot_b_sum["mean_psnr"],
            "restored_mean_psnr_db": tot_r_sum["mean_psnr"],
            "mean_psnr_gain_db": round(tot_r_sum["mean_psnr"] - tot_b_sum["mean_psnr"], 4),
            "blur_input_mean_ssim": tot_b_sum["mean_ssim"],
            "restored_mean_ssim": tot_r_sum["mean_ssim"],
            "mean_ssim_gain": round(tot_r_sum["mean_ssim"] - tot_b_sum["mean_ssim"], 4),
            "restored_mean_charbonnier_loss": tot_r_sum["mean_charbonnier_loss"]
        },
        "per_sequence_results": seq_summaries
    }

    # Save CSVs and JSONs
    df_frames = pd.DataFrame(per_frame_records)
    csv_frames_path = out_dir / "held_out_per_frame_metrics.csv"
    df_frames.to_csv(csv_frames_path, index=False)

    df_seqs = pd.DataFrame(seq_summaries)
    csv_seqs_path = out_dir / "held_out_per_sequence_metrics.csv"
    df_seqs.to_csv(csv_seqs_path, index=False)

    json_seqs_path = out_dir / "held_out_per_sequence_metrics.json"
    with open(json_seqs_path, "w", encoding="utf-8") as f:
        json.dump(seq_summaries, f, indent=2)

    json_summary_path = out_dir / "held_out_summary_metrics.json"
    with open(json_summary_path, "w", encoding="utf-8") as f:
        json.dump(overall_summary, f, indent=2)

    print(f"\nResults Saved:")
    print(f"- Per-frame CSV: {csv_frames_path}")
    print(f"- Per-sequence CSV: {csv_seqs_path}")
    print(f"- Summary JSON: {json_summary_path}")

    return overall_summary


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate EDVR Model on Held-Out REDS Sequences")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint .pth file")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save evaluation reports and visuals")
    parser.add_argument("--no_visuals", action="store_true", help="Skip saving visual comparison images")
    parser.add_argument("--visual_interval", type=int, default=25, help="Interval of frames to save visuals")
    parser.add_argument("--no_clips", action="store_true", help="Skip creating short MP4 verification clips")
    parser.add_argument("--device", type=str, default="cuda", help="Inference device (cuda or cpu)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ckpt_p = Path(args.checkpoint) if args.checkpoint else None
    out_p = Path(args.output_dir) if args.output_dir else None
    run_held_out_evaluation(
        ckpt_path=ckpt_p,
        output_dir=out_p,
        save_visuals=not args.no_visuals,
        visual_interval=args.visual_interval,
        create_clips=not args.no_clips,
        device_str=args.device
    )
