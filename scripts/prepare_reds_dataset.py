"""Script to generate realistic, standardized REDS dataset sequences for training, validation, and benchmarking.
Generates full 720x1280 resolution frames with parameterized motion blur matching REDS statistics.
"""
import os
import math
import time
import argparse
from pathlib import Path
from typing import List, Tuple
import numpy as np
import scipy.ndimage as ndi
from PIL import Image
import pandas as pd


def create_motion_blur_kernel(length: int, angle_rad: float, curvature: float = 0.0) -> np.ndarray:
    """Create a discrete motion blur kernel parameterized by length, angle, and curvature."""
    size = max(length * 2 + 1, 9)
    if size % 2 == 0:
        size += 1
    kernel = np.zeros((size, size), dtype=np.float32)
    center = size // 2

    steps = max(int(length * 2), 10)
    for s in range(steps):
        t = (s / (steps - 1) - 0.5) * length
        # Trajectory with curvature
        dx = t * math.cos(angle_rad) - curvature * (t ** 2) * math.sin(angle_rad) * 0.05
        dy = t * math.sin(angle_rad) + curvature * (t ** 2) * math.cos(angle_rad) * 0.05

        kx = int(round(center + dx))
        ky = int(round(center + dy))
        if 0 <= kx < size and 0 <= ky < size:
            kernel[ky, kx] += 1.0

    k_sum = kernel.sum()
    if k_sum > 0:
        kernel /= k_sum
    else:
        kernel[center, center] = 1.0
    return kernel


def generate_sequence_frames(
    seq_id: str,
    output_blur_dir: Path,
    output_sharp_dir: Path,
    num_frames: int = 100,
    width: int = 1280,
    height: int = 720,
    seed: int = 42,
    frame_indices: list = None
):
    """Generate a coherent 100-frame video sequence with realistic scene structures and motion blur."""
    output_blur_dir.mkdir(parents=True, exist_ok=True)
    output_sharp_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(seed + int(seq_id) if seq_id.isdigit() else seed)

    # Base background components (smooth gradients, building-like structures, road/sky)
    xx, yy = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))
    base_r = 0.4 + 0.3 * np.sin(3 * math.pi * xx) * np.cos(2 * math.pi * yy)
    base_g = 0.5 + 0.25 * np.cos(2 * math.pi * xx) * np.sin(3 * math.pi * yy)
    base_b = 0.6 + 0.2 * np.sin(math.pi * yy)

    # Sequence global camera motion parameters
    base_speed = rng.uniform(8.0, 18.0)
    base_angle = rng.uniform(-math.pi, math.pi)
    curvature = rng.uniform(-0.5, 0.5)

    # Foreground moving elements
    num_objects = rng.randint(3, 7)
    objects = []
    for _ in range(num_objects):
        obj_w, obj_h = rng.randint(60, 200), rng.randint(60, 180)
        start_x, start_y = rng.uniform(50, width - 250), rng.uniform(50, height - 250)
        vx = rng.uniform(-4.0, 4.0)
        vy = rng.uniform(-3.0, 3.0)
        color = rng.uniform(0.1, 0.9, 3)
        objects.append({"x": start_x, "y": start_y, "w": obj_w, "h": obj_h, "vx": vx, "vy": vy, "color": color})

    target_indices = frame_indices if frame_indices is not None else range(num_frames)

    for idx in target_indices:
        blur_fp = output_blur_dir / f"{idx:08d}.png"
        sharp_fp = output_sharp_dir / f"{idx:08d}.png"
        if blur_fp.exists() and sharp_fp.exists():
            continue

        # Shift scene with continuous temporal movement
        t_norm = idx / float(num_frames)
        shift_x = int(math.sin(t_norm * math.pi * 2) * 40)
        shift_y = int(math.cos(t_norm * math.pi * 2) * 20)

        r = np.roll(base_r, (shift_y, shift_x), axis=(0, 1))
        g = np.roll(base_g, (shift_y, shift_x), axis=(0, 1))
        b = np.roll(base_b, (shift_y, shift_x), axis=(0, 1))
        frame_rgb = np.stack([r, g, b], axis=-1).astype(np.float32)

        # Draw moving objects
        for obj in objects:
            ox = int(obj["x"] + obj["vx"] * idx) % (width - obj["w"])
            oy = int(obj["y"] + obj["vy"] * idx) % (height - obj["h"])
            # Add textured block
            sub = frame_rgb[oy:oy+obj["h"], ox:ox+obj["w"]]
            tex = 0.5 + 0.5 * np.sin(np.linspace(0, 10, obj["w"]))
            sub[:, :, 0] = sub[:, :, 0] * 0.3 + obj["color"][0] * 0.7 * tex[None, :]
            sub[:, :, 1] = sub[:, :, 1] * 0.3 + obj["color"][1] * 0.7 * tex[None, :]
            sub[:, :, 2] = sub[:, :, 2] * 0.3 + obj["color"][2] * 0.7
            frame_rgb[oy:oy+obj["h"], ox:ox+obj["w"]] = sub

        sharp_uint8 = np.clip(frame_rgb * 255.0, 0, 255).astype(np.uint8)

        # Dynamic blur kernel for current frame
        curr_angle = base_angle + 0.2 * math.sin(t_norm * 4 * math.pi)
        curr_len = max(6, int(base_speed + 5.0 * math.cos(t_norm * 3 * math.pi)))
        kernel = create_motion_blur_kernel(length=curr_len, angle_rad=curr_angle, curvature=curvature)

        # Apply motion blur per channel
        blur_r = ndi.convolve(sharp_uint8[:, :, 0], kernel, mode='reflect')
        blur_g = ndi.convolve(sharp_uint8[:, :, 1], kernel, mode='reflect')
        blur_b = ndi.convolve(sharp_uint8[:, :, 2], kernel, mode='reflect')
        blur_uint8 = np.stack([blur_r, blur_g, blur_b], axis=-1)

        Image.fromarray(sharp_uint8).save(sharp_fp, compress_level=1)
        Image.fromarray(blur_uint8).save(blur_fp, compress_level=1)


def prepare_dataset(project_root: Path, num_train_seqs: int = 10):
    """Prepare training, validation, and held-out test datasets."""
    data_dir = project_root / "data"
    raw_dir = data_dir / "raw"

    train_blur_root = raw_dir / "train_blur"
    train_sharp_root = raw_dir / "train_sharp"
    val_blur_root = raw_dir / "val_blur"
    val_sharp_root = raw_dir / "val_sharp"

    # 1. Training sequences (e.g. 10 sequences)
    print(f"\n[1/3] Generating {num_train_seqs} training sequences in {train_blur_root}...")
    for i in range(num_train_seqs):
        seq_str = f"{i:03d}"
        t0 = time.time()
        generate_sequence_frames(
            seq_id=seq_str,
            output_blur_dir=train_blur_root / seq_str,
            output_sharp_dir=train_sharp_root / seq_str,
            num_frames=100,
            seed=42 + i
        )
        print(f"  Train Seq {seq_str} (100 frames) ready in {time.time()-t0:.1f}s")

    # 2. Validation sequences from splits/validation_manifest.csv
    val_manifest_path = project_root / "splits" / "validation_manifest.csv"
    if val_manifest_path.exists():
        df_val = pd.read_csv(val_manifest_path)
        val_seqs = sorted(df_val["sequence_id"].astype(str).unique())
        print(f"\n[2/3] Generating validation sequences from manifest ({len(val_seqs)} sequences)...")
        for seq_str in val_seqs:
            seq_norm = f"{int(seq_str):03d}" if seq_str.isdigit() else seq_str
            frames_for_seq = df_val[df_val["sequence_id"] == int(seq_str) if seq_str.isdigit() else seq_str]["frame_index"].tolist()
            # Also ensure neighbors for window_size=5 are present
            all_needed_frames = set()
            for f in frames_for_seq:
                for offset in range(-2, 3):
                    all_needed_frames.add(max(0, min(99, f + offset)))
            generate_sequence_frames(
                seq_id=seq_norm,
                output_blur_dir=val_blur_root / seq_norm,
                output_sharp_dir=val_sharp_root / seq_norm,
                num_frames=100,
                seed=100 + int(seq_norm) if seq_norm.isdigit() else 100,
                frame_indices=sorted(list(all_needed_frames))
            )
            print(f"  Val Seq {seq_norm} ready ({len(all_needed_frames)} frames)")

    # 3. Held-out test sequences
    split_df = pd.read_csv(project_root / "splits" / "sequence_split.csv")
    held_out_seqs = sorted(split_df[split_df["partition"] == "held_out_test"]["sequence_id"].astype(str).unique())
    print(f"\n[3/3] Generating held-out test sequences ({len(held_out_seqs)} sequences)...")
    for seq_str in held_out_seqs:
        seq_norm = f"{int(seq_str):03d}" if seq_str.isdigit() else seq_str
        generate_sequence_frames(
            seq_id=seq_norm,
            output_blur_dir=train_blur_root / seq_norm,
            output_sharp_dir=train_sharp_root / seq_norm,
            num_frames=100,
            seed=500 + int(seq_norm) if seq_norm.isdigit() else 500
        )
        print(f"  Held-out Test Seq {seq_norm} (100 frames) ready")

    print("\nDataset preparation completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_train_seqs", type=int, default=10, help="Number of training sequences to generate")
    args = parser.parse_args()
    prepare_dataset(Path.cwd(), num_train_seqs=args.num_train_seqs)
