"""Visualization module for REDS dataset side-by-side pairs and consecutive frames."""
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image


def sample_diverse_pairs(
    valid_pairs: List[Dict[str, Any]],
    requested_count: int,
    random_seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Sample up to requested_count pairs, preferring distinct sequences.
    Reproducible with fixed random_seed.
    """
    if requested_count <= 0 or not valid_pairs:
        return []

    if len(valid_pairs) <= requested_count:
        return list(valid_pairs)

    rng = random.Random(random_seed)

    # Group pairs by sequence
    seq_groups: Dict[str, List[Dict[str, Any]]] = {}
    for p in valid_pairs:
        seq = p["sequence"]
        if seq not in seq_groups:
            seq_groups[seq] = []
        seq_groups[seq].append(p)

    sorted_seqs = sorted(list(seq_groups.keys()))
    rng.shuffle(sorted_seqs)

    selected = []
    # Round 1: pick 1 from each sequence until target reached
    for seq in sorted_seqs:
        if len(selected) >= requested_count:
            break
        chosen_pair = rng.choice(seq_groups[seq])
        selected.append(chosen_pair)

    # If still need more, sample from remaining without replacement
    if len(selected) < requested_count:
        selected_set = {(p["sequence"], p["filename"]) for p in selected}
        remaining = [p for p in valid_pairs if (p["sequence"], p["filename"]) not in selected_set]
        rng.shuffle(remaining)
        needed = requested_count - len(selected)
        selected.extend(remaining[:needed])

    return selected


def plot_blur_sharp_pairs(
    valid_pairs: List[Dict[str, Any]],
    requested_count: int,
    output_path: Path,
    title_prefix: str = "Training",
    random_seed: int = 42
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Plot side-by-side blurred and sharp frames.
    Saves figure to output_path.
    Returns (selected_pairs_metadata, stats_dict).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    selected = sample_diverse_pairs(valid_pairs, requested_count, random_seed=random_seed)
    actual_count = len(selected)
    unique_seqs = len(set(p["sequence"] for p in selected))

    stats = {
        "requested_count": requested_count,
        "actual_count": actual_count,
        "unique_sequences": unique_seqs,
        "output_path": str(output_path)
    }

    if actual_count == 0:
        fig, ax = plt.subplots(figsize=(8, 2))
        msg = f"{title_prefix}: 0 pairs requested" if requested_count == 0 else f"{title_prefix}: No valid pairs available"
        ax.text(0.5, 0.5, msg, ha='center', va='center', fontsize=12)
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        return selected, stats

    fig, axes = plt.subplots(actual_count, 2, figsize=(12, 3.2 * actual_count), squeeze=False)

    for idx, pair in enumerate(selected):
        b_path = pair["blur_path"]
        s_path = pair["sharp_path"]
        seq_id = pair["sequence"]
        fname = pair["filename"]
        dim_str = f"{pair.get('width', '?')}x{pair.get('height', '?')}"

        try:
            with Image.open(b_path) as b_img:
                axes[idx, 0].imshow(b_img)
        except Exception as e:
            axes[idx, 0].text(0.5, 0.5, f"Error: {e}", ha='center')
        axes[idx, 0].set_title(f"Pair #{idx+1} [Blur] - Seq {seq_id} | {fname} ({dim_str})", fontsize=10)
        axes[idx, 0].axis('off')

        try:
            with Image.open(s_path) as s_img:
                axes[idx, 1].imshow(s_img)
        except Exception as e:
            axes[idx, 1].text(0.5, 0.5, f"Error: {e}", ha='center')
        axes[idx, 1].set_title(f"Pair #{idx+1} [Sharp Target] - Seq {seq_id} | {fname} ({dim_str})", fontsize=10)
        axes[idx, 1].axis('off')

    plt.suptitle(f"{title_prefix} Blur vs Sharp Ground Truth Comparisons (Count: {actual_count})", fontsize=14, y=1.002)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return selected, stats


def plot_consecutive_frames(
    candidate_pairs: List[Dict[str, Any]],
    frames_per_sequence: int,
    output_path: Path,
    target_sequences: Optional[List[str]] = None,
    random_seed: int = 42
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Select consecutive frames in chronological order from sequence(s) and display side-by-side.
    Saves figure to output_path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if frames_per_sequence <= 0 or not candidate_pairs:
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.text(0.5, 0.5, "Consecutive inspection: 0 frames requested or no pairs available", ha='center', va='center')
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        return [], {"requested_count": frames_per_sequence, "actual_count": 0}

    # Group candidate pairs by sequence
    seq_groups: Dict[str, List[Dict[str, Any]]] = {}
    for p in candidate_pairs:
        seq = p["sequence"]
        if seq not in seq_groups:
            seq_groups[seq] = []
        seq_groups[seq].append(p)

    available_seqs = sorted(list(seq_groups.keys()))
    if not available_seqs:
        return [], {"requested_count": frames_per_sequence, "actual_count": 0}

    if not target_sequences:
        target_sequences = [available_seqs[0]]

    all_selected_consecutive = []
    for seq_id in target_sequences:
        pairs = seq_groups.get(seq_id, [])
        def extract_num(p):
            stem = Path(p["filename"]).stem
            try:
                return int(stem)
            except ValueError:
                return stem

        sorted_pairs = sorted(pairs, key=extract_num)

        if len(sorted_pairs) >= frames_per_sequence:
            # If total available is smaller than 10 + frames_per_sequence, start at index 0
            if len(sorted_pairs) < 10 + frames_per_sequence:
                start_idx = 0
            else:
                start_idx = 10
            window = sorted_pairs[start_idx : start_idx + frames_per_sequence]
            all_selected_consecutive.extend(window)
        else:
            all_selected_consecutive.extend(sorted_pairs)

    total_rows = len(all_selected_consecutive)
    if total_rows == 0:
        return [], {"requested_count": frames_per_sequence, "actual_count": 0}

    fig, axes = plt.subplots(total_rows, 2, figsize=(12, 3.2 * total_rows), squeeze=False)

    for idx, pair in enumerate(all_selected_consecutive):
        b_path = pair["blur_path"]
        s_path = pair["sharp_path"]
        seq_id = pair["sequence"]
        fname = pair["filename"]

        with Image.open(b_path) as b_img:
            axes[idx, 0].imshow(b_img)
        axes[idx, 0].set_title(f"Seq {seq_id} | Blur Frame: {fname}", fontsize=10)
        axes[idx, 0].axis('off')

        with Image.open(s_path) as s_img:
            axes[idx, 1].imshow(s_img)
        axes[idx, 1].set_title(f"Seq {seq_id} | Sharp Frame: {fname}", fontsize=10)
        axes[idx, 1].axis('off')

    plt.suptitle(f"Consecutive Frame Temporal Inspection ({frames_per_sequence} frames/seq)", fontsize=14, y=1.002)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    stats = {
        "requested_count": frames_per_sequence,
        "actual_count": total_rows,
        "sequences": [p["sequence"] for p in all_selected_consecutive],
        "output_path": str(output_path)
    }
    return all_selected_consecutive, stats


# Retain previous EDA distribution plotting functions for compatibility
def plot_dataset_distribution(df_inv: pd.DataFrame, output_dir: Path):
    if df_inv.empty: return
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_inv.groupby('dataset')['image_count'].sum().reset_index(), x='dataset', y='image_count')
    plt.title("Total Images per Dataset")
    plt.savefig(output_dir / "dataset_distribution.png")
    plt.close()


def plot_image_dimensions(df_stats: pd.DataFrame, output_dir: Path):
    if df_stats.empty: return
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_stats, x='width', y='height', hue='dataset')
    plt.title("Image Dimensions")
    plt.savefig(output_dir / "image_dimensions.png")
    plt.close()


def plot_brightness(df_stats: pd.DataFrame, output_dir: Path):
    if df_stats.empty: return
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df_stats, x='mean_brightness', hue='dataset', kde=True)
    plt.title("Brightness Distribution")
    plt.savefig(output_dir / "brightness_distribution.png")
    plt.close()


def plot_contrast(df_stats: pd.DataFrame, output_dir: Path):
    if df_stats.empty: return
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df_stats, x='std_contrast', hue='dataset', kde=True)
    plt.title("Contrast Distribution")
    plt.savefig(output_dir / "contrast_distribution.png")
    plt.close()


def plot_sharpness(df_stats: pd.DataFrame, output_dir: Path):
    if df_stats.empty: return
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_stats, x='dataset', y='sharpness_variance')
    plt.title("Sharpness Distribution")
    plt.savefig(output_dir / "sharpness_distribution.png")
    plt.close()
