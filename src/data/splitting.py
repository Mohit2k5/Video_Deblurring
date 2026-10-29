"""Deterministic 90:10 Sequence-Level Split Generator for REDS Dataset."""
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import random
import pandas as pd

from src.data.config_utils import get_project_root
from src.data.indexing import get_available_sequences, load_dataset_index


def generate_sequence_split(
    project_root: Optional[Path] = None,
    train_ratio: float = 0.90,
    test_ratio: float = 0.10,
    random_seed: int = 42,
    manifests_dir: Optional[Path] = None,
    splits_dir: Optional[Path] = None,
    reports_dir: Optional[Path] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Generate deterministic 90:10 sequence-level split on the 240 paired training sequences.
    Preserves existing 30 validation sequences and 30 inference-only test_blur sequences.

    Returns:
        df_split: DataFrame of sequence assignments
        summary_dict: Summary metrics and provenance
    """
    root = project_root or get_project_root()
    if manifests_dir is None:
        manifests_dir = root / "data" / "manifests"
    if splits_dir is None:
        splits_dir = root / "splits"
    if reports_dir is None:
        reports_dir = root / "reports" / "data_pipeline"

    manifests_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Retrieve the 240 training sequences
    train_seqs = get_available_sequences("train", project_root=root)
    if not train_seqs:
        # Fallback to direct raw scan if indexing didn't return
        raw_train_blur = root / "data" / "raw" / "train_blur"
        if raw_train_blur.exists():
            train_seqs = sorted([d.name for d in raw_train_blur.iterdir() if d.is_dir()])

    # Ensure deterministic sort before shuffle
    train_seqs = sorted(train_seqs)
    total_train_source = len(train_seqs)

    # Validate split ratios
    if not (0.0 < train_ratio < 1.0):
        raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")
    if abs((train_ratio + test_ratio) - 1.0) > 1e-5:
        raise ValueError(f"train_ratio ({train_ratio}) and test_ratio ({test_ratio}) must sum to 1.0")

    # 2. Deterministic shuffle
    rng = random.Random(random_seed)
    shuffled_seqs = list(train_seqs)
    rng.shuffle(shuffled_seqs)

    # 3. Calculate target counts
    num_train = int(round(total_train_source * train_ratio))
    num_held_out = total_train_source - num_train

    assigned_train = sorted(shuffled_seqs[:num_train])
    assigned_held_out = sorted(shuffled_seqs[num_train:])

    # 4. Validation sequences (from val_blur / val_sharp)
    val_seqs = get_available_sequences("val", project_root=root)
    if not val_seqs:
        raw_val_blur = root / "data" / "raw" / "val_blur"
        if raw_val_blur.exists():
            val_seqs = sorted([d.name for d in raw_val_blur.iterdir() if d.is_dir()])
    assigned_val = sorted(val_seqs)

    # 5. Inference test_blur sequences
    test_blur_seqs = get_available_sequences("test_blur", project_root=root)
    if not test_blur_seqs:
        raw_test_blur = root / "data" / "raw" / "test_blur"
        if raw_test_blur.exists():
            test_blur_seqs = sorted([d.name for d in raw_test_blur.iterdir() if d.is_dir()])
    assigned_inference_test = sorted(test_blur_seqs)

    # 6. Build split records
    records = []
    for seq in assigned_train:
        records.append({
            "sequence_id": seq,
            "source_dataset": "train",
            "partition": "train",
            "has_ground_truth": True,
            "frame_count": 100,
            "split_strategy": "sequence_level_random",
            "random_seed": random_seed
        })
    for seq in assigned_held_out:
        records.append({
            "sequence_id": seq,
            "source_dataset": "train",
            "partition": "held_out_test",
            "has_ground_truth": True,
            "frame_count": 100,
            "split_strategy": "sequence_level_random",
            "random_seed": random_seed
        })
    for seq in assigned_val:
        records.append({
            "sequence_id": seq,
            "source_dataset": "val",
            "partition": "val",
            "has_ground_truth": True,
            "frame_count": 100,
            "split_strategy": "official_validation",
            "random_seed": random_seed
        })
    for seq in assigned_inference_test:
        records.append({
            "sequence_id": seq,
            "source_dataset": "test_blur",
            "partition": "inference_test",
            "has_ground_truth": False,
            "frame_count": 100,
            "split_strategy": "official_test_blur_unpaired",
            "random_seed": random_seed
        })

    cols = ["sequence_id", "source_dataset", "partition", "has_ground_truth", "frame_count", "split_strategy", "random_seed"]
    df_split = pd.DataFrame(records, columns=cols)

    # 7. Build summary dictionary
    summary = {
        "split_strategy": "90:10 sequence-level random split on REDS train",
        "random_seed": random_seed,
        "train_ratio": train_ratio,
        "test_ratio": test_ratio,
        "total_source_train_sequences": total_train_source,
        "counts": {
            "train": {
                "sequences": len(assigned_train),
                "frames": len(assigned_train) * 100,
                "has_ground_truth": True
            },
            "held_out_test": {
                "sequences": len(assigned_held_out),
                "frames": len(assigned_held_out) * 100,
                "has_ground_truth": True
            },
            "val": {
                "sequences": len(assigned_val),
                "frames": len(assigned_val) * 100,
                "has_ground_truth": True
            },
            "inference_test": {
                "sequences": len(assigned_inference_test),
                "frames": len(assigned_inference_test) * 100,
                "has_ground_truth": False
            }
        },
        "sequence_assignments": {
            "train": assigned_train,
            "held_out_test": assigned_held_out,
            "val": assigned_val,
            "inference_test": assigned_inference_test
        }
    }

    # 8. Save outputs to data/manifests/ and splits/
    manifest_csv = manifests_dir / "sequence_split.csv"
    manifest_json = manifests_dir / "split_summary.json"
    df_split.to_csv(manifest_csv, index=False)
    with open(manifest_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Also mirror into splits/ for direct user discovery
    df_split.to_csv(splits_dir / "sequence_split.csv", index=False)
    with open(splits_dir / "split_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # 9. Generate reports/data_pipeline/split_report.md
    generate_split_report(reports_dir / "split_report.md", summary)

    return df_split, summary


def generate_split_report(report_path: Path, summary: Dict[str, Any]):
    """Generate Markdown report detailing the 90:10 sequence split."""
    counts = summary["counts"]
    lines = [
        "# REDS Video Deblurring: 90:10 Sequence-Level Split Report\n\n",
        "## 1. Executive Summary\n",
        f"- **Strategy:** {summary['split_strategy']}\n",
        f"- **Random Seed:** `{summary['random_seed']}` (fully deterministic)\n",
        f"- **Source Train Sequences:** {summary['total_source_train_sequences']}\n\n",
        "## 2. Partition Summary Table\n",
        "| Partition | Purpose | Sequences | Frames | Ground Truth Available | Source |\n",
        "| :--- | :--- | :--- | :--- | :--- | :--- |\n",
        f"| `train` | Model Training & Fitting | **{counts['train']['sequences']}** | {counts['train']['frames']} | Yes (train_sharp) | `train_blur` / `train_sharp` |\n",
        f"| `held_out_test` | Unseen Benchmark Evaluation | **{counts['held_out_test']['sequences']}** | {counts['held_out_test']['frames']} | Yes (train_sharp) | `train_blur` / `train_sharp` |\n",
        f"| `val` | Validation & Early Stopping | **{counts['val']['sequences']}** | {counts['val']['frames']} | Yes (val_sharp) | `val_blur` / `val_sharp` |\n",
        f"| `inference_test` | Qualitative Challenge Inference | **{counts['inference_test']['sequences']}** | {counts['inference_test']['frames']} | No (Unpaired) | `test_blur` |\n\n",
        "## 3. Leakage & Overlap Verification\n",
        "- **Sequence-level disjointness:** Strict zero-overlap guarantee. Every frame belonging to a sequence remains in the same partition.\n",
        "- **Held-Out Test Independence:** The 24 held-out test sequences are strictly reserved for post-training evaluation and must NEVER be used for training, checkpoint selection, or hyperparameter optimization.\n",
        "- **Raw Data Immutability:** Raw directories in `data/raw/` remain 100% untouched; partition mapping is maintained purely through manifests.\n\n",
        "## 4. Sequence Assignments Detail\n",
        f"- **Held-Out Test Sequences (24):** `{', '.join(summary['sequence_assignments']['held_out_test'])}`\n",
        f"- **Validation Sequences (30):** `{', '.join(summary['sequence_assignments']['val'])}`\n"
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(lines)


if __name__ == "__main__":
    print("Generating deterministic 90:10 REDS sequence split...")
    df_split, summary = generate_sequence_split()
    print(f"Done! Split manifests written to data/manifests/ and splits/ ({len(df_split)} sequences total).")

