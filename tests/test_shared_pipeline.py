"""Tests for shared dataset pipeline, sequence splitting, indexing, and paired transforms."""
import pytest
from pathlib import Path
import json
import pandas as pd
import numpy as np
from PIL import Image

from src.data.config_utils import get_project_root
from src.data.indexing import load_dataset_index, get_available_sequences
from src.data.splitting import generate_sequence_split
from src.preprocessing.temporal_windows import build_temporal_indices
from src.preprocessing.paired_transforms import (
    paired_random_crop,
    paired_horizontal_flip,
    paired_vertical_flip,
    paired_temporal_reverse,
    PairedTransformPipeline
)
from src.data.dataset import REDSVideoDeblurDataset


def create_synthetic_frame(path: Path, width=64, height=64, color=(128, 128, 128)):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (width, height), color=color)
    img.save(path)


def create_coordinate_coded_image(width: int = 128, height: int = 128, offset: int = 0) -> np.ndarray:
    """
    Generate an image where each pixel encodes its spatial coordinate (x, y).
    Channel 0: x coordinate mod 256
    Channel 1: y coordinate mod 256
    Channel 2: (x + y + offset) mod 256
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)
    x = np.arange(width, dtype=np.uint16)
    y = np.arange(height, dtype=np.uint16)
    xx, yy = np.meshgrid(x, y)
    img[:, :, 0] = (xx % 256).astype(np.uint8)
    img[:, :, 1] = (yy % 256).astype(np.uint8)
    img[:, :, 2] = ((xx + yy + offset) % 256).astype(np.uint8)
    return img


# 1. Correct split sequence counts
def test_generate_sequence_split_counts(tmp_path):
    raw = tmp_path / "data" / "raw"
    for i in range(240):
        create_synthetic_frame(raw / "train_blur" / f"{i:03d}" / "00000000.png")
        create_synthetic_frame(raw / "train_sharp" / f"{i:03d}" / "00000000.png")
    for i in range(30):
        create_synthetic_frame(raw / "val_blur" / f"{i:03d}" / "00000000.png")
        create_synthetic_frame(raw / "val_sharp" / f"{i:03d}" / "00000000.png")
        create_synthetic_frame(raw / "test_blur" / f"{i:03d}" / "00000000.png")

    manifests = tmp_path / "data" / "manifests"
    splits = tmp_path / "splits"
    reports = tmp_path / "reports" / "data_pipeline"

    df_split, summary = generate_sequence_split(
        project_root=tmp_path,
        train_ratio=0.90,
        test_ratio=0.10,
        random_seed=42,
        manifests_dir=manifests,
        splits_dir=splits,
        reports_dir=reports
    )

    counts = summary["counts"]
    assert counts["train"]["sequences"] == 216
    assert counts["held_out_test"]["sequences"] == 24
    assert counts["val"]["sequences"] == 30
    assert counts["inference_test"]["sequences"] == 30
    assert counts["train"]["frames"] == 216 * 100
    assert counts["held_out_test"]["frames"] == 24 * 100


# 2. Deterministic split generation with the same seed
def test_split_reproducibility(tmp_path):
    raw = tmp_path / "data" / "raw"
    for i in range(20):
        create_synthetic_frame(raw / "train_blur" / f"{i:03d}" / "00000000.png")

    _, s1 = generate_sequence_split(project_root=tmp_path, train_ratio=0.8, test_ratio=0.2, random_seed=999,
                                   manifests_dir=tmp_path / "m1", splits_dir=tmp_path / "s1", reports_dir=tmp_path / "r1")
    _, s2 = generate_sequence_split(project_root=tmp_path, train_ratio=0.8, test_ratio=0.2, random_seed=999,
                                   manifests_dir=tmp_path / "m2", splits_dir=tmp_path / "s2", reports_dir=tmp_path / "r2")

    assert s1["sequence_assignments"]["train"] == s2["sequence_assignments"]["train"]
    assert s1["sequence_assignments"]["held_out_test"] == s2["sequence_assignments"]["held_out_test"]


# 3. No sequence overlap between training and held-out test partitions
def test_no_sequence_leakage(tmp_path):
    raw = tmp_path / "data" / "raw"
    for i in range(10):
        create_synthetic_frame(raw / "train_blur" / f"{i:03d}" / "00000000.png")

    df_split, _ = generate_sequence_split(project_root=tmp_path, train_ratio=0.5, test_ratio=0.5, random_seed=42,
                                         manifests_dir=tmp_path / "m", splits_dir=tmp_path / "s", reports_dir=tmp_path / "r")

    train_set = set(df_split[df_split["partition"] == "train"]["sequence_id"])
    test_set = set(df_split[df_split["partition"] == "held_out_test"]["sequence_id"])

    assert len(train_set.intersection(test_set)) == 0


# 4. Temporal windows never cross sequence boundaries
def test_temporal_windows_boundary_guarantee():
    seq_len = 10
    windows = build_temporal_indices(sequence_length=seq_len, window_size=5, boundary_policy="replicate")
    assert len(windows) == seq_len
    assert windows[0] == [0, 0, 0, 1, 2]
    assert windows[9] == [7, 8, 9, 9, 9]

    for win in windows:
        for idx in win:
            assert 0 <= idx < seq_len


def test_temporal_windows_truncate_policy():
    seq_len = 10
    windows = build_temporal_indices(sequence_length=seq_len, window_size=5, boundary_policy="truncate")
    assert len(windows) == 6
    assert windows[0] == [0, 1, 2, 3, 4]


# 5. Strengthened Paired crop spatial coordinate alignment test
def test_paired_crop_spatial_alignment():
    # Use coordinate-coded images where pixel value directly encodes (x, y) position
    arr_b = create_coordinate_coded_image(width=100, height=100, offset=0)
    arr_s = create_coordinate_coded_image(width=100, height=100, offset=10)

    img_b = Image.fromarray(arr_b)
    img_s = Image.fromarray(arr_s)

    # Crop to (32, 40)
    cropped_b, cropped_s = paired_random_crop([img_b, img_b], img_s, crop_size=(32, 40), seed=42)
    assert len(cropped_b) == 2
    assert cropped_b[0].size == (40, 32)
    assert cropped_s.size == (40, 32)

    # Convert back to arrays to verify that identical coordinates were cropped
    res_b0 = np.array(cropped_b[0])
    res_b1 = np.array(cropped_b[1])
    res_s = np.array(cropped_s)

    # Channel 0 (x-coordinate) and Channel 1 (y-coordinate) MUST BE IDENTICAL
    np.testing.assert_array_equal(res_b0[:, :, 0], res_s[:, :, 0], err_msg="X-coordinates diverged between blur and sharp!")
    np.testing.assert_array_equal(res_b0[:, :, 1], res_s[:, :, 1], err_msg="Y-coordinates diverged between blur and sharp!")
    np.testing.assert_array_equal(res_b0[:, :, :2], res_b1[:, :, :2], err_msg="Frames within blur window diverged!")

    # Verify offset channel preserved relationship (target = blur + 10 mod 256)
    expected_ch2 = ((res_b0[:, :, 2].astype(np.uint16) + 10) % 256).astype(np.uint8)
    np.testing.assert_array_equal(res_s[:, :, 2], expected_ch2, err_msg="Feature alignment failed across crop!")


def test_paired_geometric_flips_preserve_correspondence():
    arr_b = create_coordinate_coded_image(width=64, height=64, offset=0)
    arr_s = create_coordinate_coded_image(width=64, height=64, offset=5)

    img_b = Image.fromarray(arr_b)
    img_s = Image.fromarray(arr_s)

    # Test horizontal flip
    fb_list, fs = paired_horizontal_flip([img_b], img_s)
    fb_arr = np.array(fb_list[0])
    fs_arr = np.array(fs)
    np.testing.assert_array_equal(fb_arr[:, :, 0], fs_arr[:, :, 0], err_msg="H-flip broke X coordinate correspondence!")
    np.testing.assert_array_equal(fb_arr[:, :, 1], fs_arr[:, :, 1], err_msg="H-flip broke Y coordinate correspondence!")

    # Test vertical flip
    vb_list, vs = paired_vertical_flip([img_b], img_s)
    vb_arr = np.array(vb_list[0])
    vs_arr = np.array(vs)
    np.testing.assert_array_equal(vb_arr[:, :, 0], vs_arr[:, :, 0], err_msg="V-flip broke X coordinate correspondence!")
    np.testing.assert_array_equal(vb_arr[:, :, 1], vs_arr[:, :, 1], err_msg="V-flip broke Y coordinate correspondence!")


# 6. Paired transforms pipeline reproducibility
def test_paired_transforms_reproducibility():
    img_b = Image.new("RGB", (50, 50), color=(10, 20, 30))
    img_s = Image.new("RGB", (50, 50), color=(40, 50, 60))

    pipe1 = PairedTransformPipeline(crop_size=(20, 20), flip_h=True, flip_v=True, temporal_reverse=True, seed=42)
    pipe2 = PairedTransformPipeline(crop_size=(20, 20), flip_h=True, flip_v=True, temporal_reverse=True, seed=42)

    res_b1, res_s1 = pipe1([img_b, img_b], img_s)
    res_b2, res_s2 = pipe2([img_b, img_b], img_s)

    assert np.array_equal(np.array(res_b1[0]), np.array(res_b2[0]))
    assert np.array_equal(np.array(res_s1), np.array(res_s2))


# 7. Dataset loader indexing and reading
def test_dataset_loader_synthetic(tmp_path):
    raw = tmp_path / "data" / "raw"
    seq_name = "000"
    for i in range(5):
        fname = f"{i:08d}.png"
        create_synthetic_frame(raw / "train_blur" / seq_name / fname)
        create_synthetic_frame(raw / "train_sharp" / seq_name / fname)

    manifest_dir = tmp_path / "data" / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    df_manifest = pd.DataFrame([{
        "sequence_id": seq_name,
        "source_dataset": "train",
        "partition": "train",
        "has_ground_truth": True,
        "frame_count": 5
    }])
    df_manifest.to_csv(manifest_dir / "sequence_split.csv", index=False)

    ds = REDSVideoDeblurDataset(
        partition="train",
        project_root=tmp_path,
        window_size=3,
        boundary_policy="replicate"
    )

    assert len(ds) == 5
    sample = ds[0]
    assert len(sample["blur_frames"]) == 3
    assert "sharp_frame" in sample
    assert sample["sequence_id"] == "000"
    assert sample["center_filename"] == "00000000.png"


# 8. Fixed split integrity and composite-key verification
def test_fixed_reds_split_integrity():
    """Stage 7: Strict verification of fixed REDS sequence-level split integrity.
    Verifies:
    1. Composite sequence identity (source_dataset, sequence_id)
    2. Exact counts: 216 train, 24 held-out test, 30 val, 30 inference test = 300 total
    3. Mutual exclusivity (0 overlap) across all partitions
    4. Exact parity between data/manifests/sequence_split.csv and splits/sequence_split.csv
    """
    root = get_project_root()
    m_path = root / "data" / "manifests" / "sequence_split.csv"
    s_path = root / "splits" / "sequence_split.csv"

    assert m_path.exists(), f"Manifest missing at {m_path}"
    assert s_path.exists(), f"Split missing at {s_path}"

    df_m = pd.read_csv(m_path, dtype={"sequence_id": str})
    df_s = pd.read_csv(s_path, dtype={"sequence_id": str})

    assert len(df_m) == 300
    assert len(df_s) == 300
    assert df_m.equals(df_s), "data/manifests/sequence_split.csv and splits/sequence_split.csv do not match"

    # Verify composite identities
    df_m["composite_id"] = df_m["source_dataset"] + "_" + df_m["sequence_id"]
    train_ids = set(df_m[df_m["partition"] == "train"]["composite_id"])
    held_out_ids = set(df_m[df_m["partition"] == "held_out_test"]["composite_id"])
    val_ids = set(df_m[df_m["partition"] == "val"]["composite_id"])
    test_ids = set(df_m[df_m["partition"] == "inference_test"]["composite_id"])

    assert len(train_ids) == 216
    assert len(held_out_ids) == 24
    assert len(val_ids) == 30
    assert len(test_ids) == 30

    # Strict disjointness
    assert len(train_ids.intersection(held_out_ids)) == 0
    assert len(train_ids.intersection(val_ids)) == 0
    assert len(train_ids.intersection(test_ids)) == 0
    assert len(held_out_ids.intersection(val_ids)) == 0
    assert len(held_out_ids.intersection(test_ids)) == 0
    assert len(val_ids.intersection(test_ids)) == 0

