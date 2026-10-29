import pytest
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import yaml

from src.data.config_utils import get_project_root, validate_config
from src.data.pairing_verification import verify_split_pairing, get_candidate_pairs
from src.data.image_validation import (
    validate_single_image,
    validate_split_integrity,
    check_matched_pairs_dimensions
)
from src.data.visualization import (
    sample_diverse_pairs,
    plot_blur_sharp_pairs,
    plot_consecutive_frames
)


def create_synthetic_image(path: Path, width: int = 1280, height: int = 720, color=(100, 150, 200), mode="RGB"):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new(mode, (width, height), color=color)
    img.save(path)


# Test 1: Correctly matched blur/sharp pairs
def test_correctly_matched_pairs(tmp_path):
    b_dir = tmp_path / "train_blur" / "000"
    s_dir = tmp_path / "train_sharp" / "000"
    for i in range(3):
        fname = f"{i:08d}.png"
        create_synthetic_image(b_dir / fname)
        create_synthetic_image(s_dir / fname)

    df_summary, df_issues = verify_split_pairing("train", tmp_path / "train_blur", tmp_path / "train_sharp")
    assert len(df_summary) == 1
    assert df_summary["pairing_status"].iloc[0] == "PERFECT_MATCH"
    assert df_summary["matched_frame_count"].iloc[0] == 3
    assert len(df_issues) == 0


# Test 2: Missing sharp frame
def test_missing_sharp_frame(tmp_path):
    b_dir = tmp_path / "train_blur" / "000"
    s_dir = tmp_path / "train_sharp" / "000"
    create_synthetic_image(b_dir / "00000000.png")
    create_synthetic_image(s_dir / "00000000.png")
    create_synthetic_image(b_dir / "00000001.png")  # Missing in sharp

    df_summary, df_issues = verify_split_pairing("train", tmp_path / "train_blur", tmp_path / "train_sharp")
    assert df_summary["pairing_status"].iloc[0] == "PARTIAL_MATCH"
    assert df_summary["missing_sharp_frames"].iloc[0] == 1
    assert any(issue["issue_type"] == "missing_sharp_frame" for _, issue in df_issues.iterrows())


# Test 3: Missing blur frame in existing sequence
def test_missing_blur_frame(tmp_path):
    b_dir = tmp_path / "train_blur" / "000"
    s_dir = tmp_path / "train_sharp" / "000"
    create_synthetic_image(b_dir / "00000000.png")
    create_synthetic_image(s_dir / "00000000.png")
    create_synthetic_image(s_dir / "00000001.png")  # Missing in blur

    df_summary, df_issues = verify_split_pairing("train", tmp_path / "train_blur", tmp_path / "train_sharp")
    assert df_summary["pairing_status"].iloc[0] == "PARTIAL_MATCH"
    assert df_summary["missing_blur_frames"].iloc[0] == 1
    assert any(issue["issue_type"] == "missing_blur_frame" for _, issue in df_issues.iterrows())


# Test 4: Sequence present only in one directory
def test_sequence_missing_in_one_directory(tmp_path):
    b_dir = tmp_path / "train_blur" / "000"
    create_synthetic_image(b_dir / "00000000.png")
    (tmp_path / "train_sharp").mkdir(parents=True, exist_ok=True)  # empty sharp directory

    df_summary, df_issues = verify_split_pairing("train", tmp_path / "train_blur", tmp_path / "train_sharp")
    assert len(df_summary) == 1
    assert df_summary["pairing_status"].iloc[0] == "MISSING_SEQUENCE"
    assert any(issue["issue_type"] == "missing_sequence_in_sharp" for _, issue in df_issues.iterrows())


# Test 5: Dimension mismatch between a matched pair
def test_dimension_mismatch(tmp_path):
    b_file = tmp_path / "train_blur" / "000" / "00000000.png"
    s_file = tmp_path / "train_sharp" / "000" / "00000000.png"
    create_synthetic_image(b_file, width=1280, height=720)
    create_synthetic_image(s_file, width=640, height=480)

    pairs = [{
        "dataset": "train", "sequence": "000", "filename": "00000000.png",
        "blur_path": b_file, "sharp_path": s_file
    }]
    df_mismatches, valid_pairs = check_matched_pairs_dimensions(pairs, pd.DataFrame())
    assert len(df_mismatches) == 1
    assert len(valid_pairs) == 0
    assert "Dimension mismatch" in df_mismatches["description"].iloc[0]


# Test 6: Corrupted or unreadable image
def test_corrupted_image(tmp_path):
    bad_file = tmp_path / "bad.png"
    bad_file.write_text("NOT_A_VALID_IMAGE_HEADER_OR_CONTENT")

    res = validate_single_image(bad_file)
    assert not res["readable"]
    assert res["status"] == "CORRUPTED"


# Test 7: Empty sequence directory
def test_empty_sequence_directory(tmp_path):
    (tmp_path / "train_blur" / "000").mkdir(parents=True, exist_ok=True)
    (tmp_path / "train_sharp" / "000").mkdir(parents=True, exist_ok=True)

    df_summary, df_issues = verify_split_pairing("train", tmp_path / "train_blur", tmp_path / "train_sharp")
    assert df_summary["pairing_status"].iloc[0] == "EMPTY_SEQUENCE"
    assert any(issue["issue_type"] == "empty_sequence_directory" for _, issue in df_issues.iterrows())


# Test 8: Unequal frame counts
def test_unequal_frame_counts(tmp_path):
    b_dir = tmp_path / "train_blur" / "000"
    s_dir = tmp_path / "train_sharp" / "000"
    create_synthetic_image(b_dir / "00000000.png")
    create_synthetic_image(b_dir / "00000001.png")
    create_synthetic_image(s_dir / "00000000.png")

    df_summary, _ = verify_split_pairing("train", tmp_path / "train_blur", tmp_path / "train_sharp")
    assert df_summary["blur_frame_count"].iloc[0] == 2
    assert df_summary["sharp_frame_count"].iloc[0] == 1


# Test 9: Reproducible sampling with a fixed random seed
def test_reproducible_sampling():
    pairs = [
        {"dataset": "train", "sequence": f"{i:03d}", "filename": "00000000.png", "blur_path": Path(f"b_{i}"), "sharp_path": Path(f"s_{i}")}
        for i in range(20)
    ]
    sample1 = sample_diverse_pairs(pairs, requested_count=5, random_seed=42)
    sample2 = sample_diverse_pairs(pairs, requested_count=5, random_seed=42)
    assert [p["sequence"] for p in sample1] == [p["sequence"] for p in sample2]


# Test 10: Fewer available pairs than requested display count
def test_fewer_available_pairs_than_requested():
    pairs = [
        {"dataset": "train", "sequence": "000", "filename": "00000000.png", "blur_path": Path("b"), "sharp_path": Path("s")}
    ]
    selected = sample_diverse_pairs(pairs, requested_count=10, random_seed=42)
    assert len(selected) == 1


# Test 11: Display count of zero
def test_display_count_zero(tmp_path):
    pairs = [
        {"dataset": "train", "sequence": "000", "filename": "00000000.png", "blur_path": Path("b"), "sharp_path": Path("s")}
    ]
    selected, stats = plot_blur_sharp_pairs(pairs, requested_count=0, output_path=tmp_path / "zero.png")
    assert len(selected) == 0
    assert stats["actual_count"] == 0
    assert (tmp_path / "zero.png").exists()


# Test 12: No valid pairs available for visualization
def test_no_valid_pairs_available(tmp_path):
    selected, stats = plot_blur_sharp_pairs([], requested_count=5, output_path=tmp_path / "none.png")
    assert len(selected) == 0
    assert (tmp_path / "none.png").exists()


# Test 13: Empty issue CSVs retaining column headers
def test_empty_issue_csv_headers(tmp_path):
    b_dir = tmp_path / "train_blur" / "000"
    s_dir = tmp_path / "train_sharp" / "000"
    create_synthetic_image(b_dir / "00000000.png")
    create_synthetic_image(s_dir / "00000000.png")

    _, df_issues = verify_split_pairing("train", tmp_path / "train_blur", tmp_path / "train_sharp")
    assert df_issues.empty
    csv_file = tmp_path / "issues.csv"
    df_issues.to_csv(csv_file, index=False)
    content = csv_file.read_text(encoding="utf-8")
    assert "dataset,issue_type,sequence,filename,affected_path,description" in content


# Test 14: Validation continuing after an individual image fails
def test_validation_continues_after_failure(tmp_path):
    d = tmp_path / "seq"
    d.mkdir(parents=True, exist_ok=True)
    create_synthetic_image(d / "00000000.png")
    (d / "00000001.png").write_text("CORRUPTED")
    create_synthetic_image(d / "00000002.png")

    results = validate_split_integrity("test_split", tmp_path)
    assert len(results) == 3
    assert results[0]["readable"] is True
    assert results[1]["readable"] is False
    assert results[2]["readable"] is True


# Test 15: Correct consecutive-frame selection and ordering
def test_consecutive_frame_selection(tmp_path):
    pairs = [
        {"dataset": "train", "sequence": "000", "filename": f"{i:08d}.png", "blur_path": tmp_path / f"b_{i}.png", "sharp_path": tmp_path / f"s_{i}.png"}
        for i in [2, 0, 1, 3]  # unordered
    ]
    for p in pairs:
        create_synthetic_image(p["blur_path"])
        create_synthetic_image(p["sharp_path"])

    selected, stats = plot_consecutive_frames(pairs, frames_per_sequence=3, output_path=tmp_path / "consec.png", target_sequences=["000"])
    assert len(selected) == 3
    # Check that filenames are strictly ordered
    fnames = [p["filename"] for p in selected]
    assert fnames == ["00000000.png", "00000001.png", "00000002.png"]


# Test 16: Configuration values being respected
def test_config_validation():
    with pytest.raises(ValueError, match="train_pair_count must be non-negative"):
        validate_config({"visualization": {"train_pair_count": -1}}, Path("."))

    with pytest.raises(ValueError, match="expected_width must be positive"):
        validate_config({"validation": {"expected_width": -10}}, Path("."))
