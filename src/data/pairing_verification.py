"""Sequence and frame pairing analysis module for REDS dataset."""
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd


def verify_split_pairing(
    split_name: str,
    blur_dir: Path,
    sharp_dir: Path,
    supported_exts: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform sequence-level and frame-level pairing checks between blurred and sharp directories.

    Returns:
        df_summary: DataFrame with sequence-level metrics and pairing status
        df_issues: DataFrame detailing any pairing discrepancies
    """
    if supported_exts is None:
        supported_exts = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
    supported_exts = [e.lower() for e in supported_exts]

    summary_records = []
    issues_records = []

    # Check directory existence
    blur_exists = blur_dir.exists() and blur_dir.is_dir()
    sharp_exists = sharp_dir.exists() and sharp_dir.is_dir()

    if not blur_exists:
        issues_records.append({
            "dataset": split_name,
            "issue_type": "missing_directory",
            "sequence": "ALL",
            "filename": "",
            "affected_path": str(blur_dir),
            "description": f"Blurred directory does not exist: {blur_dir}"
        })
    if not sharp_exists:
        issues_records.append({
            "dataset": split_name,
            "issue_type": "missing_directory",
            "sequence": "ALL",
            "filename": "",
            "affected_path": str(sharp_dir),
            "description": f"Sharp directory does not exist: {sharp_dir}"
        })

    # Discover sequence directories (empty dict if dir doesn't exist)
    blur_seq_dirs = {d.name: d for d in blur_dir.iterdir() if d.is_dir()} if blur_exists else {}
    sharp_seq_dirs = {d.name: d for d in sharp_dir.iterdir() if d.is_dir()} if sharp_exists else {}

    all_seq_ids = sorted(list(set(blur_seq_dirs.keys()) | set(sharp_seq_dirs.keys())))

    for seq_id in all_seq_ids:
        in_blur = seq_id in blur_seq_dirs
        in_sharp = seq_id in sharp_seq_dirs

        blur_path = blur_seq_dirs.get(seq_id)
        sharp_path = sharp_seq_dirs.get(seq_id)

        blur_frames: Dict[str, Path] = {}
        sharp_frames: Dict[str, Path] = {}

        if in_blur and blur_path:
            for item in blur_path.iterdir():
                if item.is_file() and item.suffix.lower() in supported_exts:
                    blur_frames[item.name] = item
        else:
            issues_records.append({
                "dataset": split_name,
                "issue_type": "missing_sequence_in_blur",
                "sequence": seq_id,
                "filename": "",
                "affected_path": str(blur_dir / seq_id),
                "description": f"Sequence {seq_id} missing in {blur_dir.name}"
            })

        if in_sharp and sharp_path:
            for item in sharp_path.iterdir():
                if item.is_file() and item.suffix.lower() in supported_exts:
                    sharp_frames[item.name] = item
        else:
            issues_records.append({
                "dataset": split_name,
                "issue_type": "missing_sequence_in_sharp",
                "sequence": seq_id,
                "filename": "",
                "affected_path": str(sharp_dir / seq_id),
                "description": f"Sequence {seq_id} missing in {sharp_dir.name}"
            })

        # Check for empty sequences
        if in_blur and len(blur_frames) == 0:
            issues_records.append({
                "dataset": split_name,
                "issue_type": "empty_sequence_directory",
                "sequence": seq_id,
                "filename": "",
                "affected_path": str(blur_path),
                "description": f"Sequence directory in {blur_dir.name} contains no valid image frames"
            })
        if in_sharp and len(sharp_frames) == 0:
            issues_records.append({
                "dataset": split_name,
                "issue_type": "empty_sequence_directory",
                "sequence": seq_id,
                "filename": "",
                "affected_path": str(sharp_path),
                "description": f"Sequence directory in {sharp_dir.name} contains no valid image frames"
            })

        blur_files_set = set(blur_frames.keys())
        sharp_files_set = set(sharp_frames.keys())

        matched_files = sorted(list(blur_files_set & sharp_files_set))
        missing_sharp = sorted(list(blur_files_set - sharp_files_set))
        missing_blur = sorted(list(sharp_files_set - blur_files_set))

        for f in missing_sharp:
            issues_records.append({
                "dataset": split_name,
                "issue_type": "missing_sharp_frame",
                "sequence": seq_id,
                "filename": f,
                "affected_path": str(sharp_path / f) if sharp_path else str(sharp_dir / seq_id / f),
                "description": f"Frame {f} exists in blur but is missing in sharp"
            })

        for f in missing_blur:
            issues_records.append({
                "dataset": split_name,
                "issue_type": "missing_blur_frame",
                "sequence": seq_id,
                "filename": f,
                "affected_path": str(blur_path / f) if blur_path else str(blur_dir / seq_id / f),
                "description": f"Frame {f} exists in sharp but is missing in blur"
            })

        if not in_blur or not in_sharp:
            status = "MISSING_SEQUENCE"
        elif len(blur_frames) == 0 and len(sharp_frames) == 0:
            status = "EMPTY_SEQUENCE"
        elif len(missing_sharp) > 0 or len(missing_blur) > 0:
            status = "PARTIAL_MATCH"
        else:
            status = "PERFECT_MATCH"

        summary_records.append({
            "dataset": split_name,
            "sequence": seq_id,
            "blur_frame_count": len(blur_frames),
            "sharp_frame_count": len(sharp_frames),
            "matched_frame_count": len(matched_files),
            "missing_blur_frames": len(missing_blur),
            "missing_sharp_frames": len(missing_sharp),
            "dimension_mismatches": 0,
            "unreadable_images": 0,
            "pairing_status": status
        })

    df_summary = pd.DataFrame(summary_records, columns=[
        "dataset", "sequence", "blur_frame_count", "sharp_frame_count",
        "matched_frame_count", "missing_blur_frames", "missing_sharp_frames",
        "dimension_mismatches", "unreadable_images", "pairing_status"
    ])
    df_issues = pd.DataFrame(issues_records, columns=[
        "dataset", "issue_type", "sequence", "filename", "affected_path", "description"
    ])
    return df_summary, df_issues


def get_candidate_pairs(
    split_name: str,
    blur_dir: Path,
    sharp_dir: Path,
    supported_exts: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Generate the list of candidate matched pairs by pairing (sequence_id, filename).
    Never pairs by sorted array index alone.
    """
    if supported_exts is None:
        supported_exts = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
    supported_exts = [e.lower() for e in supported_exts]

    candidate_pairs = []
    if not (blur_dir.exists() and sharp_dir.exists()):
        return candidate_pairs

    blur_seq_dirs = {d.name: d for d in blur_dir.iterdir() if d.is_dir()}
    sharp_seq_dirs = {d.name: d for d in sharp_dir.iterdir() if d.is_dir()}

    common_seqs = sorted(list(set(blur_seq_dirs.keys()) & set(sharp_seq_dirs.keys())))

    for seq_id in common_seqs:
        b_dir = blur_seq_dirs[seq_id]
        s_dir = sharp_seq_dirs[seq_id]

        b_files = {f.name: f for f in b_dir.iterdir() if f.is_file() and f.suffix.lower() in supported_exts}
        s_files = {f.name: f for f in s_dir.iterdir() if f.is_file() and f.suffix.lower() in supported_exts}

        common_files = sorted(list(set(b_files.keys()) & set(s_files.keys())))
        for fname in common_files:
            candidate_pairs.append({
                "dataset": split_name,
                "sequence": seq_id,
                "filename": fname,
                "blur_path": b_files[fname],
                "sharp_path": s_files[fname]
            })

    return candidate_pairs
