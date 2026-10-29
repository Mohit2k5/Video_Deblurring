"""Comprehensive image integrity and pair dimension validation for REDS dataset."""
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
from PIL import Image
import os


def validate_single_image(
    image_path: Path,
    expected_width: Optional[int] = 1280,
    expected_height: Optional[int] = 720,
    expected_color_mode: Optional[str] = "RGB"
) -> Dict[str, Any]:
    """
    Validate a single image file for existence, readability, decode errors, dimensions, and color mode.
    Uses safe Image.verify() and reopen pattern.
    """
    res = {
        "path": str(image_path),
        "exists": False,
        "readable": False,
        "width": None,
        "height": None,
        "mode": None,
        "channels": None,
        "size_bytes": 0,
        "dimension_valid": False,
        "mode_valid": False,
        "status": "FAILED",
        "error": None
    }

    if not image_path.exists():
        res["error"] = "File does not exist"
        return res

    res["exists"] = True
    try:
        res["size_bytes"] = image_path.stat().st_size
    except Exception as e:
        res["error"] = f"stat error: {e}"
        return res

    if res["size_bytes"] == 0:
        res["error"] = "Zero-byte file"
        return res

    try:
        # Step 1: verify() to detect structural corruptions
        with Image.open(image_path) as img:
            img.verify()

        # Step 2: verify invalidates image object, so reopen for dimensions and decoding
        with Image.open(image_path) as img:
            res["width"], res["height"] = img.size
            res["mode"] = img.mode
            res["channels"] = len(img.getbands()) if hasattr(img, 'getbands') else None
            # Test full pixel decode
            img.load()

        res["readable"] = True

        # Check expected dimensions
        dim_ok = True
        if expected_width is not None and res["width"] != expected_width:
            dim_ok = False
        if expected_height is not None and res["height"] != expected_height:
            dim_ok = False
        res["dimension_valid"] = dim_ok

        # Check expected color mode
        mode_ok = True
        if expected_color_mode is not None and res["mode"] != expected_color_mode:
            mode_ok = False
        res["mode_valid"] = mode_ok

        if not dim_ok:
            res["status"] = "DIMENSION_MISMATCH"
            res["error"] = f"Unexpected dimensions: {res['width']}x{res['height']}"
        elif not mode_ok:
            res["status"] = "MODE_MISMATCH"
            res["error"] = f"Unexpected mode: {res['mode']}"
        else:
            res["status"] = "VALID"

    except Exception as e:
        res["readable"] = False
        res["status"] = "CORRUPTED"
        res["error"] = str(e)

    return res


def validate_split_integrity(
    split_name: str,
    split_dir: Path,
    supported_exts: Optional[List[str]] = None,
    expected_width: Optional[int] = 1280,
    expected_height: Optional[int] = 720,
    expected_color_mode: Optional[str] = "RGB",
    sample_limit: Optional[int] = None,
    progress_interval: int = 5000
) -> List[Dict[str, Any]]:
    """
    Validate all images in a dataset split directory (e.g. train_blur).
    Processes incrementally without loading entire dataset into memory.
    """
    if supported_exts is None:
        supported_exts = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
    supported_exts = [e.lower() for e in supported_exts]

    results = []
    if not split_dir.exists():
        return results

    count = 0
    # Walk directory structure: split_dir / seq_dir / img_file
    for seq_dir in sorted(list(split_dir.iterdir())):
        if not seq_dir.is_dir():
            continue
        seq_id = seq_dir.name
        for img_file in sorted(list(seq_dir.iterdir())):
            if img_file.is_file() and img_file.suffix.lower() in supported_exts:
                img_res = validate_single_image(
                    img_file,
                    expected_width=expected_width,
                    expected_height=expected_height,
                    expected_color_mode=expected_color_mode
                )
                img_res["dataset"] = split_name
                img_res["sequence"] = seq_id
                img_res["filename"] = img_file.name
                results.append(img_res)

                count += 1
                if count % progress_interval == 0:
                    print(f"[{split_name}] Validated {count} images...")

                if sample_limit is not None and count >= sample_limit:
                    print(f"[{split_name}] Reached sample limit of {sample_limit} images.")
                    return results

    print(f"[{split_name}] Completed validation of {count} images.")
    return results


def validate_all_splits(
    raw_dir: Path,
    dataset_dirs: Dict[str, str],
    supported_exts: Optional[List[str]] = None,
    expected_width: int = 1280,
    expected_height: int = 720,
    expected_color_mode: str = "RGB",
    sample_limit: Optional[int] = None
) -> pd.DataFrame:
    """
    Validate images across all specified splits.
    Returns DataFrame matching required image_validation_results.csv schema.
    """
    all_results = []
    for split_key, dir_name in dataset_dirs.items():
        s_dir = raw_dir / dir_name
        if s_dir.exists():
            res = validate_split_integrity(
                split_name=dir_name,
                split_dir=s_dir,
                supported_exts=supported_exts,
                expected_width=expected_width,
                expected_height=expected_height,
                expected_color_mode=expected_color_mode,
                sample_limit=sample_limit
            )
            all_results.extend(res)

    cols = [
        "dataset", "sequence", "filename", "path", "width", "height",
        "mode", "channels", "size_bytes", "readable", "status", "error"
    ]
    if not all_results:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(all_results)[cols]


def check_matched_pairs_dimensions(
    candidate_pairs: List[Dict[str, Any]],
    val_df: pd.DataFrame
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Compare dimensions, color modes, and readability between matched blur/sharp pairs.
    Returns:
        df_mismatches: DataFrame of pairs that failed dimension or readability check
        valid_pairs: List of verified candidate pairs ready for visualization/modeling
    """
    # Create lookup map from validation results DataFrame by path or (dataset, sequence, filename)
    lookup = {}
    if not val_df.empty:
        for _, row in val_df.iterrows():
            key = (row["dataset"], row["sequence"], row["filename"])
            lookup[key] = row.to_dict()

    mismatches = []
    valid_pairs = []

    for pair in candidate_pairs:
        split_name = pair["dataset"]
        # Split could be 'train' or 'train_blur' - check corresponding blur and sharp dir names
        blur_ds = f"{split_name}_blur" if not split_name.endswith("_blur") else split_name
        sharp_ds = f"{split_name}_sharp" if not split_name.endswith("_sharp") else split_name.replace("_blur", "_sharp")
        if split_name in ["train", "val"]:
            blur_ds = f"{split_name}_blur"
            sharp_ds = f"{split_name}_sharp"

        blur_key = (blur_ds, pair["sequence"], pair["filename"])
        sharp_key = (sharp_ds, pair["sequence"], pair["filename"])

        b_info = lookup.get(blur_key)
        s_info = lookup.get(sharp_key)

        # If not in lookup (e.g. when checking single pair directly), validate on the fly
        if not b_info:
            b_info = validate_single_image(pair["blur_path"])
        if not s_info:
            s_info = validate_single_image(pair["sharp_path"])

        b_read = b_info.get("readable", False)
        s_read = s_info.get("readable", False)
        b_w, b_h = b_info.get("width"), b_info.get("height")
        s_w, s_h = s_info.get("width"), s_info.get("height")
        b_mode = b_info.get("mode")
        s_mode = s_info.get("mode")

        is_mismatch = False
        issue_desc = []

        if not b_read:
            is_mismatch = True
            issue_desc.append(f"Blur image unreadable: {b_info.get('error')}")
        if not s_read:
            is_mismatch = True
            issue_desc.append(f"Sharp image unreadable: {s_info.get('error')}")

        if b_read and s_read:
            if b_w != s_w or b_h != s_h:
                is_mismatch = True
                issue_desc.append(f"Dimension mismatch: blur ({b_w}x{b_h}) vs sharp ({s_w}x{s_h})")
            if b_mode != s_mode:
                is_mismatch = True
                issue_desc.append(f"Mode mismatch: blur ({b_mode}) vs sharp ({s_mode})")

        if is_mismatch:
            mismatches.append({
                "dataset": split_name,
                "sequence": pair["sequence"],
                "filename": pair["filename"],
                "blur_path": str(pair["blur_path"]),
                "sharp_path": str(pair["sharp_path"]),
                "blur_width": b_w,
                "blur_height": b_h,
                "sharp_width": s_w,
                "sharp_height": s_h,
                "blur_mode": b_mode,
                "sharp_mode": s_mode,
                "description": "; ".join(issue_desc)
            })
        else:
            pair_meta = dict(pair)
            pair_meta["width"] = b_w
            pair_meta["height"] = b_h
            pair_meta["mode"] = b_mode
            pair_meta["passed_validation"] = True
            valid_pairs.append(pair_meta)

    cols = [
        "dataset", "sequence", "filename", "blur_path", "sharp_path",
        "blur_width", "blur_height", "sharp_width", "sharp_height",
        "blur_mode", "sharp_mode", "description"
    ]
    df_mismatches = pd.DataFrame(mismatches, columns=cols)
    return df_mismatches, valid_pairs
