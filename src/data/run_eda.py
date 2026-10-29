"""Main pipeline runner for REDS Dataset Pair Verification, Integrity Validation, and Visual Analysis."""
import sys
from pathlib import Path
import pandas as pd
import yaml

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.config_utils import get_project_root, load_config
from src.data.dataset_inventory import discover_datasets
from src.data.pairing_verification import verify_split_pairing, get_candidate_pairs
from src.data.image_validation import validate_all_splits, check_matched_pairs_dimensions
from src.data.visualization import (
    plot_blur_sharp_pairs,
    plot_consecutive_frames,
    plot_dataset_distribution,
    plot_image_dimensions,
    plot_brightness,
    plot_contrast,
    plot_sharpness
)
from src.data.report_generation import generate_frame_pairing_report


def run_pipeline():
    project_root = get_project_root()
    print(f"=== Starting REDS EDA & Verification Pipeline ===")
    print(f"Project root: {project_root}")

    config = load_config(project_root=project_root)
    paths_cfg = config.get("paths", {})
    raw_dir = project_root / paths_cfg.get("raw_dir", "data/raw")
    reports_dir = project_root / paths_cfg.get("reports_dir", "reports/eda")
    figures_dir = project_root / paths_cfg.get("figures_dir", "figures/eda")
    legacy_fig_dir = project_root / config.get("figure_dir", "figures")

    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    legacy_fig_dir.mkdir(parents=True, exist_ok=True)

    supported_exts = config.get("supported_extensions", ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'])
    val_cfg = config.get("validation", {})
    vis_cfg = config.get("visualization", {})
    datasets_cfg = config.get("datasets", {
        "train_blur": "train_blur",
        "train_sharp": "train_sharp",
        "val_blur": "val_blur",
        "val_sharp": "val_sharp",
        "test_blur": "test_blur"
    })

    # Step 1: Inventory
    print("\n--- Step 1: Dataset Inventory Discovery ---")
    df_inv, df_ext = discover_datasets(raw_dir, supported_exts)
    df_inv.to_csv(reports_dir / "dataset_inventory.csv", index=False)
    df_ext.to_csv(reports_dir / "file_extensions.csv", index=False)
    print(f"Discovered {len(df_inv)} sequences across datasets. Total images: {df_inv['image_count'].sum() if not df_inv.empty else 0}")

    # Step 2: Sequence and Frame Pairing Verification
    print("\n--- Step 2: Sequence and Frame Pairing Checks ---")
    train_b_dir = raw_dir / datasets_cfg.get("train_blur", "train_blur")
    train_s_dir = raw_dir / datasets_cfg.get("train_sharp", "train_sharp")
    val_b_dir = raw_dir / datasets_cfg.get("val_blur", "val_blur")
    val_s_dir = raw_dir / datasets_cfg.get("val_sharp", "val_sharp")

    df_tr_summary, df_tr_issues = verify_split_pairing("train", train_b_dir, train_s_dir, supported_exts)
    df_val_summary, df_val_issues = verify_split_pairing("val", val_b_dir, val_s_dir, supported_exts)

    df_pairing_summary = pd.concat([df_tr_summary, df_val_summary], ignore_index=True)
    df_pairing_issues = pd.concat([df_tr_issues, df_val_issues], ignore_index=True)

    df_pairing_summary.to_csv(reports_dir / "frame_pairing_summary.csv", index=False)
    df_pairing_issues.to_csv(reports_dir / "frame_pairing_issues.csv", index=False)

    print(f"Train sequences evaluated: {len(df_tr_summary)}, Val sequences evaluated: {len(df_val_summary)}")
    print(f"Total pairing issues discovered: {len(df_pairing_issues)}")

    # Step 3: Candidate pairs gathering
    train_candidate_pairs = get_candidate_pairs("train", train_b_dir, train_s_dir, supported_exts)
    val_candidate_pairs = get_candidate_pairs("val", val_b_dir, val_s_dir, supported_exts)
    print(f"Candidate pairs found: Train={len(train_candidate_pairs)}, Val={len(val_candidate_pairs)}")

    # Step 4: Full Image-Integrity Validation
    print("\n--- Step 3: Full Image-Integrity Validation ---")
    full_val = val_cfg.get("full_validation", True)
    sample_limit = val_cfg.get("sample_limit", None) if not full_val else None

    # Check cached results freshness if existing
    val_csv_path = reports_dir / "image_validation_results.csv"
    mismatch_csv_path = reports_dir / "image_dimension_mismatches.csv"

    # Validate the 4 primary splits (train_blur, train_sharp, val_blur, val_sharp)
    splits_to_validate = {
        "train_blur": datasets_cfg.get("train_blur", "train_blur"),
        "train_sharp": datasets_cfg.get("train_sharp", "train_sharp"),
        "val_blur": datasets_cfg.get("val_blur", "val_blur"),
        "val_sharp": datasets_cfg.get("val_sharp", "val_sharp")
    }

    df_val_results = validate_all_splits(
        raw_dir=raw_dir,
        dataset_dirs=splits_to_validate,
        supported_exts=supported_exts,
        expected_width=val_cfg.get("expected_width", 1280),
        expected_height=val_cfg.get("expected_height", 720),
        expected_color_mode=val_cfg.get("expected_color_mode", "RGB"),
        sample_limit=sample_limit
    )
    df_val_results.to_csv(val_csv_path, index=False)
    print(f"Validation complete: Total images checked = {len(df_val_results)}")

    # Step 5: Check Matched Pairs Dimensions & Mode
    print("\n--- Step 4: Matched-Pair Dimension Checks ---")
    all_candidate_pairs = train_candidate_pairs + val_candidate_pairs
    df_dim_mismatches, valid_pairs = check_matched_pairs_dimensions(all_candidate_pairs, df_val_results)
    df_dim_mismatches.to_csv(mismatch_csv_path, index=False)
    print(f"Dimension/Readability mismatches across pairs: {len(df_dim_mismatches)}")
    print(f"Total valid candidate pairs ready: {len(valid_pairs)}")

    valid_train_pairs = [p for p in valid_pairs if p["dataset"] == "train"]
    valid_val_pairs = [p for p in valid_pairs if p["dataset"] == "val"]

    # Step 6: Configurable Side-by-Side Visualizations
    print("\n--- Step 5: Generating Visualizations ---")
    tr_count = vis_cfg.get("train_pair_count", 6)
    val_count = vis_cfg.get("val_pair_count", 6)
    consec_count = vis_cfg.get("consecutive_frames_per_sequence", 3)
    seed = vis_cfg.get("random_seed", 42)

    tr_fig_path = figures_dir / "train_blur_sharp_pairs.png"
    val_fig_path = figures_dir / "val_blur_sharp_pairs.png"
    consec_fig_path = figures_dir / "consecutive_frame_comparisons.png"

    sel_tr, stats_tr = plot_blur_sharp_pairs(valid_train_pairs, tr_count, tr_fig_path, "Training", random_seed=seed)
    sel_val, stats_val = plot_blur_sharp_pairs(valid_val_pairs, val_count, val_fig_path, "Validation", random_seed=seed)
    sel_consec, stats_consec = plot_consecutive_frames(valid_train_pairs, consec_count, consec_fig_path, random_seed=seed)

    # Save visualized pairs metadata
    all_visualized = []
    for idx, p in enumerate(sel_tr):
        all_visualized.append({
            "pair_number": idx + 1,
            "dataset_split": "train",
            "sequence_id": p["sequence"],
            "frame_filename": p["filename"],
            "blur_path": str(p["blur_path"]),
            "sharp_path": str(p["sharp_path"]),
            "dimensions": f"{p.get('width', 1280)}x{p.get('height', 720)}",
            "passed_validation": p.get("passed_validation", True)
        })
    for idx, p in enumerate(sel_val):
        all_visualized.append({
            "pair_number": idx + 1,
            "dataset_split": "val",
            "sequence_id": p["sequence"],
            "frame_filename": p["filename"],
            "blur_path": str(p["blur_path"]),
            "sharp_path": str(p["sharp_path"]),
            "dimensions": f"{p.get('width', 1280)}x{p.get('height', 720)}",
            "passed_validation": p.get("passed_validation", True)
        })

    cols_vis = ["pair_number", "dataset_split", "sequence_id", "frame_filename", "blur_path", "sharp_path", "dimensions", "passed_validation"]
    df_vis_pairs = pd.DataFrame(all_visualized, columns=cols_vis)
    df_vis_pairs.to_csv(reports_dir / "visualized_pairs.csv", index=False)
    print(f"Visualized pairs saved: Train={len(sel_tr)}, Val={len(sel_val)}, Consecutive={len(sel_consec)}")

    # Step 7: Final Markdown Report Generation
    print("\n--- Step 6: Generating Comprehensive EDA Report ---")
    vis_meta = {
        "train": stats_tr,
        "val": stats_val,
        "consecutive": stats_consec,
        "random_seed": seed
    }
    report_path = generate_frame_pairing_report(
        report_dir=reports_dir,
        df_inv=df_inv,
        df_pairing_summary=df_pairing_summary,
        df_pairing_issues=df_pairing_issues,
        df_val_results=df_val_results,
        df_dimension_mismatches=df_dim_mismatches,
        vis_metadata=vis_meta
    )
    print(f"Report generated successfully at: {report_path}")
    print("=== Pipeline Complete ===")


if __name__ == "__main__":
    run_pipeline()
