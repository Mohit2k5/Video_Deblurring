"""Dataset indexing module that reuses existing EDA outputs."""
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
from src.data.config_utils import get_project_root


def load_dataset_index(
    project_root: Optional[Path] = None,
    raw_dir: Optional[Path] = None,
    eda_reports_dir: Optional[Path] = None,
    force_rescan: bool = False
) -> pd.DataFrame:
    """
    Load dataset inventory by reusing existing EDA inventory if available and valid.
    Falls back to discovering if not found or if force_rescan is True.
    """
    root = project_root or get_project_root()
    if eda_reports_dir is None:
        eda_reports_dir = root / "reports" / "eda"
    if raw_dir is None:
        raw_dir = root / "data" / "raw"

    inv_path = eda_reports_dir / "dataset_inventory.csv"
    if inv_path.exists() and not force_rescan:
        df_inv = pd.read_csv(inv_path)
        # Lightweight consistency check: verify at least one sequence directory exists
        if not df_inv.empty and "dataset" in df_inv.columns and "sequence" in df_inv.columns:
            first_row = df_inv.iloc[0]
            sample_dir = raw_dir / str(first_row["dataset"]) / str(first_row["sequence"])
            if sample_dir.exists():
                return df_inv

    # If missing or inconsistent, use discover_datasets from dataset_inventory
    from src.data.dataset_inventory import discover_datasets
    df_inv, _ = discover_datasets(raw_dir, ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'])
    return df_inv


def get_available_sequences(
    dataset_name: str,
    project_root: Optional[Path] = None,
    eda_reports_dir: Optional[Path] = None
) -> List[str]:
    """
    Retrieve sorted list of sequence IDs for a dataset split (e.g. 'train', 'val', 'train_blur').
    """
    root = project_root or get_project_root()
    if eda_reports_dir is None:
        eda_reports_dir = root / "reports" / "eda"

    # Try reading from frame_pairing_summary.csv first (which groups train_blur and train_sharp under 'train')
    pairing_csv = eda_reports_dir / "frame_pairing_summary.csv"
    if pairing_csv.exists():
        df_pair = pd.read_csv(pairing_csv)
        # Format sequence IDs as 3-digit strings e.g. '000', '001'
        df_pair["seq_str"] = df_pair["sequence"].apply(lambda s: f"{int(s):03d}" if str(s).isdigit() else str(s))
        if dataset_name in df_pair["dataset"].values:
            seqs = df_pair[df_pair["dataset"] == dataset_name]["seq_str"].tolist()
            return sorted(list(set(seqs)))

    df_inv = load_dataset_index(project_root=root, eda_reports_dir=eda_reports_dir)
    if not df_inv.empty:
        # Match either exact dataset name or prefix (e.g., 'train_blur' matching 'train')
        matches = df_inv[df_inv["dataset"].str.startswith(dataset_name)]["sequence"].tolist()
        formatted = [f"{int(s):03d}" if str(s).isdigit() else str(s) for s in matches]
        return sorted(list(set(formatted)))

    return []
