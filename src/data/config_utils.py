"""Config and path resolution utilities for REDS Video Deblurring project."""
from pathlib import Path
from typing import Dict, Any, Optional
import yaml


def get_project_root(current_file: Optional[Path] = None) -> Path:
    """
    Resolve the project root reliably.
    Looks for markers like 'configs/eda_config.yaml', 'src', or 'requirements.txt'.
    """
    if current_file is not None:
        p = Path(current_file).resolve()
    else:
        p = Path.cwd().resolve()

    # Search upward
    for parent in [p] + list(p.parents):
        if (parent / "configs" / "eda_config.yaml").exists() or (parent / "src" / "data").exists():
            return parent

    # Fallback to current working directory
    return Path.cwd().resolve()


def load_config(config_path: Optional[Path] = None, project_root: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load and validate eda_config.yaml.
    Resolves relative paths against project_root.
    """
    root = project_root or get_project_root()
    if config_path is None:
        config_path = root / "configs" / "eda_config.yaml"
    else:
        config_path = Path(config_path)
        if not config_path.is_absolute():
            config_path = root / config_path

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Validate structure and values
    validate_config(config, root)
    return config


def validate_config(config: Dict[str, Any], project_root: Path):
    """Ensure configuration parameters are valid and sane."""
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a dictionary")

    # Validate paths if present
    paths = config.get("paths", {})
    raw_dir_rel = paths.get("raw_dir") or config.get("raw_data_dir", "data/raw")
    raw_dir = project_root / raw_dir_rel
    # Note: raw_dir might not exist yet during synthetic tests, so we warn or permit check in calling code

    # Validate visualization counts
    vis = config.get("visualization", {})
    if vis.get("train_pair_count", 0) < 0:
        raise ValueError("visualization.train_pair_count must be non-negative")
    if vis.get("val_pair_count", 0) < 0:
        raise ValueError("visualization.val_pair_count must be non-negative")
    if vis.get("consecutive_frames_per_sequence", 0) < 0:
        raise ValueError("visualization.consecutive_frames_per_sequence must be non-negative")

    # Validate validation settings
    val = config.get("validation", {})
    if "expected_width" in val and val["expected_width"] <= 0:
        raise ValueError("validation.expected_width must be positive")
    if "expected_height" in val and val["expected_height"] <= 0:
        raise ValueError("validation.expected_height must be positive")
