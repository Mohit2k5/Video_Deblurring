"""Shared, Model-Independent Video Deblurring Dataset Loader for REDS."""
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Callable, Union
import pandas as pd
from PIL import Image
import numpy as np

from src.data.config_utils import get_project_root, load_config
from src.data.indexing import get_available_sequences
from src.preprocessing.temporal_windows import build_temporal_indices
from src.preprocessing.paired_transforms import PairedTransformPipeline


class REDSVideoDeblurDataset:
    """
    Model-independent PyTorch-compatible dataset for REDS video deblurring.

    Supports:
    - Partition loading ('train', 'val', 'held_out_test', 'inference_test')
    - Paired (blur, sharp) or inference-only (blur) mode
    - Arbitrary odd temporal window size T (e.g. T=5 for EDVR, T=1 for single-frame models like MIMO-UNet)
    - Boundary padding policies ('replicate', 'reflect', 'truncate', 'zero_pad')
    - Deterministic indexing (never pairs across sequence boundaries)
    """
    def __init__(
        self,
        partition: str = "train",
        project_root: Optional[Path] = None,
        config: Optional[Dict[str, Any]] = None,
        window_size: int = 5,
        boundary_policy: str = "replicate",
        transform: Optional[Callable] = None,
        supported_exts: Optional[List[str]] = None
    ):
        self.project_root = project_root or get_project_root()
        self.partition = partition
        self.window_size = window_size
        self.boundary_policy = boundary_policy
        self.transform = transform
        self.supported_exts = supported_exts or ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']

        # Load configs if not supplied, with sensible defaults if file absent
        if config is None:
            data_cfg_path = self.project_root / "configs" / "data.yaml"
            eda_cfg_path = self.project_root / "configs" / "eda_config.yaml"
            if data_cfg_path.exists():
                import yaml
                with open(data_cfg_path, "r", encoding="utf-8") as f:
                    self.config = yaml.safe_load(f)
            elif eda_cfg_path.exists():
                self.config = load_config(config_path=eda_cfg_path, project_root=self.project_root)
            else:
                self.config = {
                    "paths": {"raw_dir": "data/raw", "manifests_dir": "data/manifests"},
                    "datasets": {
                        "train_blur": "train_blur", "train_sharp": "train_sharp",
                        "val_blur": "val_blur", "val_sharp": "val_sharp",
                        "test_blur": "test_blur"
                    }
                }
        else:
            self.config = config

        self.raw_dir = self.project_root / self.config.get("paths", {}).get("raw_dir", "data/raw")
        self.manifest_dir = self.project_root / self.config.get("paths", {}).get("manifests_dir", "data/manifests")

        # Load split manifest to determine sequence assignment
        self.split_manifest = self._load_manifest()

        # Build sample index
        self.samples = self._build_samples()

    def _load_manifest(self) -> pd.DataFrame:
        manifest_path = self.manifest_dir / "sequence_split.csv"
        if not manifest_path.exists():
            manifest_path = self.project_root / "splits" / "sequence_split.csv"

        if manifest_path.exists():
            df = pd.read_csv(manifest_path)
            df["sequence_id"] = df["sequence_id"].apply(lambda s: f"{int(s):03d}" if str(s).isdigit() else str(s))
            return df[df["partition"] == self.partition]

        raise FileNotFoundError(
            f"Split manifest not found at {manifest_path}. Please run generate_sequence_split() first."
        )

    def _build_samples(self) -> List[Dict[str, Any]]:
        samples = []
        if self.split_manifest.empty:
            return samples

        is_val = self.partition == "val"
        is_inference = self.partition == "inference_test"
        has_gt = not is_inference

        if is_val:
            blur_dir_name = self.config.get("datasets", {}).get("val_blur", "val_blur")
            sharp_dir_name = self.config.get("datasets", {}).get("val_sharp", "val_sharp")
        elif is_inference:
            blur_dir_name = self.config.get("datasets", {}).get("test_blur", "test_blur")
            sharp_dir_name = None
        else:
            blur_dir_name = self.config.get("datasets", {}).get("train_blur", "train_blur")
            sharp_dir_name = self.config.get("datasets", {}).get("train_sharp", "train_sharp")

        blur_root = self.raw_dir / blur_dir_name
        sharp_root = (self.raw_dir / sharp_dir_name) if sharp_dir_name else None

        for _, row in self.split_manifest.iterrows():
            seq_id = str(row["sequence_id"])
            seq_blur_dir = blur_root / seq_id
            seq_sharp_dir = (sharp_root / seq_id) if sharp_root else None

            if not seq_blur_dir.exists():
                continue

            blur_files = sorted([
                f for f in seq_blur_dir.iterdir()
                if f.is_file() and f.suffix.lower() in self.supported_exts
            ])
            n_frames = len(blur_files)
            if n_frames == 0:
                continue

            windows = build_temporal_indices(
                sequence_length=n_frames,
                window_size=self.window_size,
                boundary_policy=self.boundary_policy
            )

            for win in windows:
                center_idx = win[self.window_size // 2]
                center_filename = blur_files[center_idx].name

                win_paths = []
                for idx in win:
                    if idx >= 0:
                        win_paths.append(blur_files[idx])
                    else:
                        win_paths.append(None)

                sharp_target_path = None
                if has_gt and seq_sharp_dir:
                    sharp_target_path = seq_sharp_dir / center_filename

                samples.append({
                    "partition": self.partition,
                    "sequence_id": seq_id,
                    "center_index": center_idx,
                    "center_filename": center_filename,
                    "blur_paths": win_paths,
                    "sharp_path": sharp_target_path,
                    "has_ground_truth": has_gt
                })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        blur_images = []
        for p in sample["blur_paths"]:
            if p is not None and p.exists():
                with Image.open(p) as img:
                    blur_images.append(img.convert("RGB"))
            else:
                blur_images.append(Image.new("RGB", (1280, 720), (0, 0, 0)))

        sharp_image = None
        if sample["has_ground_truth"] and sample["sharp_path"] and sample["sharp_path"].exists():
            with Image.open(sample["sharp_path"]) as img:
                sharp_image = img.convert("RGB")

        if self.transform is not None:
            if sharp_image is not None:
                blur_images, sharp_image = self.transform(blur_images, sharp_image)
            else:
                dummy = Image.new("RGB", blur_images[0].size, (0, 0, 0))
                blur_images, _ = self.transform(blur_images, dummy)

        item = {
            "blur_frames": blur_images,
            "sequence_id": sample["sequence_id"],
            "center_index": sample["center_index"],
            "center_filename": sample["center_filename"],
            "has_ground_truth": sample["has_ground_truth"]
        }
        if sharp_image is not None:
            item["sharp_frame"] = sharp_image

        return item
