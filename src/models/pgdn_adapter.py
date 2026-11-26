"""PGDN Model Adapter and Training Interface.
Connects the REDS dataset pipeline to the PGDN (Prior-Guided Deblurring Network, 2025) architecture.
Handles tensor collation, batch preparation, Charbonnier loss calculation, and robust checkpoint serialization.
"""
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union, List
import yaml
import numpy as np
from PIL import Image
import torch
import torch.nn as nn

from src.data.config_utils import get_project_root
from src.models.pgdn_arch import PGDN


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (differentiable L1 approximation)."""
    def __init__(self, eps: float = 1e-12):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps)
        return torch.mean(loss)


class PGDNAdapter(nn.Module):
    """Adapter interfacing REDS dataset loader outputs with PGDN."""
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        config_path: Optional[Path] = None,
        project_root: Optional[Path] = None
    ):
        super().__init__()
        self.project_root = project_root or get_project_root()

        if config is None:
            if config_path is None:
                config_path = self.project_root / "configs" / "pgdn.yaml"
            if config_path.exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    self.cfg = yaml.safe_load(f)
            else:
                self.cfg = {}
        else:
            self.cfg = config

        m_cfg = self.cfg.get("model", {})
        self.num_frames = m_cfg.get("num_frames", 5)
        self.center_frame_idx = m_cfg.get("center_frame_idx", 2)
        self.in_channels = m_cfg.get("in_channels", 3)
        self.embed_dim = m_cfg.get("embed_dim", 64)
        self.num_blocks = m_cfg.get("num_blocks", 4)
        self.prior_dim = m_cfg.get("motion_prior_dim", 32)

        self.model = PGDN(
            num_frames=self.num_frames,
            center_frame_idx=self.center_frame_idx,
            in_channels=self.in_channels,
            embed_dim=self.embed_dim,
            num_blocks=self.num_blocks,
            prior_dim=self.prior_dim
        )

        t_cfg = self.cfg.get("train", {})
        eps = float(t_cfg.get("loss", {}).get("eps", 1e-12))
        self.criterion = CharbonnierLoss(eps=eps)

    def _to_tensor(self, img: Union[Image.Image, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Convert a single image to normalized float32 Tensor (C, H, W)."""
        if isinstance(img, torch.Tensor):
            if img.dim() == 3:
                t = img.float()
                if t.max() > 1.0:
                    t = t / 255.0
                return t
            elif img.dim() == 4 and img.shape[0] == 1:
                return self._to_tensor(img.squeeze(0))
            else:
                raise ValueError(f"Expected 3D Tensor (C, H, W), got shape {img.shape}")
        elif isinstance(img, np.ndarray):
            arr = img.astype(np.float32)
            if arr.max() > 1.0:
                arr = arr / 255.0
            if arr.ndim == 3 and arr.shape[2] in (1, 3):
                return torch.from_numpy(arr).permute(2, 0, 1)
            elif arr.ndim == 3 and arr.shape[0] in (1, 3):
                return torch.from_numpy(arr)
            else:
                raise ValueError(f"Unsupported numpy image array shape: {arr.shape}")
        elif isinstance(img, Image.Image):
            arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
            return torch.from_numpy(arr).permute(2, 0, 1)
        else:
            raise TypeError(f"Unsupported image type: {type(img)}")

    def collate_fn(self, batch_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Custom collate function for PyTorch DataLoader."""
        b_tensors = []
        s_tensors = []
        seq_ids = []
        center_indices = []
        center_filenames = []
        has_gt_flags = []

        for item in batch_list:
            frames = item["blur_frames"]
            frame_tensors = [self._to_tensor(f) for f in frames]
            stacked_frames = torch.stack(frame_tensors, dim=0)  # (T, C, H, W)
            b_tensors.append(stacked_frames)

            seq_ids.append(item.get("sequence_id", ""))
            center_indices.append(item.get("center_index", -1))
            center_filenames.append(item.get("center_filename", ""))
            has_gt_flags.append(item.get("has_ground_truth", False))

            if "sharp_frame" in item and item["sharp_frame"] is not None:
                s_tensors.append(self._to_tensor(item["sharp_frame"]))

        batch = {
            "blur_frames": torch.stack(b_tensors, dim=0),  # (B, T, C, H, W)
            "sequence_id": seq_ids,
            "center_index": center_indices,
            "center_filename": center_filenames,
            "has_ground_truth": has_gt_flags
        }
        if len(s_tensors) == len(b_tensors):
            batch["sharp_frame"] = torch.stack(s_tensors, dim=0)  # (B, C, H, W)

        return batch

    def prepare_batch(
        self,
        batch: Union[Dict[str, Any], torch.Tensor, List[Any]],
        device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Prepare batch data for model input and loss computation."""
        device = device or next(self.model.parameters()).device

        if isinstance(batch, torch.Tensor):
            x = batch.to(device)
            return x, None

        if isinstance(batch, dict):
            blur_data = batch["blur_frames"]
            if isinstance(blur_data, torch.Tensor):
                if blur_data.dim() == 5:
                    x = blur_data.to(device)
                elif blur_data.dim() == 4:
                    x = blur_data.unsqueeze(0).to(device)
                else:
                    raise ValueError(f"Unexpected blur_frames tensor shape: {blur_data.shape}")
            elif isinstance(blur_data, list):
                if len(blur_data) > 0 and isinstance(blur_data[0], torch.Tensor) and blur_data[0].dim() == 4:
                    x = torch.stack(blur_data, dim=1).to(device)
                else:
                    tensors = [self._to_tensor(f) for f in blur_data]
                    x = torch.stack(tensors, dim=0).unsqueeze(0).to(device)
            else:
                raise TypeError(f"Unsupported blur_frames type: {type(blur_data)}")

            y = None
            if "sharp_frame" in batch and batch["sharp_frame"] is not None:
                sharp_data = batch["sharp_frame"]
                if isinstance(sharp_data, torch.Tensor):
                    if sharp_data.dim() == 4:
                        y = sharp_data.to(device)
                    elif sharp_data.dim() == 3:
                        y = sharp_data.unsqueeze(0).to(device)
                    else:
                        raise ValueError(f"Unexpected sharp_frame tensor shape: {sharp_data.shape}")
                else:
                    y = self._to_tensor(sharp_data).unsqueeze(0).to(device)

            b, t, c, h, w = x.shape
            if t != self.num_frames:
                raise ValueError(f"Temporal dimension mismatch: expected {self.num_frames}, got {t}")

            return x, y

        raise TypeError(f"Unsupported batch type: {type(batch)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.criterion(pred, target)

    def save_checkpoint(
        self,
        path: Union[str, Path],
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
        step: int = 0,
        epoch: int = 0,
        best_val_metric: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None
    ) -> Path:
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        ckpt = {
            "model_state_dict": self.model.state_dict(),
            "step": step,
            "epoch": epoch,
            "best_val_metric": best_val_metric,
            "config": self.cfg,
            "architecture": "PGDN-2025"
        }
        if optimizer is not None:
            ckpt["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler is not None:
            ckpt["scheduler_state_dict"] = scheduler.state_dict()
        if scaler is not None:
            ckpt["scaler_state_dict"] = scaler.state_dict()
        if extra:
            ckpt.update(extra)

        torch.save(ckpt, save_path)
        return save_path

    def load_checkpoint(
        self,
        path: Union[str, Path],
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
        device: Optional[torch.device] = None,
        strict: bool = True
    ) -> Dict[str, Any]:
        load_path = Path(path)
        if not load_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {load_path}")

        map_location = device or next(self.model.parameters()).device
        ckpt = torch.load(load_path, map_location=map_location, weights_only=False)

        state_dict = ckpt.get("model_state_dict", ckpt)
        self.model.load_state_dict(state_dict, strict=strict)

        if optimizer is not None and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if scaler is not None and "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])

        return ckpt
