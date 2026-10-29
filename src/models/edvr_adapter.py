"""EDVR Model Adapter and Training Interface.
Connects the shared REDS dataset pipeline to the EDVR model, manages tensor conversion,
batched DataLoader collation, Charbonnier loss computation, AMP execution, and checkpoint persistence.
"""
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union, List
import yaml
import numpy as np
from PIL import Image
import torch
import torch.nn as nn

from src.data.config_utils import get_project_root
from src.models.edvr_arch import EDVR


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (differentiable L1 approximation).
    
    Formula: L = sqrt((pred - target)^2 + eps^2) or sqrt((pred - target)^2 + eps)
    """
    def __init__(self, eps: float = 1e-12):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps)
        return torch.mean(loss)


class EDVRAdapter(nn.Module):
    """
    Adapter interfacing REDS dataset loader outputs with the EDVR model.
    
    Features:
    - Accepts both single-sample PIL/array dicts and batched DataLoader tensors
    - Provides a robust collate_fn for standard PyTorch DataLoader pipelines
    - Performs value normalization to [0, 1] RGB float32
    - Enforces spatial dimension divisibility by 16 for hr_in mode
    - Selects correct center-frame sharp ground-truth target
    - Computes Charbonnier loss
    - Full checkpoint save & restore with optimizer and AMP scaler states
    - Dynamic project root resolution (no hard-coded machine paths)
    """
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
                config_path = self.project_root / "configs" / "edvr.yaml"
            with open(config_path, "r", encoding="utf-8") as f:
                self.cfg = yaml.safe_load(f)
        else:
            self.cfg = config

        m_cfg = self.cfg.get("model", {})
        self.num_feat = m_cfg.get("num_feat", 64)
        self.num_frame = m_cfg.get("num_frame", 5)
        self.num_extract_block = m_cfg.get("num_extract_block", 5)
        self.num_reconstruct_block = m_cfg.get("num_reconstruct_block", 10)
        self.center_frame_idx = m_cfg.get("center_frame_idx", 2)
        if self.center_frame_idx is None:
            self.center_frame_idx = self.num_frame // 2
        self.hr_in = m_cfg.get("hr_in", True)
        self.with_predeblur = m_cfg.get("with_predeblur", True)
        self.with_tsa = m_cfg.get("with_tsa", True)
        self.deformable_groups = m_cfg.get("deformable_groups", 8)

        self.model = EDVR(
            num_in_ch=m_cfg.get("num_in_ch", 3),
            num_out_ch=m_cfg.get("num_out_ch", 3),
            num_feat=self.num_feat,
            num_frame=self.num_frame,
            deformable_groups=self.deformable_groups,
            num_extract_block=self.num_extract_block,
            num_reconstruct_block=self.num_reconstruct_block,
            center_frame_idx=self.center_frame_idx,
            hr_in=self.hr_in,
            with_predeblur=self.with_predeblur,
            with_tsa=self.with_tsa
        )

        t_cfg = self.cfg.get("train", {})
        eps = float(t_cfg.get("loss_eps", 1e-12))
        self.criterion = CharbonnierLoss(eps=eps)

    def _to_tensor(self, img: Union[Image.Image, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Convert a single image (PIL, numpy, or Tensor) to normalized float32 Tensor (C, H, W)."""
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
        """Custom collate function for PyTorch DataLoader.
        
        Transforms individual dataset sample dicts into a collated batch:
        - blur_frames: Tensor (B, T, C, H, W)
        - sharp_frame: Tensor (B, C, H, W) if present
        - sequence_ids, center_indices, center_filenames: lists
        """
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
        """Prepare batch data for model input and loss computation.
        
        Accepts:
        - Collated DataLoader batch dict
        - Raw single-sample dict
        - Direct torch Tensor (B, T, C, H, W)
        
        Returns:
        - x: (B, T, C, H, W) float32 Tensor on target device
        - y: (B, C, H, W) float32 Tensor on target device, or None
        """
        device = device or next(self.model.parameters()).device

        if isinstance(batch, torch.Tensor):
            x = batch.to(device)
            return x, None

        if isinstance(batch, dict):
            # Check if blur_frames is already a collated Tensor
            blur_data = batch["blur_frames"]
            if isinstance(blur_data, torch.Tensor):
                if blur_data.dim() == 5:
                    x = blur_data.to(device)
                elif blur_data.dim() == 4:
                    x = blur_data.unsqueeze(0).to(device)
                else:
                    raise ValueError(f"Unexpected blur_frames tensor shape: {blur_data.shape}")
            elif isinstance(blur_data, list):
                # Either list of Tensors or list of PIL/numpy images
                if len(blur_data) > 0 and isinstance(blur_data[0], torch.Tensor) and blur_data[0].dim() == 4:
                    # List of (B, C, H, W) tensors per timestep
                    x = torch.stack(blur_data, dim=1).to(device)
                else:
                    # Single sequence of images: list of T images
                    tensors = [self._to_tensor(f) for f in blur_data]
                    x = torch.stack(tensors, dim=0).unsqueeze(0).to(device)
            else:
                raise TypeError(f"Unsupported blur_frames type: {type(blur_data)}")

            # Process ground truth target
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

            # Validate input dimensions
            b, t, c, h, w = x.shape
            if t != self.num_frame:
                raise ValueError(f"Temporal dimension mismatch: expected {self.num_frame}, got {t}")
            if self.hr_in and (h % 16 != 0 or w % 16 != 0):
                raise ValueError(
                    f"Height and width must be divisible by 16 in hr_in mode, got ({h}, {w})"
                )

            return x, y

        raise TypeError(f"Unsupported batch type: {type(batch)}")

    def prepare_input(self, blur_frames_list: List[Image.Image], device: torch.device) -> torch.Tensor:
        """Convert list of PIL images to normalized Tensor (1, T, C, H, W). Kept for backward compatibility."""
        x, _ = self.prepare_batch({"blur_frames": blur_frames_list}, device=device)
        return x

    def prepare_target(self, sharp_frame: Image.Image, device: torch.device) -> torch.Tensor:
        """Convert single PIL image target to normalized Tensor (1, C, H, W). Kept for backward compatibility."""
        return self._to_tensor(sharp_frame).unsqueeze(0).to(device)

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
        extra_meta: Optional[Dict[str, Any]] = None
    ) -> Path:
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "step": step,
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "config": self.cfg,
            "arch_signature": {
                "num_feat": self.num_feat,
                "num_frame": self.num_frame,
                "num_extract_block": self.num_extract_block,
                "num_reconstruct_block": self.num_reconstruct_block,
                "center_frame_idx": self.center_frame_idx,
                "hr_in": self.hr_in,
                "with_predeblur": self.with_predeblur,
                "with_tsa": self.with_tsa,
                "deformable_groups": self.deformable_groups
            },
            "rng_state": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            }
        }
        if optimizer is not None:
            state["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler is not None and hasattr(scheduler, "state_dict"):
            state["scheduler_state_dict"] = scheduler.state_dict()
        if scaler is not None and hasattr(scaler, "state_dict"):
            state["scaler_state_dict"] = scaler.state_dict()
        if extra_meta:
            state["extra_meta"] = extra_meta

        torch.save(state, save_path)
        return save_path

    def load_checkpoint(
        self,
        path: Union[str, Path],
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
        strict: bool = True
    ) -> Dict[str, Any]:
        load_path = Path(path)
        if not load_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {load_path}")

        state = torch.load(load_path, map_location="cpu")
        if "model_state_dict" not in state:
            raise KeyError("Checkpoint missing required field: 'model_state_dict'")
        
        # Verify architecture signature compatibility
        saved_sig = state.get("arch_signature", {})
        if saved_sig:
            mismatches = []
            for k in [
                "num_feat", "num_frame", "num_extract_block", "num_reconstruct_block",
                "center_frame_idx", "hr_in", "with_predeblur", "with_tsa", "deformable_groups"
            ]:
                if k in saved_sig and getattr(self, k, None) != saved_sig[k]:
                    mismatches.append(f"{k} (current: {getattr(self, k, None)}, saved: {saved_sig[k]})")
            if mismatches:
                msg = f"Incompatible checkpoint architecture parameters: {', '.join(mismatches)}"
                if strict:
                    raise ValueError(msg)
                else:
                    import warnings
                    warnings.warn(msg)

        self.model.load_state_dict(state["model_state_dict"], strict=strict)

        if optimizer is not None and "optimizer_state_dict" in state:
            optimizer.load_state_dict(state["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in state and hasattr(scheduler, "load_state_dict"):
            scheduler.load_state_dict(state["scheduler_state_dict"])
        if scaler is not None and "scaler_state_dict" in state and hasattr(scaler, "load_state_dict"):
            scaler.load_state_dict(state["scaler_state_dict"])
        if "rng_state" in state and state["rng_state"]:
            rng = state["rng_state"]
            if isinstance(rng, dict):
                if "torch" in rng and rng["torch"] is not None:
                    torch.set_rng_state(rng["torch"])
                if "cuda" in rng and rng["cuda"] is not None and torch.cuda.is_available():
                    try:
                        torch.cuda.set_rng_state_all(rng["cuda"])
                    except Exception:
                        pass

        return {
            "step": state.get("step", 0),
            "epoch": state.get("epoch", 0),
            "config": state.get("config", {}),
            "extra_meta": state.get("extra_meta", {})
        }
