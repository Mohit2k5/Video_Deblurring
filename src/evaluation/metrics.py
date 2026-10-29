"""Evaluation metrics for REDS Video Deblurring.
Includes standard, mathematically verified implementations of:
- Peak Signal-to-Noise Ratio (PSNR) with aggregate pixel MSE and per-frame calculation
- Structural Similarity Index Measure (SSIM) based on Wang et al. (2004) with circular Gaussian window
- Differentiable Charbonnier Loss
- Metric aggregation structures for dataset-wide benchmarking
"""
import math
from typing import Dict, Any, Optional, Tuple, List
import torch
import torch.nn.functional as F


def calculate_psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Calculate Peak Signal-to-Noise Ratio (PSNR) in dB for tensors in range [0, 1].
    
    Args:
        pred: Predicted tensor (C, H, W) or (B, C, H, W)
        gt: Ground-truth tensor with same shape
        
    Returns:
        float: PSNR in dB, or float('inf') if MSE == 0
    """
    pred_c = pred.clamp(0.0, 1.0)
    gt_c = gt.clamp(0.0, 1.0)
    mse = torch.mean((pred_c - gt_c) ** 2).item()
    if mse == 0.0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def create_gaussian_window(
    window_size: int = 11,
    sigma: float = 1.5,
    channels: int = 3,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """Generate normalized 2D circular Gaussian window matching Wang et al. (2004)."""
    gauss = torch.tensor(
        [math.exp(-(x - window_size // 2) ** 2 / (2 * sigma ** 2)) for x in range(window_size)],
        dtype=torch.float32,
        device=device
    )
    gauss = gauss / gauss.sum()
    _1d = gauss.unsqueeze(1)
    _2d = _1d.mm(_1d.t()).unsqueeze(0).unsqueeze(0)
    return _2d.expand(channels, 1, window_size, window_size).contiguous()


def calculate_ssim(
    pred: torch.Tensor,
    gt: torch.Tensor,
    window: Optional[torch.Tensor] = None,
    window_size: int = 11,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
    data_range: float = 1.0
) -> float:
    """Calculate mean Structural Similarity Index Measure (SSIM) for tensors in [0, 1].
    
    Computes per-channel SSIM with an 11x11 circular Gaussian window (sigma=1.5) and averages
    across spatial dimensions and RGB channels.
    """
    pred_c = pred.clamp(0.0, 1.0)
    gt_c = gt.clamp(0.0, 1.0)

    if pred_c.ndim == 3:
        pred_c = pred_c.unsqueeze(0)
    if gt_c.ndim == 3:
        gt_c = gt_c.unsqueeze(0)

    b, c, h, w = pred_c.shape
    if window is None or window.device != pred_c.device or window.shape[0] != c:
        window = create_gaussian_window(window_size, sigma, c, device=pred_c.device)

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    pad = window_size // 2

    mu1 = F.conv2d(pred_c, window, padding=pad, groups=c)
    mu2 = F.conv2d(gt_c, window, padding=pad, groups=c)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred_c * pred_c, window, padding=pad, groups=c) - mu1_sq
    sigma2_sq = F.conv2d(gt_c * gt_c, window, padding=pad, groups=c) - mu2_sq
    sigma12 = F.conv2d(pred_c * gt_c, window, padding=pad, groups=c) - mu1_mu2

    ssim_map = ((2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return ssim_map.mean().item()


def calculate_charbonnier_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    eps: float = 1e-12
) -> float:
    """Calculate mean Charbonnier loss: sqrt((pred - gt)^2 + eps)."""
    diff = pred - gt
    loss = torch.mean(torch.sqrt(diff * diff + eps))
    return loss.item()


class AggregateMetricsTracker:
    """Tracks per-frame metrics and computes pixel-weighted aggregate PSNR."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.squared_error_sum = 0.0
        self.pixel_count = 0
        self.frame_count = 0
        self.psnr_sum = 0.0
        self.ssim_sum = 0.0
        self.loss_sum = 0.0

    def update(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        psnr_val: float,
        ssim_val: float,
        loss_val: float
    ):
        pred_f = pred.float().clamp(0.0, 1.0)
        gt_f = gt.float().clamp(0.0, 1.0)
        sq_err = (pred_f - gt_f).square().sum().item()
        num_pixels = gt_f.numel()

        self.squared_error_sum += sq_err
        self.pixel_count += num_pixels
        self.frame_count += 1
        self.psnr_sum += psnr_val
        self.ssim_sum += ssim_val
        self.loss_sum += loss_val

    def summary(self) -> Dict[str, float]:
        if self.frame_count == 0 or self.pixel_count == 0:
            return {
                "aggregate_psnr": 0.0,
                "mean_psnr": 0.0,
                "mean_ssim": 0.0,
                "mean_charbonnier_loss": 0.0,
                "frame_count": 0
            }

        agg_mse = self.squared_error_sum / self.pixel_count
        agg_psnr = float("inf") if agg_mse == 0.0 else -10.0 * math.log10(agg_mse)

        return {
            "aggregate_psnr": round(agg_psnr, 4),
            "mean_psnr": round(self.psnr_sum / self.frame_count, 4),
            "mean_ssim": round(self.ssim_sum / self.frame_count, 4),
            "mean_charbonnier_loss": round(self.loss_sum / self.frame_count, 6),
            "frame_count": self.frame_count
        }
