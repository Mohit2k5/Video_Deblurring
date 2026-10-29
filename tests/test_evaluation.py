"""Tests for evaluation metrics and tracker."""
import math
import pytest
import torch
from src.evaluation.metrics import (
    calculate_psnr,
    calculate_ssim,
    create_gaussian_window,
    calculate_charbonnier_loss,
    AggregateMetricsTracker
)


def test_calculate_psnr_identical():
    x = torch.rand(1, 3, 32, 32)
    psnr = calculate_psnr(x, x)
    assert math.isinf(psnr) and psnr > 0


def test_calculate_psnr_known_value():
    x = torch.zeros(1, 3, 10, 10)
    y = torch.full((1, 3, 10, 10), 0.1)
    # MSE = 0.01 -> PSNR = -10*log10(0.01) = 20.0 dB
    psnr = calculate_psnr(x, y)
    assert abs(psnr - 20.0) < 1e-4


def test_calculate_ssim_identical():
    x = torch.rand(1, 3, 64, 64)
    ssim = calculate_ssim(x, x)
    assert abs(ssim - 1.0) < 1e-4


def test_calculate_ssim_range():
    x = torch.rand(1, 3, 64, 64)
    y = torch.rand(1, 3, 64, 64)
    ssim = calculate_ssim(x, y)
    assert -1.0 <= ssim <= 1.0


def test_calculate_charbonnier_loss_zero():
    x = torch.rand(1, 3, 32, 32)
    loss = calculate_charbonnier_loss(x, x, eps=1e-12)
    assert loss < 1e-5


def test_aggregate_metrics_tracker():
    tracker = AggregateMetricsTracker()
    x1 = torch.zeros(1, 1, 10, 10)
    y1 = torch.full((1, 1, 10, 10), 0.1) # MSE=0.01 -> PSNR=20.0 dB (100 pixels)

    x2 = torch.zeros(1, 1, 10, 10)
    y2 = torch.full((1, 1, 10, 10), 0.01) # MSE=0.0001 -> PSNR=40.0 dB (100 pixels)

    tracker.update(x1, y1, psnr_val=20.0, ssim_val=0.8, loss_val=0.1)
    tracker.update(x2, y2, psnr_val=40.0, ssim_val=0.95, loss_val=0.01)

    summary = tracker.summary()
    assert summary["frame_count"] == 2
    assert summary["mean_psnr"] == 30.0
    assert summary["mean_ssim"] == (0.8 + 0.95) / 2
    # Combined MSE = (0.01*100 + 0.0001*100) / 200 = 0.00505
    # Agg PSNR = -10*log10(0.00505) = 22.967 dB
    expected_agg_psnr = -10.0 * math.log10(0.00505)
    assert abs(summary["aggregate_psnr"] - expected_agg_psnr) < 1e-2


def test_verify_held_out_split():
    from src.data.config_utils import get_project_root
    from src.evaluation.evaluate_held_out import verify_held_out_split
    root = get_project_root()
    seqs = verify_held_out_split(root)
    assert len(seqs) == 24
    assert len(set(seqs)) == 24


def test_verify_checkpoint():
    from src.data.config_utils import get_project_root
    from src.evaluation.evaluate_held_out import verify_checkpoint
    root = get_project_root()
    ckpt_path = root / "experiments" / "edvr_run_20261016_080811" / "checkpoints" / "best_checkpoint.pth"
    if not ckpt_path.exists():
        pytest.skip("Checkpoint not found")
    adapter, meta = verify_checkpoint(ckpt_path, project_root=root)
    assert meta["step"] == 1000
    assert adapter.num_feat == 64

