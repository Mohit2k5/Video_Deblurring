"""Unit tests for PGDN (Prior-Guided Deblurring Network, 2025)."""
import tempfile
from pathlib import Path
import pytest
import torch
import numpy as np

from src.models.pgdn_arch import (
    PGDN,
    MotionPriorGuidanceModule,
    ContextualTemporalBranch,
    MotionGuidedDetailBranch,
    CrossBranchInteractiveFusion
)
from src.models.pgdn_adapter import PGDNAdapter, CharbonnierLoss


def test_pgdn_architecture_forward():
    model = PGDN(num_frames=5, center_frame_idx=2, in_channels=3, embed_dim=32, num_blocks=2, prior_dim=16)
    x = torch.rand(2, 5, 3, 64, 64)
    out = model(x)
    assert out.shape == (2, 3, 64, 64)
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_mpgm_components():
    mpgm = MotionPriorGuidanceModule(num_frames=5, center_idx=2, in_channels=3, prior_dim=16)
    x = torch.rand(2, 5, 3, 64, 64)
    res = mpgm(x)
    assert "mask" in res
    assert "length" in res
    assert "angle" in res
    assert "curvature" in res
    assert "guidance" in res
    assert res["mask"].shape == (2, 1, 64, 64)
    assert (res["mask"] >= 0.0).all() and (res["mask"] <= 1.0).all()


def test_dual_branch_and_fusion():
    embed_dim = 32
    prior_dim = 16
    ctx_b = ContextualTemporalBranch(in_channels=3, num_frames=5, embed_dim=embed_dim)
    det_b = MotionGuidedDetailBranch(in_channels=3, prior_dim=prior_dim, embed_dim=embed_dim)
    fusion = CrossBranchInteractiveFusion(embed_dim=embed_dim)

    x = torch.rand(2, 5, 3, 64, 64)
    center = x[:, 2]
    motion_info = {
        "mask": torch.rand(2, 1, 64, 64),
        "guidance": torch.rand(2, prior_dim, 64, 64)
    }

    f_ctx = ctx_b(x)
    f_det = det_b(center, motion_info)
    fused = fusion(f_ctx, f_det)

    assert f_ctx.shape == (2, embed_dim, 64, 64)
    assert f_det.shape == (2, embed_dim, 64, 64)
    assert fused.shape == (2, embed_dim, 64, 64)


def test_pgdn_backward_gradient_flow():
    model = PGDN(num_frames=5, center_frame_idx=2, in_channels=3, embed_dim=32, num_blocks=2, prior_dim=16)
    x = torch.rand(1, 5, 3, 48, 48)
    target = torch.rand(1, 3, 48, 48)
    criterion = CharbonnierLoss()

    out = model(x)
    loss = criterion(out, target)
    loss.backward()

    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Parameter {name} has no gradient"
            assert not torch.isnan(p.grad).any(), f"Parameter {name} gradient contains NaN"


def test_pgdn_adapter_checkpoint_roundtrip():
    adapter = PGDNAdapter()
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "test_pgdn.pth"
        adapter.save_checkpoint(ckpt_path, step=100)
        assert ckpt_path.exists()

        new_adapter = PGDNAdapter()
        ckpt = new_adapter.load_checkpoint(ckpt_path)
        assert ckpt["step"] == 100
        assert ckpt["architecture"] == "PGDN-2025"
