"""Automated test suite for MSVSSM (Motion-aware Visual State Space Model)."""
import tempfile
from pathlib import Path
import pytest
import torch

from src.models.msvssm_arch import MSVSSM
from src.models.msvssm_adapter import MSVSSMAdapter
from src.evaluation.metrics import calculate_psnr, calculate_ssim, calculate_charbonnier_loss


@pytest.mark.parametrize("algo", ["base", "fft", "tsff"])
def test_msvssm_construction_and_shapes(algo):
    """Verify MSVSSM initializes and produces correct output dimensions for all 3 algorithms."""
    model = MSVSSM(num_feat=32, num_blocks=2, num_frame=5, algo=algo)
    x = torch.randn(1, 5, 3, 64, 64)
    out = model(x)
    assert out.shape == (1, 3, 64, 64), f"Expected (1, 3, 64, 64), got {out.shape}"


@pytest.mark.parametrize("algo", ["base", "fft", "tsff"])
def test_msvssm_backward_pass(algo):
    """Verify gradients propagate through all trainable parameters."""
    model = MSVSSM(num_feat=32, num_blocks=2, num_frame=5, algo=algo)
    x = torch.randn(1, 5, 3, 64, 64, requires_grad=True)
    out = model(x)
    loss = out.sum()
    loss.backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Parameter {name} has None gradient"
            assert not torch.isnan(p.grad).any(), f"Parameter {name} has NaN gradient"


def test_msvssm_checkpoint_save_and_load():
    """Verify saving and reloading MSVSSM checkpoint."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "test_msvssm.pth"
        adapter = MSVSSMAdapter(algo="base")
        opt = torch.optim.Adam(adapter.model.parameters(), lr=1e-4)

        saved = adapter.save_checkpoint(ckpt_path, optimizer=opt, step=42)
        assert saved.exists()

        new_adapter = MSVSSMAdapter(algo="base")
        new_opt = torch.optim.Adam(new_adapter.model.parameters(), lr=1e-4)
        meta = new_adapter.load_checkpoint(saved, optimizer=new_opt)

        assert meta["step"] == 42
        for p1, p2 in zip(adapter.model.parameters(), new_adapter.model.parameters()):
            assert torch.allclose(p1, p2)
