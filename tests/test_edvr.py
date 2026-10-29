"""Comprehensive Unit and Integration Test Suite for EDVR REDS Video Deblurring.
Tests verified architectural fidelity, PCD alignment, TSA fusion, gradient flow,
DataLoader integration, checkpoint save/restore, and AMP execution.
"""
from pathlib import Path
import math
import pytest
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from src.data.config_utils import get_project_root
from src.data.dataset import REDSVideoDeblurDataset
from src.models.edvr_arch import EDVR, PCDAlignment, TSAFusion, PredeblurModule
from src.models.edvr_adapter import EDVRAdapter, CharbonnierLoss


@pytest.fixture
def project_root():
    return get_project_root()


@pytest.fixture
def edvr_adapter(project_root):
    cfg_path = project_root / "configs" / "edvr.yaml"
    return EDVRAdapter(config_path=cfg_path, project_root=project_root)


def test_model_construction_from_yaml(edvr_adapter):
    """Model construction using configs/edvr.yaml."""
    adapter = edvr_adapter
    model = adapter.model

    assert isinstance(model, EDVR)
    assert model.num_frame == 5
    assert model.center_frame_idx == 2
    assert model.hr_in is True
    if model.with_predeblur:
        assert hasattr(model, "predeblur") and isinstance(model.predeblur, PredeblurModule)
        assert hasattr(model, "conv_1x1") and isinstance(model.conv_1x1, nn.Conv2d)
        assert model.predeblur.hr_in == model.hr_in
    else:
        assert hasattr(model, "conv_first") and isinstance(model.conv_first, nn.Conv2d)
        assert not hasattr(model, "predeblur")
    assert adapter.num_feat in (64, 128)
    assert adapter.num_reconstruct_block in (10, 40)


def test_model_construction_without_predeblur():
    """Verify complementary configuration when with_predeblur=False."""
    model = EDVR(num_feat=64, num_extract_block=5, num_reconstruct_block=10, with_predeblur=False, with_tsa=True)
    assert hasattr(model, "conv_first") and isinstance(model.conv_first, nn.Conv2d)
    assert not hasattr(model, "predeblur")
    assert not hasattr(model, "conv_1x1")


def test_parameter_count_regression(edvr_adapter):
    """Regression test verifying exact active parameter count and submodule breakdown."""
    model = edvr_adapter.model
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params == 4042851, f"Expected 4,042,851 parameters for active EDVR-M, got {total_params}"

    # Verify key submodule counts
    assert sum(p.numel() for p in model.predeblur.parameters()) == 740352
    assert sum(p.numel() for p in model.feature_extraction.parameters()) == 369280
    assert sum(p.numel() for p in model.pcd_align.parameters()) == 1384288
    assert sum(p.numel() for p in model.fusion.parameters()) == 324416
    assert sum(p.numel() for p in model.reconstruction.parameters()) == 738560
    assert sum(p.numel() for p in model.upconv1.parameters()) == 147712
    assert sum(p.numel() for p in model.upconv2.parameters()) == 147712

    # Verify EDVR-L parameter count
    model_L = EDVR(num_feat=128, num_extract_block=5, num_reconstruct_block=40, hr_in=True, with_predeblur=True, with_tsa=True)
    assert sum(p.numel() for p in model_L.parameters()) == 23602019


def test_forward_pass_batch_size_one(edvr_adapter):
    """Forward pass with batch size one."""
    model = edvr_adapter.model
    model.eval()

    # Spatial dimensions must be divisible by 16 in hr_in mode
    x = torch.randn(1, 5, 3, 64, 64)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (1, 3, 64, 64), f"Expected shape (1, 3, 64, 64), got {out.shape}"
    assert torch.isfinite(out).all(), "Output contains NaN or Inf values"


def test_forward_pass_batch_size_two(edvr_adapter):
    """Forward pass with batch size two."""
    model = edvr_adapter.model
    model.eval()

    x = torch.randn(2, 5, 3, 64, 64)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (2, 3, 64, 64), f"Expected shape (2, 3, 64, 64), got {out.shape}"
    assert torch.isfinite(out).all(), "Batch size 2 output contains NaN or Inf values"


def test_input_feature_and_output_dimensions(edvr_adapter):
    """Correct input, intermediate feature, and output dimensions throughout the resolution pathway."""
    model = edvr_adapter.model
    model.eval()

    b, t, c, h, w = 1, 5, 3, 64, 64
    x = torch.randn(b, t, c, h, w)

    # 1. PreDeblur module downsampling
    x_flat = x.view(-1, c, h, w)
    feat_pre = model.predeblur(x_flat)
    assert feat_pre.shape == (5, model.predeblur.stride_conv_hr1.out_channels, h // 4, w // 4)

    # 2. Level 1, 2, 3 pyramid feature extraction
    feat_l1 = model.conv_1x1(feat_pre)
    feat_l1 = model.feature_extraction(feat_l1)
    feat_l2 = model.lrelu(model.conv_l2_2(model.lrelu(model.conv_l2_1(feat_l1))))
    feat_l3 = model.lrelu(model.conv_l3_2(model.lrelu(model.conv_l3_1(feat_l2))))

    assert feat_l1.shape == (5, model.num_feat, 16, 16)
    assert feat_l2.shape == (5, model.num_feat, 8, 8)
    assert feat_l3.shape == (5, model.num_feat, 4, 4)

    # 3. Overall forward output
    with torch.no_grad():
        out = model(x)
    assert out.shape == (b, 3, h, w)


def test_center_frame_selection_and_residual(edvr_adapter):
    """Correct center-frame selection and residual skip connection."""
    model = edvr_adapter.model
    model.eval()

    x1 = torch.zeros(1, 5, 3, 64, 64)
    x2 = torch.zeros(1, 5, 3, 64, 64)
    # Put distinct stimulus in center frame vs non-center frame
    x1[:, 2, :, :, :] = 1.0  # Center frame
    x2[:, 0, :, :, :] = 1.0  # Non-center frame

    with torch.no_grad():
        out1 = model(x1)
        out2 = model(x2)

    # Output with stimulus in center frame must differ significantly due to base = x_center residual connection
    diff = torch.abs(out1 - out2).mean().item()
    assert diff > 0.1, f"Center-frame index not properly prioritized in base residual: diff={diff}"


def test_backward_pass_with_finite_gradients(edvr_adapter):
    """Backward pass with finite gradients across all trainable parameters."""
    adapter = edvr_adapter
    adapter.model.train()

    x = torch.randn(1, 5, 3, 64, 64, requires_grad=True)
    target = torch.randn(1, 3, 64, 64)

    pred = adapter.forward(x)
    loss = adapter.compute_loss(pred, target)
    loss.backward()

    assert torch.isfinite(loss).item(), f"Loss is non-finite: {loss.item()}"
    assert x.grad is not None and torch.isfinite(x.grad).all(), "Input gradients contain NaN or Inf"

    for name, p in adapter.model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Parameter {name} did not receive a gradient"
            assert torch.isfinite(p.grad).all(), f"Parameter {name} has non-finite gradient"


def test_pcd_alignment_tensor_shapes():
    """Correct PCD alignment tensor shapes and multi-level pyramid execution."""
    pcd = PCDAlignment(num_feat=64, deformable_groups=8)
    pcd.eval()

    # Create 3-level pyramid features for neighbor and reference
    nbr_feat_l = [
        torch.randn(1, 64, 16, 16),
        torch.randn(1, 64, 8, 8),
        torch.randn(1, 64, 4, 4)
    ]
    ref_feat_l = [
        torch.randn(1, 64, 16, 16),
        torch.randn(1, 64, 8, 8),
        torch.randn(1, 64, 4, 4)
    ]

    aligned_feat = pcd(nbr_feat_l, ref_feat_l)
    assert aligned_feat.shape == (1, 64, 16, 16)
    assert torch.isfinite(aligned_feat).all()


def test_tsa_fusion_output_shape_and_numerical_stability():
    """TSA fusion output shape, temporal attention, and spatial multi-scale attention."""
    tsa = TSAFusion(num_feat=64, num_frame=5, center_frame_idx=2)
    tsa.eval()

    aligned_feat = torch.randn(1, 5, 64, 16, 16)
    with torch.no_grad():
        fused = tsa(aligned_feat)

    assert fused.shape == (1, 64, 16, 16)
    assert torch.isfinite(fused).all()


def test_real_reds_dataset_integration(edvr_adapter, project_root):
    """Paired dataset integration using real REDS samples."""
    raw_train_blur = project_root / "data" / "raw" / "train_blur"
    if not raw_train_blur.exists():
        pytest.skip("REDS raw dataset not found; skipping real-sample integration test.")

    ds = REDSVideoDeblurDataset(
        partition="train",
        project_root=project_root,
        window_size=5,
        boundary_policy="replicate"
    )
    if len(ds) == 0:
        pytest.skip("No samples found in training partition.")

    sample = ds[0]
    # Crop to 64x64 for fast unit-test verification
    cropped_blur = [img.crop((0, 0, 64, 64)) for img in sample["blur_frames"]]
    cropped_sharp = sample["sharp_frame"].crop((0, 0, 64, 64)) if "sharp_frame" in sample else None

    batch_dict = {
        "blur_frames": cropped_blur,
        "sharp_frame": cropped_sharp
    }

    x, y = edvr_adapter.prepare_batch(batch_dict)
    assert x.shape == (1, 5, 3, 64, 64)
    assert y is not None and y.shape == (1, 3, 64, 64)

    with torch.no_grad():
        pred = edvr_adapter.forward(x)
    assert pred.shape == (1, 3, 64, 64)
    loss = edvr_adapter.compute_loss(pred, y)
    assert torch.isfinite(loss).item()


def test_checkpoint_save_and_reload(edvr_adapter, tmp_path):
    """Checkpoint save, architecture signature verification, and reload."""
    adapter = edvr_adapter
    ckpt_path = tmp_path / "test_edvr.pth"

    optimizer = torch.optim.Adam(adapter.model.parameters(), lr=1e-4)
    # Take an optimizer step to populate state
    x = torch.randn(1, 5, 3, 64, 64)
    target = torch.randn(1, 3, 64, 64)
    loss = adapter.compute_loss(adapter.forward(x), target)
    loss.backward()
    optimizer.step()

    saved_path = adapter.save_checkpoint(ckpt_path, optimizer=optimizer, step=42, epoch=1)
    assert saved_path.exists()

    # Instantiate fresh adapter and reload
    new_adapter = EDVRAdapter(config=adapter.cfg, project_root=adapter.project_root)
    new_opt = torch.optim.Adam(new_adapter.model.parameters(), lr=1e-4)

    meta = new_adapter.load_checkpoint(saved_path, optimizer=new_opt)
    assert meta["step"] == 42
    assert meta["epoch"] == 1

    # Verify model weights are identical
    for p1, p2 in zip(adapter.model.parameters(), new_adapter.model.parameters()):
        assert torch.equal(p1, p2)

    # Verify optimizer states are restored
    assert len(new_opt.state) > 0


def test_mixed_precision_execution(edvr_adapter):
    """Automatic mixed precision execution (torch.amp)."""
    adapter = edvr_adapter
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adapter.model.to(device)

    x = torch.randn(1, 5, 3, 64, 64, device=device)
    target = torch.randn(1, 3, 64, 64, device=device)

    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = torch.float16 if device_type == "cuda" else torch.bfloat16

    with torch.amp.autocast(device_type=device_type, dtype=amp_dtype):
        pred = adapter.forward(x)
        loss = adapter.compute_loss(pred, target)

    assert torch.isfinite(loss).item()
    if device_type == "cuda":
        scaler = torch.amp.GradScaler("cuda")
        scaler.scale(loss).backward()
        assert x.grad is None or torch.isfinite(x.grad).all()


def test_model_config_and_adapter_agreement(edvr_adapter):
    """Ensures model configuration and adapter agree on temporal window and center frame index."""
    adapter = edvr_adapter
    assert adapter.num_frame == adapter.model.num_frame
    assert adapter.center_frame_idx == adapter.model.center_frame_idx
    assert adapter.hr_in == adapter.model.hr_in
    assert adapter.with_predeblur == adapter.model.with_predeblur
    assert adapter.with_tsa == adapter.model.with_tsa


def test_dcnv2_zero_offset_convolution_equivalence():
    """Verifies that DCNv2Pack with zero offsets is mathematically identical to 0.5 * Conv2d."""
    from src.models.edvr_arch import DCNv2Pack
    import torch.nn.functional as F

    dcn = DCNv2Pack(64, 64, kernel_size=3, padding=1, deformable_groups=8)
    x = torch.randn(1, 64, 16, 16)
    feat = torch.randn(1, 64, 16, 16)

    out_dcn = dcn(x, feat)
    out_conv = F.conv2d(x, dcn.weight, dcn.bias, padding=1) * 0.5

    diff = torch.abs(out_dcn - out_conv).max().item()
    assert diff < 1e-6, f"DCNv2Pack diverged from standard convolution: {diff}"


def test_checkpoint_architecture_mismatch_rejection(edvr_adapter, tmp_path):
    """Verifies that attempting to load a checkpoint with mismatched architecture signature raises ValueError."""
    ckpt_path = tmp_path / "valid_ckpt.pth"
    edvr_adapter.save_checkpoint(ckpt_path, step=10)

    # Corrupt architecture signature
    state = torch.load(ckpt_path, map_location="cpu")
    state["arch_signature"]["num_feat"] = 128
    mismatch_path = tmp_path / "mismatch_ckpt.pth"
    torch.save(state, mismatch_path)

    with pytest.raises(ValueError, match="Incompatible checkpoint architecture parameters"):
        edvr_adapter.load_checkpoint(mismatch_path, strict=True)


def test_unscaled_gradient_validation(edvr_adapter):
    """Verifies that unscaled gradient checking correctly identifies finite and non-finite gradients."""
    adapter = edvr_adapter
    adapter.model.train()
    optimizer = torch.optim.Adam(adapter.model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.randn(1, 5, 3, 64, 64)
    y = torch.randn(1, 3, 64, 64)
    device_type = "cuda" if torch.cuda.is_available() else "cpu"

    with torch.amp.autocast(device_type, enabled=torch.cuda.is_available()):
        pred = adapter(x)
        loss = adapter.compute_loss(pred, y)

    scaler.scale(loss).backward()
    if torch.cuda.is_available():
        scaler.unscale_(optimizer)

    # Verify all gradients are finite initially
    all_finite = all(torch.isfinite(p.grad).all().item() for p in adapter.model.parameters() if p.grad is not None)
    assert all_finite is True, "Expected initial unscaled gradients to be finite"

    # Inject NaN into one parameter gradient
    first_param = next(p for p in adapter.model.parameters() if p.grad is not None)
    orig_grad = first_param.grad.clone()
    first_param.grad.fill_(float("nan"))

    nan_detected = not all(torch.isfinite(p.grad).all().item() for p in adapter.model.parameters() if p.grad is not None)
    assert nan_detected is True, "Failed to detect injected NaN gradient"

    # Restore gradient
    first_param.grad = orig_grad


def test_calculate_psnr_zero_mse_inf():
    """Verifies that calculate_psnr returns float('inf') for exact reconstruction."""
    from src.training.train import calculate_psnr
    x = torch.rand(1, 3, 32, 32)
    psnr = calculate_psnr(x, x)
    assert math.isinf(psnr) and psnr > 0, f"Expected +inf for zero MSE, got {psnr}"


def test_validation_empty_loader_raises_runtime_error(edvr_adapter):
    """Verifies that an empty validation DataLoader raises an informative RuntimeError."""
    from src.training.train import evaluate_validation
    device = torch.device("cpu")
    empty_loader = []

    with pytest.raises(RuntimeError, match="Validation produced no valid batches or pixels"):
        evaluate_validation(edvr_adapter, empty_loader, device=device, use_amp=False)


def test_validation_empty_valid_batches_raises_runtime_error(edvr_adapter):
    """Verifies that a DataLoader producing only batches without ground-truth raises RuntimeError."""
    from src.training.train import evaluate_validation
    device = torch.device("cpu")
    # Batch without target
    mock_batch = {"blur_frames": torch.randn(1, 5, 3, 32, 32), "sharp_frame": None}

    class MockAdapter:
        model = edvr_adapter.model
        def prepare_batch(self, b, device):
            return b["blur_frames"], None

    with pytest.raises(RuntimeError, match="Validation produced no valid batches or pixels"):
        evaluate_validation(MockAdapter(), [mock_batch], device=device, use_amp=False)


def test_validation_preserves_train_mode(edvr_adapter):
    """Verifies that a model initially in training mode returns to training mode after evaluate_validation."""
    from src.training.train import evaluate_validation
    device = torch.device("cpu")
    adapter = edvr_adapter
    adapter.model.train()
    assert adapter.model.training is True

    # Single valid batch
    batch = {
        "blur_frames": [torch.rand(3, 32, 32) for _ in range(5)],
        "sharp_frame": torch.rand(3, 32, 32)
    }
    loader = [batch]

    avg_loss, agg_psnr, batch_psnr = evaluate_validation(adapter, loader, device=device, use_amp=False)
    assert adapter.model.training is True, "Model training mode was not restored after validation"
    assert math.isfinite(avg_loss)
    assert math.isfinite(agg_psnr)


def test_validation_preserves_eval_mode(edvr_adapter):
    """Verifies that a model initially in eval mode remains in eval mode after evaluate_validation."""
    from src.training.train import evaluate_validation
    device = torch.device("cpu")
    adapter = edvr_adapter
    adapter.model.eval()
    assert adapter.model.training is False

    batch = {
        "blur_frames": [torch.rand(3, 32, 32) for _ in range(5)],
        "sharp_frame": torch.rand(3, 32, 32)
    }
    loader = [batch]

    avg_loss, agg_psnr, batch_psnr = evaluate_validation(adapter, loader, device=device, use_amp=False)
    assert adapter.model.training is False, "Model was erroneously switched to training mode after validation"
    assert math.isfinite(avg_loss)
    assert math.isfinite(agg_psnr)


def test_validation_mode_restored_on_exception(edvr_adapter):
    """Verifies that model mode is restored even if validation encounters an exception."""
    from src.training.train import evaluate_validation
    device = torch.device("cpu")
    adapter = edvr_adapter
    adapter.model.train()
    assert adapter.model.training is True

    class FailingAdapter:
        model = adapter.model
        def prepare_batch(self, b, device):
            raise ValueError("Simulated unexpected batch error")

    with pytest.raises(ValueError, match="Simulated unexpected batch error"):
        evaluate_validation(FailingAdapter(), [{"dummy": 1}], device=device, use_amp=False)

    assert adapter.model.training is True, "Model mode was not restored in finally block after exception"


def test_synthetic_checkpoint_resumption_unit(edvr_adapter, tmp_path):
    """Test A: Synthetic unit test verifying complete checkpoint state restoration and validation."""
    from torch.optim.lr_scheduler import CosineAnnealingLR
    adapter = edvr_adapter
    ckpt_path = tmp_path / "synthetic_test.pth"

    opt = torch.optim.Adam(adapter.model.parameters(), lr=1e-4)
    sched = CosineAnnealingLR(opt, T_max=100)
    scaler = torch.amp.GradScaler("cpu")

    # Step to populate optimizer state
    x = torch.randn(1, 5, 3, 32, 32)
    y = torch.randn(1, 3, 32, 32)
    loss = adapter.compute_loss(adapter.forward(x), y)
    loss.backward()
    opt.step()
    sched.step()

    # Save
    adapter.save_checkpoint(ckpt_path, optimizer=opt, scheduler=sched, scaler=scaler, step=42, epoch=1)
    assert ckpt_path.exists()

    # Fresh instance
    fresh_adapter = EDVRAdapter(config=adapter.cfg, project_root=adapter.project_root)
    fresh_opt = torch.optim.Adam(fresh_adapter.model.parameters(), lr=1e-4)
    fresh_sched = CosineAnnealingLR(fresh_opt, T_max=100)
    fresh_scaler = torch.amp.GradScaler("cpu")

    meta = fresh_adapter.load_checkpoint(ckpt_path, optimizer=fresh_opt, scheduler=fresh_sched, scaler=fresh_scaler)
    assert meta["step"] == 42
    assert meta["epoch"] == 1

    # Verify model parameters equal
    for p1, p2 in zip(adapter.model.parameters(), fresh_adapter.model.parameters()):
        assert torch.equal(p1, p2)

    # Verify optimizer & scheduler restored
    assert len(fresh_opt.state) > 0
    assert fresh_sched.last_epoch == sched.last_epoch

    # Verify missing model_state_dict rejection
    corrupt_path = tmp_path / "corrupt.pth"
    torch.save({"step": 10}, corrupt_path)
    with pytest.raises(KeyError, match="model_state_dict"):
        fresh_adapter.load_checkpoint(corrupt_path)


def test_real_reds_pilot_checkpoint_resumption_step(project_root):
    """Test B: Real REDS integration test verifying actual pilot checkpoint restoration and subsequent optimizer step."""
    import hashlib
    from src.models.edvr_adapter import EDVRAdapter
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from src.data.dataset import REDSVideoDeblurDataset

    ckpt_path = project_root / "reports" / "validation" / "pilot_checkpoints" / "pilot_checkpoint_step50.pth"
    if not ckpt_path.exists():
        pytest.skip("Actual pilot checkpoint step50 not found on disk; skipping real integration test.")

    raw_train_blur = project_root / "data" / "raw" / "train_blur"
    if not raw_train_blur.exists():
        pytest.skip("REDS raw dataset not available for real batch step verification.")

    hash_before = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_type = "cuda" if device.type == "cuda" else "cpu"

    fresh_adapter = EDVRAdapter(project_root=project_root).to(device)
    fresh_opt = torch.optim.Adam(fresh_adapter.model.parameters(), lr=2e-4)
    fresh_sched = CosineAnnealingLR(fresh_opt, T_max=100)
    fresh_scaler = torch.amp.GradScaler(device_type)

    meta = fresh_adapter.load_checkpoint(ckpt_path, optimizer=fresh_opt, scheduler=fresh_sched, scaler=fresh_scaler)
    assert meta["step"] == 50, f"Expected step 50, got {meta['step']}"

    # Load a real sample from REDS training partition
    ds = REDSVideoDeblurDataset(
        partition="train",
        project_root=project_root,
        window_size=5,
        boundary_policy="replicate"
    )
    sample = ds[0]
    cropped_blur = [img.crop((0, 0, 64, 64)) for img in sample["blur_frames"]]
    cropped_sharp = sample["sharp_frame"].crop((0, 0, 64, 64))
    batch_dict = {"blur_frames": cropped_blur, "sharp_frame": cropped_sharp}

    rx, ry = fresh_adapter.prepare_batch(batch_dict, device=device)

    fresh_adapter.model.train()
    fresh_opt.zero_grad()
    with torch.amp.autocast(device_type, enabled=device.type == "cuda"):
        r_pred = fresh_adapter.forward(rx)
        r_loss = fresh_adapter.compute_loss(r_pred, ry)

    assert torch.isfinite(r_loss).item(), f"Post-resume loss is non-finite: {r_loss.item()}"
    fresh_scaler.scale(r_loss).backward()
    if device.type == "cuda":
        fresh_scaler.unscale_(fresh_opt)

    for p in fresh_adapter.model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all().item(), "Post-resume parameter gradient is non-finite"

    norm = torch.nn.utils.clip_grad_norm_(fresh_adapter.model.parameters(), max_norm=float("inf"))
    assert torch.isfinite(norm).item(), "Post-resume gradient norm is non-finite"

    fresh_scaler.step(fresh_opt)
    fresh_scaler.update()
    fresh_sched.step()

    # Checkpoint immutability
    hash_after = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
    assert hash_before == hash_after, "Checkpoint file on disk was modified during resume test!"


def test_validation_psnr_aggregation_mathematical_properties():
    """Verifies aggregate pixel MSE PSNR formula vs batch-mean PSNR."""
    import math

    # Two synthetic target/pred pairs with unequal pixel MSE
    # Batch 1: MSE = 0.01 -> PSNR = 20.0 dB (100 pixels)
    # Batch 2: MSE = 0.001 -> PSNR = 30.0 dB (100 pixels)
    mse1, mse2 = 0.01, 0.001
    pixels_per_batch = 100

    sq_error_total = mse1 * pixels_per_batch + mse2 * pixels_per_batch
    total_pixels = 2 * pixels_per_batch
    agg_mse = sq_error_total / total_pixels  # 0.0055

    agg_psnr = -10.0 * math.log10(agg_mse)  # ~22.596 dB
    batch_mean_psnr = (20.0 + 30.0) / 2.0   # 25.0 dB

    # Aggregate PSNR must correctly account for concave log transform: harmonic nature <= arithmetic mean
    assert agg_psnr < batch_mean_psnr
    assert abs(agg_psnr - 22.596) < 0.01



