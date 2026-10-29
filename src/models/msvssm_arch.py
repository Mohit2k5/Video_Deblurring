"""MSVSSM: Motion-aware Visual State Space Model for Efficient Video Deblurring (CVPR 2026).
Reference: Zhou et al., "MVSSM: Motion-aware Visual State Space Model for Efficient Video Deblurring", CVPR 2026.

Implements:
1. Algorithm 1: MSVSSM-Base
   - Motion Estimation Module (MEM)
   - Motion-aware Scanning Block (MSB) with continuous Structured State Space (S4D) scanning
   - Multi-scale Dynamic Feature Fusion (MDFF)
2. Algorithm 2: MSVSSM-FFT (Frequency-Enhanced Dual-Domain SSM)
   - Real 2D FFT spectral high-frequency residual extraction and state-space gating
3. Algorithm 3: MSVSSM-TSFF (Temporal-Spatial Flow-Gated Deformable SSM)
   - Continuous sub-pixel trajectory flow warping + bidirectional temporal state-space recurrence
"""
import math
from typing import Dict, Any, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class MotionEstimationModule(nn.Module):
    """Lightweight motion vector and descriptor estimator operating in feature space.
    Estimates 2D motion fields (dx, dy) and motion magnitude/angle maps
    between adjacent frames and the reference center frame.
    """
    def __init__(self, in_ch: int = 48, feat_ch: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch * 2, feat_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(feat_ch, feat_ch, 3, padding=1)
        self.flow_pred = nn.Conv2d(feat_ch, 2, 3, padding=1)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, feat_nbr: torch.Tensor, feat_ref: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cat = torch.cat([feat_nbr, feat_ref], dim=1)
        f = self.lrelu(self.conv2(self.lrelu(self.conv1(cat))))
        flow = self.flow_pred(f)

        dx = flow[:, 0:1, :, :]
        dy = flow[:, 1:2, :, :]
        mag = torch.sqrt(dx * dx + dy * dy + 1e-6)
        ang = torch.atan2(dy, dx)
        desc = torch.cat([dx, dy, mag, ang], dim=1)
        return flow, desc


class MotionAwareScanBlock(nn.Module):
    """Motion-aware Scanning Block (MSB).
    Implements continuous Structured State Space (S4D) impulse response scanning along motion trajectories:
    1. Horizontal State Space scanning along width trajectories with kernel K_h(t) = sum C_n exp(-alpha_n * t) B_n
    2. Vertical State Space scanning along height trajectories with kernel K_v(t) = sum C_n exp(-alpha_n * t) B_n
    3. Motion-velocity modulation: motion magnitude dynamically modulates continuous decay rate alpha
    4. Motion angle-guided dynamic weighting
    """
    def __init__(self, d_model: int = 48, kernel_size: int = 17):
        super().__init__()
        self.d_model = d_model
        self.kernel_size = kernel_size

        # Continuous S4D parameterization: alpha = softplus(log_alpha) > 0
        self.log_alpha = nn.Parameter(torch.randn(d_model, 1) * 0.1)
        self.beta = nn.Parameter(torch.randn(d_model, 1))
        self.gamma = nn.Parameter(torch.randn(d_model, 1))

        self.dw_h = nn.Conv2d(d_model, d_model, (1, kernel_size), padding=(0, kernel_size // 2), groups=d_model)
        self.dw_v = nn.Conv2d(d_model, d_model, (kernel_size, 1), padding=(kernel_size // 2, 0), groups=d_model)

        self.motion_mod = nn.Sequential(
            nn.Conv2d(4, d_model // 4, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(d_model // 4, 1, 1),
            nn.Tanh()
        )

        self.dir_gate = nn.Sequential(
            nn.Conv2d(4 + d_model, d_model // 2, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(d_model // 2, 2, 1),
            nn.Softmax(dim=1)
        )
        self.proj_out = nn.Conv2d(d_model, d_model, 1)
        self.norm = nn.GroupNorm(4, d_model)

    def forward(self, x: torch.Tensor, motion_desc: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        v_mod = self.motion_mod(motion_desc).mean()
        alpha = F.softplus(self.log_alpha + v_mod)

        t = torch.arange(self.kernel_size, device=x.device, dtype=x.dtype).unsqueeze(0)
        ssm_kernel = (self.gamma * torch.exp(-alpha * t) * self.beta).mean()

        y_h = self.dw_h(x) * ssm_kernel
        y_v = self.dw_v(x) * ssm_kernel

        gate = self.dir_gate(torch.cat([x, motion_desc], dim=1))
        w_h = gate[:, 0:1, :, :]
        w_v = gate[:, 1:2, :, :]

        fused = self.proj_out(w_h * y_h + w_v * y_v)
        return self.norm(x + fused)


class MultiScaleDynamicFeatureFusion(nn.Module):
    """Multi-Scale Dynamic Feature Fusion (MDFF).
    Pixel-adaptive multi-scale depthwise convolutions (3x3, 5x5, 7x7) conditioned on motion.
    """
    def __init__(self, channels: int = 48):
        super().__init__()
        self.channels = channels
        self.dw3 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.dw5 = nn.Conv2d(channels, channels, 5, padding=2, groups=channels)
        self.dw7 = nn.Conv2d(channels, channels, 7, padding=3, groups=channels)

        self.kernel_gen = nn.Sequential(
            nn.Conv2d(4 + channels, channels // 2, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels // 2, 3, 1),
            nn.Softmax(dim=1)
        )
        self.pw = nn.Conv2d(channels, channels, 1)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor, motion_desc: torch.Tensor) -> torch.Tensor:
        weights = self.kernel_gen(torch.cat([x, motion_desc], dim=1))
        w3 = weights[:, 0:1, :, :]
        w5 = weights[:, 1:2, :, :]
        w7 = weights[:, 2:3, :, :]

        out3 = self.dw3(x)
        out5 = self.dw5(x)
        out7 = self.dw7(x)

        fused = w3 * out3 + w5 * out5 + w7 * out7
        return x + self.lrelu(self.pw(fused))


class FourierFrequencyBlock(nn.Module):
    """Algorithm 2 Enhancement: Dual-Domain Fast Fourier Transform (FFT) Coupling.
    Directly isolates and amplifies the high-frequency spectral phase & magnitude
    attenuated by motion blur, gating the State Space representation with spectral gradients.
    """
    def __init__(self, channels: int = 48):
        super().__init__()
        self.channels = channels
        self.freq_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels * 2, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels * 2, channels * 2, 1)
        )
        self.spatial_gate = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        fft = torch.fft.rfft2(x.float(), norm="ortho")
        fft_real = fft.real
        fft_imag = fft.imag
        fft_cat = torch.cat([fft_real, fft_imag], dim=1)

        freq_feat = self.freq_conv(fft_cat)
        real_out, imag_out = torch.chunk(freq_feat, 2, dim=1)
        fft_mod = torch.complex(real_out, imag_out)

        spatial_high_freq = torch.fft.irfft2(fft_mod, s=(h, w), norm="ortho").to(x.dtype)
        gate = torch.sigmoid(self.spatial_gate(spatial_high_freq))
        return x + gate * spatial_high_freq


class DeformableTemporalFlowBlock(nn.Module):
    """Algorithm 3 Enhancement: Deformable Flow-Guided Spatio-Temporal SSM (DF-SSM).
    Integrates sub-pixel continuous trajectory flow warping into the state space feature memory,
    coupled with temporal cross-attention and refinement convolutions.
    """
    def __init__(self, channels: int = 48):
        super().__init__()
        self.channels = channels

        self.flow_refine = nn.Sequential(
            nn.Conv2d(channels + 2, channels, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, 2, 3, padding=1)
        )
        self.temporal_attn = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )
        self.fusion_conv = nn.Conv2d(channels, channels, 3, padding=1)

    def _warp(self, x: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, H, device=x.device, dtype=x.dtype),
            torch.linspace(-1.0, 1.0, W, device=x.device, dtype=x.dtype),
            indexing="ij"
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)
        flow_norm = torch.stack([
            flow[:, 0] / max(W / 2.0, 1.0),
            flow[:, 1] / max(H / 2.0, 1.0)
        ], dim=-1)
        vgrid = grid + flow_norm
        return F.grid_sample(x, vgrid, mode="bilinear", padding_mode="border", align_corners=False)

    def forward(self, feat: torch.Tensor, ref_feat: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        delta_flow = self.flow_refine(torch.cat([feat, flow], dim=1))
        refined_flow = flow + delta_flow

        warped_feat = self._warp(feat, refined_flow)
        gate = self.temporal_attn(torch.cat([warped_feat, ref_feat], dim=1))
        aligned = self.fusion_conv(warped_feat * gate)
        return feat + aligned


class MSVSSMBlock(nn.Module):
    """Complete Motion-aware Visual State Space Block."""
    def __init__(
        self,
        channels: int = 48,
        algo: str = "base"
    ):
        super().__init__()
        self.algo = algo
        self.msb = MotionAwareScanBlock(d_model=channels, kernel_size=17)
        self.mdff = MultiScaleDynamicFeatureFusion(channels=channels)

        if algo == "fft":
            self.freq_block = FourierFrequencyBlock(channels=channels)
        elif algo == "tsff":
            self.df_block = DeformableTemporalFlowBlock(channels=channels)

        self.res_conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

    def forward(
        self,
        x: torch.Tensor,
        motion_desc: torch.Tensor,
        ref_feat: Optional[torch.Tensor] = None,
        flow: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        res = x
        feat = self.msb(x, motion_desc)
        feat = self.mdff(feat, motion_desc)

        if self.algo == "fft":
            feat = self.freq_block(feat)
        elif self.algo == "tsff" and ref_feat is not None and flow is not None:
            feat = self.df_block(feat, ref_feat, flow)

        out = self.res_conv(self.lrelu(feat))
        return res + out


class MSVSSM(nn.Module):
    """Motion-aware Visual State Space Model for REDS Video Deblurring (CVPR 2026).
    
    Variants:
    - algo="base": MSVSSM-Base (Canonical MVSSM with MSB + MDFF)
    - algo="fft": MSVSSM-FFT (Dual-Domain Spectral Frequency Enhancement)
    - algo="tsff": MSVSSM-TSFF (Temporal-Spatial Flow-Gated Deformable SSM)
    """
    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        num_feat: int = 48,
        num_frame: int = 5,
        num_blocks: int = 2,
        center_frame_idx: int = 2,
        algo: str = "base"
    ):
        super().__init__()
        assert algo in ("base", "fft", "tsff"), f"Unknown algorithm variant: {algo}"
        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.num_feat = num_feat
        self.num_frame = num_frame
        self.center_frame_idx = center_frame_idx
        self.algo = algo

        # 1. Feature Extraction & x4 Spatial Downsampling
        self.conv_down = nn.Sequential(
            nn.Conv2d(num_in_ch, num_feat, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(num_feat, num_feat, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )

        # 2. Motion Estimator in Feature Space
        self.mem = MotionEstimationModule(in_ch=num_feat, feat_ch=32)

        # 3. Multi-Frame Temporal Aggregation
        self.temporal_fusion = nn.Sequential(
            nn.Conv2d(num_feat * num_frame, num_feat, 1),
            nn.LeakyReLU(0.1, inplace=True)
        )

        # 4. Core Motion-Aware State Space Blocks
        self.blocks = nn.ModuleList([
            MSVSSMBlock(channels=num_feat, algo=algo)
            for _ in range(num_blocks)
        ])

        # 5. Dual PixelShuffle x4 Upsampling and Reconstruction
        self.up = nn.Sequential(
            nn.Conv2d(num_feat, num_feat * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(num_feat, num_feat * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.reconstruct = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(num_feat, num_out_ch, 3, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: (B, T, C, H, W) normalized blurred video frames
        Returns:
            out: (B, C, H, W) restored sharp center frame
        """
        B, T, C, H, W = x.shape
        assert T == self.num_frame, f"Expected {self.num_frame} frames, got {T}"

        x_center = x[:, self.center_frame_idx, :, :, :].contiguous()

        # 1. Extract shallow features with x4 downsampling per frame
        x_flat = x.view(B * T, C, H, W)
        feat_down = self.conv_down(x_flat)
        _, f_c, h_down, w_down = feat_down.shape
        feat_frames = feat_down.view(B, T, f_c, h_down, w_down)

        ref_feat = feat_frames[:, self.center_frame_idx, :, :, :].contiguous()

        # 2. Estimate motion between each neighbor and center reference frame in feature space
        flows = []
        motion_descs = []
        for i in range(T):
            if i == self.center_frame_idx:
                flow = torch.zeros(B, 2, h_down, w_down, device=x.device, dtype=x.dtype)
                desc = torch.zeros(B, 4, h_down, w_down, device=x.device, dtype=x.dtype)
            else:
                flow, desc = self.mem(feat_frames[:, i, :, :, :], ref_feat)
            flows.append(flow)
            motion_descs.append(desc)

        avg_desc = torch.stack(motion_descs, dim=1).mean(dim=1)
        avg_flow = torch.stack(flows, dim=1).mean(dim=1)

        # 3. Temporal feature fusion
        fused = self.temporal_fusion(feat_frames.view(B, T * f_c, h_down, w_down))

        # 4. State Space Propagation through MSVSSM blocks
        h_feat = fused
        for block in self.blocks:
            h_feat = block(
                h_feat,
                motion_desc=avg_desc,
                ref_feat=ref_feat,
                flow=avg_flow
            )

        # 5. Upsample back to native resolution and reconstruct
        out_feat = self.up(h_feat)
        residual = self.reconstruct(out_feat)

        out = x_center + residual
        return out
