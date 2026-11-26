"""PGDN: Prior-Guided Deblurring Network (2025).
Motion-prior-guided dual-branch restoration for video motion deblurring.

Architectural Components:
1. Motion Prior Guidance Module (MPGM):
   - Explicitly estimates parameterized motion kernel properties:
     Length L(x, y), Angle θ(x, y), Curvature κ(x, y), and Blur Severity Mask M_blur(x, y).
2. Dual-Branch Restoration Backbone:
   - Branch A (Context Branch): Global contextual and temporal multi-frame feature aggregation.
   - Branch B (Detail Branch): Motion-prior-guided spatial high-frequency detail restoration.
3. Cross-Branch Interactive Fusion (CBIF):
   - Bidirectional feature interaction and cross-gated attention between Context and Detail branches.
4. Reconstruction Trunk:
   - Deep residual blocks with Channel Attention and center-frame identity skip connection.
"""
import math
from typing import Dict, Any, Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation Channel Attention Block."""
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        reduced_ch = max(8, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, reduced_ch, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_ch, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.fc(self.avg_pool(x))
        return x * attn


class ResidualBlock(nn.Module):
    """Residual Block with Channel Attention and GELU activation."""
    def __init__(self, dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=True)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=True)
        self.ca = ChannelAttention(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.act(self.conv1(x))
        res = self.conv2(res)
        res = self.ca(res)
        return x + res


class MotionPriorGuidanceModule(nn.Module):
    """
    Motion-Prior Guidance Module (MPGM).
    Estimates parameterized blur kernel priors:
    - Motion length L (magnitude)
    - Motion angle θ (direction in radians [-π, π])
    - Motion curvature κ (trajectory curvature)
    - Blur severity mask M_blur in [0, 1]
    - Motion-prior feature embedding for spatial modulation
    """
    def __init__(self, num_frames: int = 5, center_idx: int = 2, in_channels: int = 3, prior_dim: int = 32):
        super().__init__()
        self.num_frames = num_frames
        self.center_idx = center_idx
        self.prior_dim = prior_dim

        # Temporal difference channels: (num_frames - 1) * in_channels + center_frame
        diff_in_channels = num_frames * in_channels

        self.net = nn.Sequential(
            nn.Conv2d(diff_in_channels, prior_dim * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(prior_dim * 2, prior_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # Heads for parameterized motion kernel priors
        self.head_mask = nn.Sequential(
            nn.Conv2d(prior_dim, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.head_length = nn.Sequential(
            nn.Conv2d(prior_dim, 1, kernel_size=3, padding=1),
            nn.Softplus()
        )
        self.head_angle = nn.Sequential(
            nn.Conv2d(prior_dim, 1, kernel_size=3, padding=1),
            nn.Tanh()
        )
        self.head_curvature = nn.Sequential(
            nn.Conv2d(prior_dim, 1, kernel_size=3, padding=1),
            nn.Tanh()
        )
        self.prior_proj = nn.Conv2d(prior_dim + 4, prior_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x: (B, T, C, H, W)
        b, t, c, h, w = x.shape
        center_frame = x[:, self.center_idx]  # (B, C, H, W)

        # Concatenate center frame with temporal differences
        diffs = [center_frame]
        for i in range(t):
            if i != self.center_idx:
                diffs.append(x[:, i] - center_frame)
        cat_diffs = torch.cat(diffs, dim=1)  # (B, T*C, H, W)

        feat = self.net(cat_diffs)

        mask = self.head_mask(feat)
        length = self.head_length(feat)
        angle = self.head_angle(feat) * math.pi
        curvature = self.head_curvature(feat)

        # Composite guidance embedding
        guidance = torch.cat([feat, mask, length, angle, curvature], dim=1)
        guidance = self.prior_proj(guidance)

        return {
            "mask": mask,
            "length": length,
            "angle": angle,
            "curvature": curvature,
            "guidance": guidance
        }


class ContextualTemporalBranch(nn.Module):
    """
    Branch 1: Global Contextual / Temporal Multi-Frame Aggregation.
    Captures non-local scene coherence and inter-frame temporal dependencies.
    """
    def __init__(self, in_channels: int = 3, num_frames: int = 5, embed_dim: int = 64):
        super().__init__()
        self.num_frames = num_frames
        self.in_proj = nn.Sequential(
            nn.Conv2d(in_channels * num_frames, embed_dim, kernel_size=3, padding=1),
            nn.GELU()
        )
        # Multi-scale dilated convolutions for global receptive field
        self.dconv1 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, dilation=1)
        self.dconv2 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=2, dilation=2)
        self.dconv3 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=4, dilation=4)
        self.fuse = nn.Sequential(
            nn.Conv2d(embed_dim * 3, embed_dim, kernel_size=1),
            nn.GELU(),
            ChannelAttention(embed_dim)
        )
        self.block = ResidualBlock(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W)
        b, t, c, h, w = x.shape
        flat_x = x.view(b, t * c, h, w)
        feat = self.in_proj(flat_x)
        d1 = F.gelu(self.dconv1(feat))
        d2 = F.gelu(self.dconv2(feat))
        d3 = F.gelu(self.dconv3(feat))
        fused = self.fuse(torch.cat([d1, d2, d3], dim=1))
        return self.block(fused)


class MotionGuidedDetailBranch(nn.Module):
    """
    Branch 2: Motion-Prior-Guided Spatial Detail Branch.
    Focuses on high-blur regions guided by the estimated blur mask and motion parameters.
    Uses motion-modulated dynamic depthwise filtering along estimated trajectory.
    """
    def __init__(self, in_channels: int = 3, prior_dim: int = 32, embed_dim: int = 64):
        super().__init__()
        self.center_proj = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1),
            nn.GELU()
        )
        self.motion_mod = nn.Sequential(
            nn.Conv2d(prior_dim, embed_dim * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1)
        )
        self.detail_conv1 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim)
        self.detail_pw1 = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.detail_conv2 = nn.Conv2d(embed_dim, embed_dim, kernel_size=5, padding=2, groups=embed_dim)
        self.detail_pw2 = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.ca = ChannelAttention(embed_dim)
        self.block = ResidualBlock(embed_dim)

    def forward(self, center_frame: torch.Tensor, motion_info: Dict[str, torch.Tensor]) -> torch.Tensor:
        feat = self.center_proj(center_frame)
        mask = motion_info["mask"]          # (B, 1, H, W)
        guidance = motion_info["guidance"]  # (B, prior_dim, H, W)

        # Modulate features with learned motion prior
        mod = self.motion_mod(guidance)
        feat = feat + mod * (1.0 + mask)

        # Depthwise spatial detail extraction
        d1 = self.detail_pw1(F.gelu(self.detail_conv1(feat)))
        d2 = self.detail_pw2(F.gelu(self.detail_conv2(d1)))
        d = self.ca(d2)

        return self.block(feat + d * mask)


class CrossBranchInteractiveFusion(nn.Module):
    """
    Cross-Branch Interactive Fusion (CBIF).
    Bidirectional feature interaction with cross-gated attention.
    """
    def __init__(self, embed_dim: int = 64):
        super().__init__()
        self.gate_ctx = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
            nn.Sigmoid()
        )
        self.gate_det = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
            nn.Sigmoid()
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=3, padding=1),
            nn.GELU(),
            ChannelAttention(embed_dim),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)
        )

    def forward(self, f_ctx: torch.Tensor, f_det: torch.Tensor) -> torch.Tensor:
        g_ctx = self.gate_ctx(f_ctx)
        g_det = self.gate_det(f_det)

        # Cross-gated interaction
        ctx_guided = f_ctx * g_det
        det_guided = f_det * g_ctx

        cat = torch.cat([ctx_guided, det_guided], dim=1)
        return self.fusion(cat)


class PGDN(nn.Module):
    """
    Prior-Guided Deblurring Network (PGDN, 2025).
    Full End-to-End Motion-Prior-Guided Dual-Branch Video Deblurring Architecture.
    """
    def __init__(
        self,
        num_frames: int = 5,
        center_frame_idx: int = 2,
        in_channels: int = 3,
        embed_dim: int = 64,
        num_blocks: int = 4,
        prior_dim: int = 32
    ):
        super().__init__()
        self.num_frames = num_frames
        self.center_frame_idx = center_frame_idx
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        # 1. Motion Prior Guidance Module (MPGM)
        self.mpgm = MotionPriorGuidanceModule(
            num_frames=num_frames,
            center_idx=center_frame_idx,
            in_channels=in_channels,
            prior_dim=prior_dim
        )

        # 2. Dual-Branch Restoration Backbone
        self.context_branch = ContextualTemporalBranch(
            in_channels=in_channels,
            num_frames=num_frames,
            embed_dim=embed_dim
        )
        self.detail_branch = MotionGuidedDetailBranch(
            in_channels=in_channels,
            prior_dim=prior_dim,
            embed_dim=embed_dim
        )

        # 3. Cross-Branch Interactive Fusion (CBIF)
        self.fusion = CrossBranchInteractiveFusion(embed_dim=embed_dim)

        # 4. Reconstruction Trunk
        self.blocks = nn.ModuleList([
            ResidualBlock(embed_dim) for _ in range(num_blocks)
        ])

        # Final projection to RGB residual
        self.out_conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(embed_dim, in_channels, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input tensor of shape (B, T, C, H, W) or (B, T*C, H, W)
        Returns:
            Restored sharp center frame (B, C, H, W) clamped to [0.0, 1.0]
        """
        if x.dim() == 4:
            # (B, T*C, H, W) -> reshape to (B, T, C, H, W)
            b, tc, h, w = x.shape
            c = self.in_channels
            t = tc // c
            x = x.view(b, t, c, h, w)

        b, t, c, h, w = x.shape
        center_frame = x[:, self.center_frame_idx]  # (B, C, H, W)

        # 1. Extract Motion Prior Guidance
        motion_info = self.mpgm(x)

        # 2. Dual-Branch Feature Extraction
        f_ctx = self.context_branch(x)
        f_det = self.detail_branch(center_frame, motion_info)

        # 3. Cross-Branch Interactive Fusion
        f_fused = self.fusion(f_ctx, f_det)

        # 4. Deep Residual Reconstruction
        res = f_fused
        for block in self.blocks:
            res = block(res)

        residual_img = self.out_conv(res)

        # Center frame skip connection & numerical clamping
        output = center_frame + residual_img
        return output.clamp(0.0, 1.0)
