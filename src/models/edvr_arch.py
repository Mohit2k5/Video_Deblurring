"""EDVR network architecture for REDS video deblurring.
Faithfully reproduced from official EDVR (Wang et al., CVPRW 2019) / BasicSR implementation.
Upstream commit reference: xinntao/EDVR commit b02e63a0fc5854cad7b11a87f74601612e356eff (BasicSR v1.2.0).

Uses native torchvision.ops.deform_conv2d for cross-platform zero-dependency modulated DCNv2 execution.
"""
import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.ops as ops


def default_init_weights(module_list, scale=1.0, bias_fill=0.0, **kwargs):
    """Initialize network weights with Kaiming normal distribution scaled by `scale`."""
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


def make_layer(basic_block, num_basic_block, **kwargs):
    """Stack identical basic blocks into an nn.Sequential container."""
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwargs))
    return nn.Sequential(*layers)


class ResidualBlockNoBN(nn.Module):
    """Residual block without Batch Normalization.
    
    Structure:
        ---Conv-ReLU-Conv-+-
         |________________|
    """
    def __init__(self, num_feat=64, res_scale=1.0, pytorch_init=False):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


class DCNv2Pack(nn.Module):
    """Modulated Deformable Convolution v2 operator using torchvision.ops.deform_conv2d.
    
    Faithfully mirrors official BasicSR DCNv2Pack behavior:
    Predicts offsets and modulation masks from feature representations,
    concatenates (o1, o2), applies sigmoid to masks, and applies modulated deform conv.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        deformable_groups: int = 8
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.deformable_groups = deformable_groups

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, *self.kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels))

        # Predict horizontal offset, vertical offset, and modulation mask (3 components per kernel weight per group)
        out_offset_dim = self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1]
        self.conv_offset = nn.Conv2d(
            self.in_channels,
            out_offset_dim,
            self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=True
        )

        self.init_weights()

    def init_weights(self):
        n = self.in_channels * self.kernel_size[0] * self.kernel_size[1]
        stdv = 1.0 / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.zero_()
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()

    def forward(self, x: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        out = self.conv_offset(feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        return ops.deform_conv2d(
            input=x,
            offset=offset,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            mask=mask
        )


class PCDAlignment(nn.Module):
    """Pyramid, Cascading and Deformable (PCD) alignment module.
    
    Faithfully matches official BasicSR PCDAlignment:
    - 3-level pyramid (L3: 1/4 size, L2: 1/2 size, L1: original feature size)
    - Level 3: offset_conv1 -> offset_conv2
    - Levels 2 and 1: offset_conv1 -> offset_conv2 (concatenating with 2x upsampled offset) -> offset_conv3
    - Feature warping via DCNv2Pack, feature fusion conv (i < 3), and LeakyReLU activations
    - Cascading alignment at Level 1 with cas_offset_conv1, cas_offset_conv2, and cas_dcnpack
    """
    def __init__(self, num_feat=64, deformable_groups=8):
        super().__init__()
        self.offset_conv1 = nn.ModuleDict()
        self.offset_conv2 = nn.ModuleDict()
        self.offset_conv3 = nn.ModuleDict()
        self.dcn_pack = nn.ModuleDict()
        self.feat_conv = nn.ModuleDict()

        # Pyramids: L3 -> L2 -> L1
        for i in range(3, 0, -1):
            level = f'l{i}'
            self.offset_conv1[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
            if i == 3:
                self.offset_conv2[level] = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            else:
                self.offset_conv2[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
                self.offset_conv3[level] = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.dcn_pack[level] = DCNv2Pack(
                num_feat, num_feat, 3, padding=1, deformable_groups=deformable_groups
            )
            if i < 3:
                self.feat_conv[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)

        # Cascading DCN
        self.cas_offset_conv1 = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
        self.cas_offset_conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.cas_dcnpack = DCNv2Pack(
            num_feat, num_feat, 3, padding=1, deformable_groups=deformable_groups
        )

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, nbr_feat_l: List[torch.Tensor], ref_feat_l: List[torch.Tensor]) -> torch.Tensor:
        """Align neighboring frame features to reference frame features.
        
        Args:
            nbr_feat_l: List of neighboring features [L1, L2, L3]
            ref_feat_l: List of reference features [L1, L2, L3]
        """
        upsampled_offset, upsampled_feat = None, None
        for i in range(3, 0, -1):
            level = f'l{i}'
            offset = torch.cat([nbr_feat_l[i - 1], ref_feat_l[i - 1]], dim=1)
            offset = self.lrelu(self.offset_conv1[level](offset))
            if i == 3:
                offset = self.lrelu(self.offset_conv2[level](offset))
            else:
                offset = self.lrelu(self.offset_conv2[level](torch.cat([offset, upsampled_offset], dim=1)))
                offset = self.lrelu(self.offset_conv3[level](offset))

            feat = self.dcn_pack[level](nbr_feat_l[i - 1], offset)
            if i < 3:
                feat = self.feat_conv[level](torch.cat([feat, upsampled_feat], dim=1))
            if i > 1:
                feat = self.lrelu(feat)

            if i > 1:
                upsampled_offset = self.upsample(offset) * 2.0
                upsampled_feat = self.upsample(feat)

        # Cascading alignment at Level 1
        offset = torch.cat([feat, ref_feat_l[0]], dim=1)
        offset = self.lrelu(self.cas_offset_conv2(self.lrelu(self.cas_offset_conv1(offset))))
        feat = self.lrelu(self.cas_dcnpack(feat, offset))
        return feat


class TSAFusion(nn.Module):
    """Temporal and Spatial Attention (TSA) fusion module.
    
    Faithfully matches official BasicSR TSAFusion:
    - Temporal attention: Dot-product correlation between reference frame and neighboring frames,
      passed through sigmoid, channel-expanded, and multiplied onto aligned features.
    - Fusion 1x1 conv across all T frames.
    - Spatial attention: Multi-scale pooling (max_pool & avg_pool), pyramid levels,
      spatial_attn_add residual branch, sigmoid weighting (* 2), and additive branch.
    """
    def __init__(self, num_feat=64, num_frame=5, center_frame_idx=2):
        super().__init__()
        self.center_frame_idx = center_frame_idx
        self.num_frame = num_frame

        # Temporal attention
        self.temporal_attn1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.temporal_attn2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.feat_fusion = nn.Conv2d(num_frame * num_feat, num_feat, 1, 1)

        # Spatial attention
        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)
        self.spatial_attn1 = nn.Conv2d(num_frame * num_feat, num_feat, 1)
        self.spatial_attn2 = nn.Conv2d(num_feat * 2, num_feat, 1)
        self.spatial_attn3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.spatial_attn4 = nn.Conv2d(num_feat, num_feat, 1)
        self.spatial_attn5 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.spatial_attn_l1 = nn.Conv2d(num_feat, num_feat, 1)
        self.spatial_attn_l2 = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
        self.spatial_attn_l3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.spatial_attn_add1 = nn.Conv2d(num_feat, num_feat, 1)
        self.spatial_attn_add2 = nn.Conv2d(num_feat, num_feat, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, aligned_feat: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = aligned_feat.size()

        # 1. Temporal attention
        embedding_ref = self.temporal_attn1(
            aligned_feat[:, self.center_frame_idx, :, :, :].clone()
        )
        embedding = self.temporal_attn2(aligned_feat.view(-1, c, h, w))
        embedding = embedding.view(b, t, -1, h, w)

        corr_l = []
        for i in range(t):
            emb_neighbor = embedding[:, i, :, :, :]
            corr = torch.sum(emb_neighbor * embedding_ref, dim=1)  # (b, h, w)
            corr_l.append(corr.unsqueeze(1))  # (b, 1, h, w)
        corr_prob = torch.sigmoid(torch.cat(corr_l, dim=1))  # (b, t, h, w)
        corr_prob = corr_prob.unsqueeze(2).expand(b, t, c, h, w)
        corr_prob = corr_prob.contiguous().view(b, -1, h, w)  # (b, t*c, h, w)
        aligned_feat = aligned_feat.view(b, -1, h, w) * corr_prob

        # Fusion conv
        feat = self.lrelu(self.feat_fusion(aligned_feat))

        # 2. Spatial attention
        attn = self.lrelu(self.spatial_attn1(aligned_feat))
        attn_max = self.max_pool(attn)
        attn_avg = self.avg_pool(attn)
        attn = self.lrelu(self.spatial_attn2(torch.cat([attn_max, attn_avg], dim=1)))

        # Multi-scale pyramid processing
        attn_level = self.lrelu(self.spatial_attn_l1(attn))
        attn_max = self.max_pool(attn_level)
        attn_avg = self.avg_pool(attn_level)
        attn_level = self.lrelu(self.spatial_attn_l2(torch.cat([attn_max, attn_avg], dim=1)))
        attn_level = self.lrelu(self.spatial_attn_l3(attn_level))
        attn_level = self.upsample(attn_level)

        attn = self.lrelu(self.spatial_attn3(attn)) + attn_level
        attn = self.lrelu(self.spatial_attn4(attn))
        attn = self.upsample(attn)
        attn = self.spatial_attn5(attn)
        attn_add = self.spatial_attn_add2(self.lrelu(self.spatial_attn_add1(attn)))
        attn = torch.sigmoid(attn)

        # Scale by 2.0 so initialization yields ~1.0 multiplier, plus residual addition
        feat = feat * attn * 2.0 + attn_add
        return feat


class PredeblurModule(nn.Module):
    """Pre-deblur sub-network for restoring features before multi-frame alignment.
    
    Faithfully matches official BasicSR PredeblurModule:
    - If hr_in=True: Downsamples x4 via two 3x3 stride-2 convolutions
    - 3-level pyramid with stride-2 downsampling
    - Residual processing at L3, L2 (resblock_l2_1, resblock_l2_2), and L1 (5 resblocks with skip add)
    """
    def __init__(self, num_in_ch=3, num_feat=64, hr_in=True):
        super().__init__()
        self.hr_in = hr_in

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        if self.hr_in:
            # Downsample x4 by two stride-2 convolutions
            self.stride_conv_hr1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
            self.stride_conv_hr2 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)

        # Feature pyramid downsampling
        self.stride_conv_l2 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.stride_conv_l3 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)

        self.resblock_l3 = ResidualBlockNoBN(num_feat=num_feat)
        self.resblock_l2_1 = ResidualBlockNoBN(num_feat=num_feat)
        self.resblock_l2_2 = ResidualBlockNoBN(num_feat=num_feat)
        self.resblock_l1 = nn.ModuleList([ResidualBlockNoBN(num_feat=num_feat) for _ in range(5)])

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat_l1 = self.lrelu(self.conv_first(x))
        if self.hr_in:
            feat_l1 = self.lrelu(self.stride_conv_hr1(feat_l1))
            feat_l1 = self.lrelu(self.stride_conv_hr2(feat_l1))

        # Generate feature pyramid
        feat_l2 = self.lrelu(self.stride_conv_l2(feat_l1))
        feat_l3 = self.lrelu(self.stride_conv_l3(feat_l2))

        feat_l3 = self.upsample(self.resblock_l3(feat_l3))
        feat_l2 = self.resblock_l2_1(feat_l2) + feat_l3
        feat_l2 = self.upsample(self.resblock_l2_2(feat_l2))

        for i in range(2):
            feat_l1 = self.resblock_l1[i](feat_l1)
        feat_l1 = feat_l1 + feat_l2
        for i in range(2, 5):
            feat_l1 = self.resblock_l1[i](feat_l1)
        return feat_l1


class EDVR(nn.Module):
    """
    EDVR Network for Video Restoration and Deblurring.
    Architecture faithfully conforms to official EDVR (Wang et al., CVPRW 2019) / BasicSR.
    
    Supports:
    - hr_in=True (REDS deblurring full-resolution mode):
      Downsamples x4 in PreDeblur, aligns and fuses at 1/4 resolution,
      reconstructs via 2 PixelShuffle(2) stages to restore native resolution.
    - with_tsa: Temporal and Spatial Attention fusion.
    - with_predeblur: Multi-scale Pre-deblur feature extraction.
    """
    def __init__(
        self,
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_frame=5,
        deformable_groups=8,
        num_extract_block=5,
        num_reconstruct_block=10,
        center_frame_idx=2,
        hr_in=True,
        with_predeblur=True,
        with_tsa=True
    ):
        super().__init__()
        if center_frame_idx is None:
            self.center_frame_idx = num_frame // 2
        else:
            self.center_frame_idx = center_frame_idx

        self.num_frame = num_frame
        self.num_feat = num_feat
        self.num_extract_block = num_extract_block
        self.num_reconstruct_block = num_reconstruct_block
        self.hr_in = hr_in
        self.with_predeblur = with_predeblur
        self.with_tsa = with_tsa

        # 1. Feature extraction
        if self.with_predeblur:
            self.predeblur = PredeblurModule(num_in_ch=num_in_ch, num_feat=num_feat, hr_in=self.hr_in)
            self.conv_1x1 = nn.Conv2d(num_feat, num_feat, 1, 1)
        else:
            self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)

        # Feature extractor residual blocks
        self.feature_extraction = make_layer(ResidualBlockNoBN, num_extract_block, num_feat=num_feat)

        # Multi-scale downsampling for PCD pyramid (stride 2 then stride 1)
        self.conv_l2_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l2_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_l3_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l3_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # 2. PCD alignment module
        self.pcd_align = PCDAlignment(num_feat=num_feat, deformable_groups=deformable_groups)

        # 3. Fusion module (TSA or 1x1 conv)
        if self.with_tsa:
            self.fusion = TSAFusion(num_feat=num_feat, num_frame=num_frame, center_frame_idx=self.center_frame_idx)
        else:
            self.fusion = nn.Conv2d(num_frame * num_feat, num_feat, 1, 1)

        # 4. Reconstruction module
        self.reconstruction = make_layer(ResidualBlockNoBN, num_reconstruct_block, num_feat=num_feat)

        # 5. Upsampling and output reconstruction
        self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
        self.upconv2 = nn.Conv2d(num_feat, 64 * 4, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, num_out_ch, 3, 1, 1)

        # Activation
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.size()
        assert t == self.num_frame, f"Expected {self.num_frame} frames, got {t}"
        if self.hr_in:
            assert h % 16 == 0 and w % 16 == 0, f"Height and width must be multiple of 16 for hr_in mode, got ({h}, {w})"
        else:
            assert h % 4 == 0 and w % 4 == 0, f"Height and width must be multiple of 4, got ({h}, {w})"

        x_center = x[:, self.center_frame_idx, :, :, :].contiguous()

        # 1. Feature extraction for each frame
        if self.with_predeblur:
            feat_l1 = self.conv_1x1(self.predeblur(x.view(-1, c, h, w)))
            if self.hr_in:
                h, w = h // 4, w // 4
        else:
            feat_l1 = self.lrelu(self.conv_first(x.view(-1, c, h, w)))

        feat_l1 = self.feature_extraction(feat_l1)
        # L2
        feat_l2 = self.lrelu(self.conv_l2_1(feat_l1))
        feat_l2 = self.lrelu(self.conv_l2_2(feat_l2))
        # L3
        feat_l3 = self.lrelu(self.conv_l3_1(feat_l2))
        feat_l3 = self.lrelu(self.conv_l3_2(feat_l3))

        feat_l1 = feat_l1.view(b, t, -1, h, w)
        feat_l2 = feat_l2.view(b, t, -1, h // 2, w // 2)
        feat_l3 = feat_l3.view(b, t, -1, h // 4, w // 4)

        # 2. PCD alignment
        ref_feat_l = [
            feat_l1[:, self.center_frame_idx, :, :, :].clone(),
            feat_l2[:, self.center_frame_idx, :, :, :].clone(),
            feat_l3[:, self.center_frame_idx, :, :, :].clone()
        ]
        aligned_feat = []
        for i in range(t):
            nbr_feat_l = [
                feat_l1[:, i, :, :, :].clone(),
                feat_l2[:, i, :, :, :].clone(),
                feat_l3[:, i, :, :, :].clone()
            ]
            aligned_feat.append(self.pcd_align(nbr_feat_l, ref_feat_l))
        aligned_feat = torch.stack(aligned_feat, dim=1)  # (b, t, c, h, w)

        # 3. Fusion
        if not self.with_tsa:
            aligned_feat = aligned_feat.view(b, -1, h, w)
        feat = self.fusion(aligned_feat)

        # 4. Reconstruction
        out = self.reconstruction(feat)

        # 5. Upsampling (x4 back to native resolution)
        out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
        out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        out = self.lrelu(self.conv_hr(out))
        out = self.conv_last(out)

        # Center-frame residual skip
        if self.hr_in:
            base = x_center
        else:
            base = F.interpolate(x_center, scale_factor=4, mode='bilinear', align_corners=False)
        out = out + base

        return out
