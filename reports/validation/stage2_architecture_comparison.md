# Stage 2 Architecture Comparison and Verification Report

## 1. Upstream Source Verification
- **Upstream Repository**: `https://github.com/xinntao/EDVR`
- **Pinned Commit**: `b02e63a0fc5854cad7b11a87f74601612e356eff`
- **Target Architecture File**: `basicsr/models/archs/edvr_arch.py`
- **Verification Evidence**: Fetched raw source directly from GitHub API; SHA256 checksum: `7f2db9c39751f26d007e9f54d5f6bcb50c9acf1c0d5a20b880f3d3447511ae81`.

---

## 2. Component-by-Component Comparison Matrix

| Component | Upstream Implementation (`b02e63a`) | Local Implementation (`src/models/edvr_arch.py`) | Discrepancy Found | Corrective Change / Status | Verification Evidence |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **PreDeblurModule** | `stride_conv_hr1` & `stride_conv_hr2` downsampling ($x4$), 3-level pyramid (`stride_conv_l2`, `stride_conv_l3`), and 5 residual blocks with skip connections | Same structure and layer configuration | Previous baseline lacked pyramid and downsampling | Fully restored to match official BasicSR PreDeblur | Output shape $H/4 \times W/4$, unit tests passing |
| **Feature Extraction** | `make_layer(ResidualBlockNoBN, num_extract_block)` | `make_layer(ResidualBlockNoBN, num_extract_block)` | None | Preserved | Exact parameter count match |
| **Pyramid Downsampling** | 2-stage convs: `conv_l2_1` (stride 2) $\rightarrow$ `conv_l2_2` (stride 1); `conv_l3_1` (stride 2) $\rightarrow$ `conv_l3_2` (stride 1) | Same 2-stage conv pairs | Previous baseline used single conv per level | Restored 2 convs per pyramid level | Layer shapes: $L_2$ ($H/8$), $L_3$ ($H/16$) verified |
| **PCD Alignment** | 3-level pyramid ($L_3 \rightarrow L_2 \rightarrow L_1$); Level 3 without `offset_conv3`; Level 2 & 1 with $2\times$ offset upsampling; Level 1 cascading deformable alignment | Same pyramid and cascading DCNv2 alignment | Previous baseline incorrectly invoked `offset_conv3` on Level 3 | Restored official pyramid routing | `test_pcd_alignment_tensor_shapes` passed |
| **DCNv2Pack** | `conv_offset` predicts $(o_1, o_2, \text{mask})$, zero-initialized weights/biases, modulated deform conv | Native `torchvision.ops.deform_conv2d` wrapper with zero initialization | None | Implemented and verified zero-offset identity | Zero-offset max diff vs $0.5 \times \text{Conv2d} = 0.00000000e+00$ |
| **TSAFusion (Temporal)** | Embedding ref vs neighbor dot-product, sigmoid correlation reweighting, `feat_fusion` 1x1 conv | Same embedding and correlation reweighting | None | Preserved | Temporal correlation unit tests passed |
| **TSAFusion (Spatial)** | `spatial_attn1` takes `aligned_feat` ($5 \times C$ channels), multi-scale `MaxPool2d` + `AvgPool2d` pyramid, `spatial_attn_add` residual | Same spatial pyramid taking `aligned_feat`, outputting `attn * 2 + attn_add` | Previous inquiry questioned if spatial attention should receive `feat` instead of `aligned_feat` | Confirmed upstream design: `spatial_attn1` has $5 \times C$ input channels, intentionally taking `aligned_feat` | Dimension assertion and numerical stability verified |
| **Reconstruction & Upsampling** | Residual blocks $\rightarrow$ 2-stage `PixelShuffle(2)` (`upconv1`, `upconv2`) $\rightarrow$ `conv_hr` $\rightarrow$ `conv_last` | Same 2-stage PixelShuffle reconstruction | Previous baseline lacked PixelShuffle upconvolution | Restored 2 PixelShuffle stages restoring native resolution | Forward pass output shape matches input $H \times W$ |
| **Base Skip Connection** | `out += x_center` (when `hr_in=True`) | `out = out + base` where `base = x_center` | None | Preserved | `test_center_frame_selection_and_residual` passed |

---

## 3. Explicit Model Variant Distinction
1. **Official EDVR-L Configuration**:
   - `num_feat`: 128
   - `num_extract_block`: 5
   - `num_reconstruct_block`: 40
   - Total Parameters: **23,602,019**
2. **Official EDVR-M / Standard Configuration**:
   - `num_feat`: 64
   - `num_extract_block`: 5
   - `num_reconstruct_block`: 10
   - Total Parameters: **4,042,851**
3. **Active Hardware-Adapted Configuration (`configs/edvr.yaml`)**:
   - Matches official EDVR-M architecture exactly: `num_feat: 64`, `num_extract_block: 5`, `num_reconstruct_block: 10`, `deformable_groups: 8`, `num_frame: 5`, `hr_in: true`, `with_predeblur: true`, `with_tsa: true`.
   - Total Parameters: **4,042,851**.

---

## 4. Stage 2 Gate Assessment
- **Status**: **PASSED**.
- All components have been verified against the pinned upstream source.
- Spatial attention input design is verified and mathematically sound.
- Deformable convolution equivalence to zero-offset modulated convolution is verified to machine precision.
- Proceeding to Stage 3 is safe.
