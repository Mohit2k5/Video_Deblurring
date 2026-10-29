# REDS Video Motion Deblurring Documentation (`docs/`)

This directory serves as the documentation portal for architectural specifications, validation gates, and evaluation reports for the REDS video motion deblurring project.

---

## Key Reports & Documentation

### 1. Benchmark & Evaluation
- **[Held-Out Benchmark Evaluation Report](../reports/evaluation/edvr_held_out_evaluation_report.md)**:
  Full empirical evaluation of the 1,000-step EDVR-M baseline on all 24 unseen REDS test sequences (2,400 frames at full $720 \times 1280$ resolution). Includes aggregate and per-frame PSNR/SSIM, per-sequence breakdowns, and visual analysis.

### 2. Readiness & Validation Gates
- **[Final EDVR Readiness Report](../reports/validation/final_edvr_readiness_report.md)**:
  Final readiness certification detailing resolved reliability issues, validation mode restoration, AMP gradient unscaling, and genuine post-resume optimizer steps.
- **[Stage 9 Continuation Report](../reports/validation/stage9_readiness_and_continuation_report.md)**:
  Audit report reconciling historical parameter discrepancies (resolving 4,042,851 params) and memory profiling.

### 3. Architecture & Data Pipeline
- **[EDVR Architecture & Hardware Validation Report](../reports/data_pipeline/edvr_architecture_and_validation_report.md)**:
  Component-by-component comparison with upstream BasicSR EDVR (Wang et al., CVPRW 2019), detailing PreDeblur pyramid, PCD alignment with DCNv2, TSA fusion, and VRAM scaling.
- **[Shared Data Pipeline Documentation](../reports/data_pipeline/README.md)**:
  Specifications for deterministic sequence splitting (seed 42), sliding window temporal indexing, and paired data augmentation.
