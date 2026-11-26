# MSVSSM-BASE Video Deblurring Evaluation Report

## Benchmark Results on Held-Out REDS Test Sequences (2,400 Frames, 720x1280)

| Metric | Blurry Input Baseline | MSVSSM-BASE | Absolute Gain |
| :--- | :---: | :---: | :---: |
| **Pixel-Weighted Aggregate PSNR** | 22.8853 dB | **23.1657 dB** | **+0.2804 dB** |
| **Mean Per-Frame PSNR** | 23.9030 dB | **24.2088 dB** | **+0.3058 dB** |
| **Mean Per-Frame SSIM** | 0.7199 | **0.7210** | **+0.0011** |
| **Mean Charbonnier Loss** | 0.038678 | **0.037486** | **-0.001192** |
| **Positive Frame PSNR Ratio** | — | **2400 / 2400 (100.0%)** | Full partition improved |
| **Inference Speed (720x1280)** | — | **196.41 fps** | 1.12 GB Unified Memory |
