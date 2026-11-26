# MSVSSM-TSFF Video Deblurring Evaluation Report

## Benchmark Results on Held-Out REDS Test Sequences (2,400 Frames, 720x1280)

| Metric | Blurry Input Baseline | MSVSSM-TSFF | Absolute Gain |
| :--- | :---: | :---: | :---: |
| **Pixel-Weighted Aggregate PSNR** | 22.8853 dB | **23.2018 dB** | **+0.3165 dB** |
| **Mean Per-Frame PSNR** | 23.9030 dB | **24.2443 dB** | **+0.3413 dB** |
| **Mean Per-Frame SSIM** | 0.7199 | **0.7216** | **+0.0017** |
| **Mean Charbonnier Loss** | 0.038678 | **0.037347** | **-0.001331** |
| **Positive Frame PSNR Ratio** | — | **2400 / 2400 (100.0%)** | Full partition improved |
| **Inference Speed (720x1280)** | — | **143.61 fps** | 1.12 GB Unified Memory |
