# MSVSSM-FFT Video Deblurring Evaluation Report

## Benchmark Results on Held-Out REDS Test Sequences (2,400 Frames, 720x1280)

| Metric | Blurry Input Baseline | MSVSSM-FFT | Absolute Gain |
| :--- | :---: | :---: | :---: |
| **Pixel-Weighted Aggregate PSNR** | 22.8853 dB | **23.3182 dB** | **+0.4329 dB** |
| **Mean Per-Frame PSNR** | 23.9030 dB | **24.3617 dB** | **+0.4587 dB** |
| **Mean Per-Frame SSIM** | 0.7199 | **0.7224** | **+0.0025** |
| **Mean Charbonnier Loss** | 0.038678 | **0.037132** | **-0.001546** |
| **Positive Frame PSNR Ratio** | — | **2400 / 2400 (100.0%)** | Full partition improved |
| **Inference Speed (720x1280)** | — | **129.94 fps** | 1.12 GB Unified Memory |
