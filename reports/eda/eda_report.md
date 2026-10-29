# EDA Report: REDS Video Deblurring

## 1. Project Objective
To explore the REDS dataset for video motion deblurring.

## 2. Dataset and Sequence Counts
    dataset  total_sequences  min_frames  max_frames  mean_frames  median_frames
  test_blur               30         100         100        100.0          100.0
 train_blur              240         100         100        100.0          100.0
train_sharp              240         100         100        100.0          100.0
   val_blur               30         100         100        100.0          100.0
  val_sharp               30         100         100        100.0          100.0

## 3. Image Validation Issues
No validation issues found in sampled data.

## 4. Figures
![Dataset Distribution](../../figures/dataset_distribution.png)
![Brightness](../../figures/brightness_distribution.png)
## 5. Next Steps
Implement deblurring model preprocessing.
