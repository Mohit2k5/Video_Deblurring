# Sequence-Level Dataset Splits (`splits/`)

This directory maintains the fixed, reproducible sequence-level split manifests for the REDS video deblurring project.

---

## Split Strategy & Partitions

The dataset uses a fixed sequence-level random split generated with **seed 42** and a **90:10 ratio** on the original 240 paired training sequences:

| Partition | Sequence Count | Frame Count | Source Dataset | Ground Truth | Purpose |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`train`** | 216 | 21,600 | `train_blur` / `train_sharp` | Yes | Model optimization |
| **`held_out_test`** | 24 | 2,400 | `train_blur` / `train_sharp` | Yes | Unseen benchmark evaluation |
| **`val`** | 30 | 3,000 | `val_blur` / `val_sharp` | Yes | Validation checkpoint selection |
| **`inference_test`** | 30 | 3,000 | `test_blur` | No | Blind inference only |
| **Total** | **300** | **30,000** | — | — | Zero partition leakage |

---

## Files

- **`sequence_split.csv`**:
  CSV table containing `sequence_id`, `source_dataset`, `partition`, `has_ground_truth`, `frame_count`, `split_strategy`, and `random_seed`.
- **`split_summary.json`**:
  Structured metadata file containing exact sequence assignments for programmatic validation.

*Note: These files are identical to `data/manifests/sequence_split.csv` and `data/manifests/split_summary.json` (verified with 100% hash parity).*
