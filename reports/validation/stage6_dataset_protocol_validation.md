# Stage 6 Dataset Protocol and Partition Validation Report

## 1. Executive Summary & Verification
The established REDS sequence-level partition has been audited and confirmed intact:
- **Canonical Manifest**: `data/manifests/sequence_split.csv`
- **Mirrored Manifest**: `splits/sequence_split.csv`
- **Hash Parity**: Exact match (`SHA256: 0e5df67dfc7576c9ad2c286d9b0521eec4653f2bc89faee32ad5106e2060f2bb`).
- **Total Registered Sequences**: **300**.

---

## 2. Partition Breakdown and Frame Allocation
| Partition Name | Source Directory | Ground Truth Available | Sequence Count | Frame Count | Role in ML Lifecycle |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `train` | `data/raw/train_blur` | Yes (`train_sharp`) | **216** | 21,600 | Model parameter optimization |
| `val` | `data/raw/val_blur` | Yes (`val_sharp`) | **30** | 3,000 | Validation loss & PSNR checkpoint selection |
| `held_out_test` | `data/raw/train_blur` | Yes (`train_sharp`) | **24** | 2,400 | Final benchmark evaluation (strictly isolated) |
| `inference_test`| `data/raw/test_blur` | No (unpaired) | **30** | 3,000 | Competition / inference submission |
| **Total** | | | **300** | **30,000** | |

---

## 3. Strict Partition Isolation & Leakage Checks
- **Train vs Held-Out Test Sequence Overlap**: **0** ($\emptyset$).
- **Composite Key Identification**: Every sequence is tracked as `(source_dataset, sequence_id)` preventing cross-directory naming collision.
- **Data Integrity**: Raw dataset files in `data/raw/` have not been moved, renamed, or modified.
- **Boundary Policy**: Fixed replicate boundary policy maintains consistent $T=5$ temporal windows without pairing across video boundaries.
- **Paired Crop Alignment**: Synchronized random crops and geometric flips ensure pixel coordinate $(x, y)$ in blur maps to $(x, y)$ in sharp ground truth.

---

## 4. Stage 6 Gate Assessment
- **Status**: **PASSED**.
- All partition boundaries, pairing guarantees, and isolation rules are validated and preserved.
- Proceeding to Stage 7 is safe.
