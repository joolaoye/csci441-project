# baseline: two-layer CNN, no augmentation (RQ1 and RQ2 reference), validation results

Owner: Chidubem Okoye. One training run, seed 42. All numbers are **validation** results; the official test partition was not used.

## Setup

| Item | Value |
|---|---|
| Configuration | `baseline.yaml` (shared baseline, unchanged) |
| Architecture | 2 × (3×3 conv, BatchNorm, ReLU), pooling schedule `[1, 1]` → 12×12, GAP, dropout 0.2, dense 128, softmax 43 |
| Parameters | 52,715 trainable / 52,971 total |
| Training | 42 epochs, batch 64, Adam lr 0.001, no augmentation, no early stopping |
| Data | `configs/splits/gtsrb_v1.json` (SHA-256 `e32ea068…98d7`), 31,380 train / 7,829 validation images |
| Code | `git_commit` 7ea3ece, `git_dirty: false` |
| Hardware | CPU only (Apple Silicon; TensorFlow 2.21.0 exposed no GPU) |
| Training time | 2,666.5 s (44.4 min) |
| Selected checkpoint | Epoch 40, highest clean validation accuracy (0.715034). `best_val_loss` 1.0069 is the loss at that epoch; the lowest loss in the history is 0.9806 at epoch 42 |

## Results: same 7,829 validation images under five conditions

All five evaluations use one checkpoint, the committed `evaluation.yaml`, and identical image IDs (`sample_ids_sha256` `8d88abc9…`, the same set as depth3 and depth4).

| Condition | Accuracy | Drop from clean | Macro-F1 | Loss |
|---|---:|---:|---:|---:|
| clean | 71.50% | — | 0.576 | 1.003 |
| zoom (center 1.1×) | 69.01% | 2.49 pp | 0.549 | 1.060 |
| brightness (+0.1) | 64.80% | 6.71 pp | 0.504 | 1.263 |
| rotation (+10°) | 60.66% | 10.84 pp | 0.479 | 1.310 |
| lowres (24→48) | 53.19% | 18.32 pp | 0.390 | 1.625 |

## Observations

- **Learning curves.** The model had not converged at epoch 42. Validation accuracy first reached 50% at epoch 18, 60% at epoch 28, and 70% at epoch 40; validation loss reached its minimum at the final epoch. After epoch 30 the epoch-to-epoch swing in validation accuracy was at most 3.1 pp. The final train/validation accuracy gap was 4.6 pp (76.1% vs 71.5%).
- **Condition ranking.** Same order as depth3: zoom costs least, then brightness, rotation, and low resolution. Brightness costs this model considerably more than it costs depth3 (6.7 pp vs 3.8 pp).
- **Accuracy vs macro-F1.** Macro-F1 (0.576) is far below accuracy (0.715): 35 of 43 classes have F1 below 0.80 and 14 below 0.50, so the errors are concentrated in particular, mostly smaller, classes.
- **Weak classes (clean).** Four classes have F1 0.00: *Speed limit 20* (30 images), *Pedestrians* (60), *Bicycles crossing* (60), and *Go straight or left* (30). Next weakest: *End of no passing* (0.13), *Roundabout mandatory* (0.20), *Dangerous curve to the right* (0.29), *Dangerous curve to the left* (0.32), *Turn left ahead* (0.45).
- **Confusions.** 37% of all errors (820 of 2,231) are one speed-limit sign read as another. The largest are *Speed limit 30 → 50* (102 of 450), *100 → 120* (67 of 300), *60 → 70* (62 of 270), *70 → 50* (58 of 390), and *100 → 80* (57 of 300).
- **Weak classes (altered).** The four zero-F1 classes stay at 0.00 under every altered condition. Under rotation and low resolution, *Dangerous curve to the left* also falls to 0.00; under low resolution, *Slippery road* does too.

## RQ2: depth comparison (now complete)

From `scripts/report.py compare --type depth --runs shared/baseline shared/depth3 shared/depth4 --conditions clean rotation zoom brightness lowres`. The evaluations are compatible: same manifest, image IDs, and condition parameters.

| Depth | Params | Epoch | clean | rotation | zoom | brightness | lowres |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2 | 52,715 | 40 | 71.50 | 60.66 (−10.84) | 69.01 (−2.49) | 64.80 (−6.71) | 53.19 (−18.32) |
| 3 | 89,707 | 40 | 92.59 | 82.90 (−9.69) | 89.85 (−2.75) | 88.79 (−3.81) | 75.21 (−17.38) |
| 4 | 126,699 | 39 | 98.25 | 96.00 (−2.25) | 96.97 (−1.28) | 97.25 (−1.00) | 91.11 (−7.14) |

Accuracy in %, with the percentage-point drop from that model's own clean accuracy in parentheses. Clean macro-F1: 0.576, 0.888, 0.977. Each added layer raised clean accuracy and, from three to four layers, also shrank every drop.

## Possible explanations (not tested)

- The two-layer model is undertrained within the shared 42-epoch budget: its best epoch is near the end and its loss is still falling. Part of the gap to depth3 and depth4 is likely convergence speed, not only final capacity. This applies to RQ1 as well, since every augmentation run uses this architecture.
- With two pooling stages and global average pooling, each final feature summarizes the whole 12×12 map. Fine distinctions inside the sign, such as the digits on speed-limit signs, may be hard to keep at this depth, which fits the dominance of speed-limit confusions. Not verified.
- *Speed limit 20* and *Go straight or left* have one validation track each (30 images), so one misread physical sign decides their score.

## Limitations

- One seed: an observed result, with no standard deviation or significance test.
- Validation images come from 261 physical tracks of about 30 frames each, so errors are correlated.
- CPU training; CPU and GPU runs of the same seed can differ slightly.
- `shared/brightness_only` records a different split manifest (`575d19ad…`), so `report.py compare` refuses to place it next to this baseline until it is retrained on the committed manifest.
