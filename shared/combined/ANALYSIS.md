# combined: two-layer CNN with rotation, brightness, and zoom augmentation (RQ1), validation results

Owner: Chidubem Okoye. One training run, seed 42. All numbers are **validation** results; the official test partition was not used.

## Setup

| Item | Value |
|---|---|
| Configuration | `configs/experiments/combined.yaml`: `baseline.yaml` with `experiment: combined` and the augmentation block below |
| Augmentation (training only) | rotation ±10°, brightness ±0.1 (on pixels in [0, 1]), zoom scale drawn from [0.9, 1.1]; values from the experiment design |
| Architecture | Same as the baseline: 2 × (3×3 conv, BatchNorm, ReLU), pooling `[1, 1]` → 12×12, GAP, dropout 0.2, dense 128, softmax 43 |
| Parameters | 52,715 trainable / 52,971 total |
| Training | 42 epochs, batch 64, Adam lr 0.001, no early stopping |
| Data | `configs/splits/gtsrb_v1.json` (SHA-256 `e32ea068…98d7`), 31,380 train / 7,829 validation images |
| Code | `git_commit` ccb404e, `git_dirty: false` |
| Hardware | CPU only (Apple Silicon; TensorFlow 2.21.0 exposed no GPU) |
| Training time | 2,804.1 s (46.7 min) |
| Selected checkpoint | Epoch 42, the final epoch (clean validation accuracy 0.717333, loss 0.8984, also the lowest loss in the history) |

Before training, `scripts/run_baseline_combined.sh` confirmed that only `experiment` and `augmentation` differ from `baseline.yaml`.

## Results: same 7,829 validation images under five conditions

All five evaluations use one checkpoint, the committed `evaluation.yaml`, and identical image IDs (`sample_ids_sha256` `8d88abc9…`).

| Condition | Accuracy | Drop from clean | Macro-F1 | Loss |
|---|---:|---:|---:|---:|
| clean | 71.73% | — | 0.601 | 0.898 |
| zoom (center 1.1×) | 72.60% | −0.87 pp | 0.596 | 0.847 |
| brightness (+0.1) | 70.38% | 1.35 pp | 0.568 | 0.963 |
| rotation (+10°) | 69.87% | 1.86 pp | 0.581 | 0.924 |
| lowres (24→48) | 57.50% | 14.23 pp | 0.433 | 1.353 |

## RQ1: comparison with the baseline

From `scripts/report.py compare --type augmentation --runs shared/baseline shared/combined --conditions clean rotation zoom brightness lowres`. Same architecture, manifest, training protocol, image IDs, and condition parameters.

| Training | Epoch | clean | rotation | zoom | brightness | lowres |
|---|---:|---:|---:|---:|---:|---:|
| none (baseline) | 40 | 71.50 | 60.66 (−10.84) | 69.01 (−2.49) | 64.80 (−6.71) | 53.19 (−18.32) |
| combined | 42 | 71.73 | 69.87 (−1.86) | 72.60 (+0.87) | 70.38 (−1.35) | 57.50 (−14.23) |
| change | | +0.23 | **+9.21** | **+3.59** | **+5.58** | **+4.31** |

Accuracy in %, with the change from that model's own clean accuracy in parentheses. Clean macro-F1: 0.576 → 0.601.

- **Clean accuracy is unchanged** within what one seed can resolve (+0.23 pp).
- **Robustness improved under every altered condition.** The rotation drop fell from 10.8 to 1.9 pp and the brightness drop from 6.7 to 1.4 pp. These are the transforms the model saw during training, at the same magnitudes used in evaluation.
- **Low resolution also improved (+4.3 pp)** even though it was not part of training, although it remains the hardest condition (14.2 pp drop).
- **Zoom scored slightly above clean** (−0.87 pp drop). Treat this as within noise, not as a benefit of zooming.

## Observations

- **Learning curves.** Like the baseline, the model had not converged: its best epoch and lowest loss are both at epoch 42. Validation accuracy first reached 50% at epoch 20 (baseline: 18), 60% at epoch 29 (28), and 70% at epoch 40 (40), so augmentation slowed early learning only slightly. Validation accuracy was noisier between epochs than the baseline's (largest swing 14.1 pp at epoch 15; up to 5.1 pp after epoch 30, vs 3.1 pp for the baseline).
- **Regularization.** Final training accuracy was lower than the baseline's (74.3% vs 76.1%) while validation loss was lower (0.898 vs 1.003), and the train/validation gap shrank from 4.6 to 2.6 pp.
- **Weak classes (clean).** *End of no passing* has F1 0.00 and is never predicted: 52 of its 60 images are labelled *End of no passing by vehicles over 3.5 metric tons*. The other baseline zero-F1 classes improved but remain weak: *Pedestrians* (0.03), *Speed limit 20* (0.19), *Go straight or left* (0.20), *Bicycles crossing* (0.23). Also weak: *Keep left* (0.30), *Beware of ice/snow* (0.32), *Slippery road* (0.37).
- **Confusions.** Speed-limit-to-speed-limit errors are still 36% of all errors (794 of 2,213). The largest single confusion is *Speed limit 80 → 120* (138 of 360), followed by *30 → 50* (96 of 450), *120 → 100* (79 of 270), and *100 → 120* (73 of 300).
- **Weak classes (altered).** *End of no passing* and *Pedestrians* stay near 0.00 under rotation, zoom, and brightness. Under low resolution, *Speed limit 20*, *Dangerous curve to the right*, *Pedestrians*, *Bicycles crossing*, and *Roundabout mandatory* all have F1 0.00.

## Possible explanations (not tested)

- The training transforms match the evaluation transforms in type and magnitude, so most of the robustness gain is the expected effect of training on the condition being tested. The low-resolution gain suggests some broader invariance, but one seed cannot separate that from noise.
- Augmentation makes each training batch harder, which fits the lower training accuracy, the noisier validation curve, and the best epoch landing on the last one. With a longer budget, combined might pull ahead on clean accuracy too; the 42-epoch budget cannot show that.
- Both "End of no passing" signs are white circles with a diagonal band and a small inner pictogram. The model's loss of that class may reflect how hard such fine detail is to keep with two pooled layers, not the augmentation itself.

## Limitations

- One seed per configuration: an observed comparison, with no standard deviation or significance test.
- Validation images come from 261 physical tracks of about 30 frames each, so errors are correlated.
- Both models are undertrained within the shared 42-epoch budget, so these results describe augmentation under that budget, not at convergence.
- The full RQ1 table also needs the rotation-only, zoom-only, and brightness-only runs. `shared/brightness_only` records a different split manifest (`575d19ad…`) and must be retrained on the committed manifest before it can be compared with these runs.
- CPU training; CPU and GPU runs of the same seed can differ slightly.
