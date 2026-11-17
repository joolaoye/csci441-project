# depth3: three-layer CNN (RQ2), validation results

Owner: Jon (Yohannes Nigusse). One training run, seed 42. All numbers are **validation** results; the official test partition was not used.

## Setup

| Item | Value |
|---|---|
| Configuration | `configs/experiments/depth3.yaml`: `baseline.yaml` with `experiment: depth3` and `channels: [64, 64, 64]` |
| Architecture | 3 × (3×3 conv, BatchNorm, ReLU), pooling schedule `[0, 1, 1]` → 12×12, GAP, dropout 0.2, dense 128, softmax 43 |
| Parameters | 89,707 trainable / 90,091 total |
| Training | 42 epochs, batch 64, Adam lr 0.001, no augmentation, no early stopping |
| Data | `configs/splits/gtsrb_v1.json` (SHA-256 `e32ea068…98d7`), 31,380 train / 7,829 validation images |
| Code | `git_commit` dccafae, `git_dirty: false` (code identical to `main` at d514b58) |
| Hardware | CPU only (Apple M1 Pro; TensorFlow 2.21.0 exposed no GPU) |
| Training time | 4,306.9 s (71.8 min) |
| Selected checkpoint | Epoch 40, highest clean validation accuracy (0.925916). `best_val_loss` 0.2868 is the loss at that epoch; the lowest loss in the history is 0.2517 at epoch 42 |

## Results: same 7,829 validation images under five conditions

All five evaluations use one checkpoint, the committed `evaluation.yaml`, and identical image IDs (`sample_ids_sha256` `8d88abc9…`).

| Condition | Accuracy | Drop from clean | Macro-F1 | Loss |
|---|---:|---:|---:|---:|
| clean | 92.59% | — | 0.888 | 0.279 |
| zoom (center 1.1×) | 89.85% | 2.75 pp | 0.844 | 0.380 |
| brightness (+0.1) | 88.79% | 3.81 pp | 0.847 | 0.423 |
| rotation (+10°) | 82.90% | 9.69 pp | 0.779 | 0.736 |
| lowres (24→48) | 75.21% | 17.38 pp | 0.644 | 0.957 |

The clean evaluation loss (0.279) differs slightly from the loss Keras logged at epoch 40 during training (0.287); both are reported as recorded.

## Observations

- **Learning curves.** Training and validation loss were both still falling at epoch 42. Validation accuracy first reached 90% at epoch 29, and the selected epoch (40) is near the end of the budget. Validation accuracy rose with large epoch-to-epoch swings (up to 11.8 pp) before about epoch 30, then varied within roughly 88–93%. The final train/validation accuracy gap was about 4.6 pp (97.1% vs 92.5%).
- **Condition ranking.** Zoom and brightness cost under 4 pp. Rotation cost about 10 pp. Low resolution cost the most, at 17 pp, and macro-F1 fell further than accuracy under every altered condition, so the losses are concentrated in particular classes.
- **Weak classes (clean).** Seven classes have F1 below 0.80. The weakest is *Keep left* (class 39): F1 0.18, recall 0.10, with 34 of 60 images predicted as *Turn right ahead*. Others: *Dangerous curve to the right* (0.71; recall 1.00, precision low), *Road narrows on the right* (0.77), *Beware of ice/snow* (0.77), *Roundabout mandatory* (0.78), *Speed limit 20* (0.78). The largest single confusion is *Speed limit 100 → 120* (45 images).
- **Weak classes (altered).** Under lowres, *Roundabout mandatory* has F1 0.00 and *Keep left*, *Speed limit 20*, *Dangerous curve to the right*, *Pedestrians* and *Bicycles crossing* are all at or below 0.24. Under rotation, *Speed limit 20* (0.33) and *Dangerous curve to the left/right* (0.47/0.44) are the weakest.

## Comparison with depth4 (partial: the two-layer baseline is not yet available)

From `scripts/report.py compare --type depth` on `shared/depth3` and `shared/depth4`. The evaluations are compatible: same manifest, image IDs and condition parameters.

| Depth | Params | Epoch | clean | rotation | zoom | brightness | lowres |
|---|---:|---:|---:|---:|---:|---:|---:|
| 3 | 89,707 | 40 | 92.59 | 82.90 (−9.69) | 89.85 (−2.75) | 88.79 (−3.81) | 75.21 (−17.38) |
| 4 | 126,699 | 39 | 98.25 | 96.00 (−2.25) | 96.97 (−1.28) | 97.25 (−1.00) | 91.11 (−7.14) |

Accuracy in %, with the percentage-point drop from that model's own clean accuracy in parentheses. Depth4 reached 96.9% clean validation accuracy by epoch 15; depth3 was at 70.6% at epoch 15.

## Possible explanations (not tested)

- Depth3 had not converged within 42 epochs. Its loss was still decreasing and its best epoch was near the end. Some of the gap to depth4 may reflect convergence speed within the shared budget rather than final capacity.
- The configurations differ in more than depth. Depth4 has about 41% more parameters, and its pools sit at blocks 2 and 4 rather than 2 and 3. These results compare the two configured models, not depth alone.
- *Keep left* has only two validation tracks (60 images), so one misread physical sign moves its score heavily. Its confusion with *Turn right ahead* (both blue circular signs with an arrow) is a plausible but unverified cause.

## Limitations

- One seed per configuration: this is an observed comparison, with no standard deviation or significance test.
- Validation images come from 261 physical tracks of about 30 frames each, so errors are correlated and the swings between epochs are larger than independent images would give.
- The training run was on CPU; CPU and GPU runs of the same seed can differ slightly.
- The two-layer baseline (Chidubem) is not yet in the repository, so the full three-depth RQ2 table is pending.
