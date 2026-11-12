# Traffic Sign CNN Experiments

CSCI 441 Neural Networks project for multi-class traffic sign classification on the [German Traffic Sign Recognition Benchmark](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign). The benchmark has 43 classes, 39,209 training images, and 12,630 official test images. This repository is a small, configuration-driven harness for training and comparing convolutional networks.

The experiments address two questions:

- **RQ1.** How do different data augmentation techniques affect CNN classification accuracy on traffic-sign images under varying visual conditions?
- **RQ2.** How does increasing the number of convolutional layers in a CNN affect classification accuracy on low-resolution or degraded traffic sign images compared to clean images?

`baseline.yaml` is the shared training configuration, and `evaluation.yaml` is the shared evaluation configuration. Use those files so every run starts from the same baseline and the same conditions. `training.yaml.example` and `evaluation.yaml.example` stay in the repository for anyone who wants to try other settings.

Course reports live in [`reports/`](reports/README.md). This file is the guide for the code.

## Setup

Install [uv](https://docs.astral.sh/uv/). The project pins Python 3.12 in `.python-version` because current TensorFlow wheels support Python 3.11, 3.12, and 3.13, not 3.14. From the repository root:

```bash
uv sync
```

`uv sync` creates `.venv` and installs TensorFlow, NumPy, Pillow, PyYAML, Matplotlib, and pytest at the versions in `uv.lock` when that file is present. Run the commands below from the repository root. Relative paths in the YAML files and the commands are resolved from this directory, not from the shell's current directory, because the scripts locate the repository from their own path.

Check which devices TensorFlow can see:

```bash
uv run python -c "import tensorflow as tf; print(tf.config.list_physical_devices())"
```

`device: auto` uses a GPU when that list includes one, including an Apple Silicon GPU when the installed TensorFlow build exposes it. Otherwise it uses the CPU. `device: cpu` forces the CPU. `device: gpu` stops with an error when no GPU is visible, so a run does not silently switch devices. This project does not install a Metal plugin for you. If the command above shows only a CPU, training still works on the CPU.

## Dataset preparation

Images are not stored in Git and the script does not download them. The target dataset is the 43-class [German Traffic Sign Recognition Benchmark](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign). Two extracts of that same benchmark are accepted:

- The Kaggle extract, with `Train.csv`, `Test.csv`, `Train/<class>/Class_Track_Frame.png`, and `Test/<index>.png`.
- The official archive, with `Final_Training/Images/00000` through `00042` and `GT-final_test.csv`.

The class id is the official ClassId, from 0 through 42. A training track is the sequence id in the filename: the middle token of `00020_00004_00002.png`, or the first token of `00004_00023.ppm`. Every frame of that track stays in one split. About 20% of the tracks in each class go to validation. The official test images stay in the test partition and are not used for training or validation. That partition is for the final report, not for choosing a model.

`configs/splits/gtsrb_v1.json` is the 43-class GTSRB split. Commit it and reuse it. Do not regenerate it unless the team has agreed to replace it.

The `--seed` on this command is the split seed. It is independent of the `seed` in an experiment YAML file. The manifest records the split seed. Training never builds a new split; it reads the manifest and opens the image files directly.

`--path` is required. There is no default dataset directory, because each person extracts GTSRB in a different place.

```bash
uv run python scripts/prepare.py \
  --path /path/to/gtsrb \
  --seed 42
```

That writes `configs/splits/gtsrb_v1.json`. Paths inside the file are relative to the dataset directory, so the same manifest works for every extract. `--path` accepts `~` and an absolute path. The command prints a SHA-256 hash of the manifest. Training stores that hash in `metadata.json`.

If the manifest already exists, the command refuses to replace it. Pass `--overwrite` only when you intend to create a new split. Commit the manifest after it is generated. Do not commit the images.

Sample ids in the manifest are joined to the directory you pass. The example configuration does not set `data.root`. `--data-root` is required for training and evaluation, and training records the path you passed in that run's `config.yaml`. Figure generation needs it only for an augmentation example.

```bash
uv run python scripts/train.py \
  --config baseline.yaml \
  --data-root /path/to/gtsrb
```

`~` is expanded. A relative path is resolved from the repository root.

## Training, evaluation, and prediction

```bash
uv run python scripts/train.py \
  --config baseline.yaml \
  --data-root /path/to/gtsrb

uv run python scripts/evaluate.py \
  --checkpoint artifacts/rotation10/model.keras \
  --split validation \
  --conditions clean rotation zoom brightness lowres \
  --data-root /path/to/gtsrb

uv run python scripts/predict.py \
  --checkpoint artifacts/rotation10/model.keras \
  --image path/to/image.png
```

Each command accepts `--help`.

Training loads the YAML file, checks it, seeds Python, NumPy, and TensorFlow, builds one CNN, and calls Keras `model.fit`. A `ModelCheckpoint` callback keeps the weights with the highest clean validation accuracy and writes them to `model.keras`. Validation images used for that choice are not rotated, zoomed, brightened, or downsampled. The training budget still runs every configured epoch; there is no early stopping. The artifact folder is the `experiment` value, such as `artifacts/depth2` or `artifacts/rotation10`. Set that name in the YAML file. Training the same experiment again replaces that folder, including its checkpoint, metrics, and evaluations.

Evaluation loads that checkpoint and does not train. It writes a new JSON file under `artifacts/<experiment>/evaluations/` and does not change `metrics.json`. `--split validation` is the split to use while comparing augmentation and depth. `--split test` scores the official GTSRB test partition and records a warning. Use it for the final numbers, not for deciding which configuration to keep.

`--conditions` scores every listed condition in that one command and loads the checkpoint once. `--condition` scores a single condition and defaults to `clean` when neither flag is passed. Both read `evaluation.yaml`, or `evaluation.yaml.example` until that copy exists. `clean` is the normal preprocessing. `rotation`, `zoom`, `brightness`, and `lowres` change the images after that preprocessing and do not change their size or labels. Rotation is counterclockwise, sampled bilinearly, with nearest-border fill. Zoom is a center zoom-in: `factor: 0.1` uses scale 1.1, keeps the center, and bilinearly restores the model input. That is separate from training zoom, which randomly draws a scale from `[1 - zoom, 1 + zoom]`. Brightness adds `delta` to intensities in `[0, 1]` and clips. Low resolution bilinearly shrinks the image, with antialiasing, then bilinearly restores the model input size. The condition name stays generic. Change `degrees`, `factor`, `delta`, or `image_size` in the YAML file instead of creating a name such as `rotation10`. A later evaluation at a different value writes a separate file. The same saved checkpoint is scored for every condition.

```bash
uv run python scripts/evaluate.py \
  --checkpoint artifacts/depth2/model.keras \
  --split validation \
  --condition rotation \
  --data-root /path/to/gtsrb
```

Prediction prints the class id, the GTSRB class description when the model has 43 outputs, and the softmax confidence. There is no separate export command. `keras.models.load_model` reads the `.keras` checkpoint directly.

Evaluation JSON includes accuracy, macro precision, macro recall, and macro-F1. Macro averages are the unweighted mean over every class the model can predict, and a class with no support or no predictions contributes 0. The confusion matrix has true classes on the rows and predicted classes on the columns. Per-class precision, recall, F1, support, and the class name are stored with it. `predictions.csv` lists each sample id, true class, predicted class, and the predicted class's confidence.

## Reporting

Figures and tables are built from saved files. The commands do not train a model or download the dataset. Figures are PNG files at 300 DPI.

```bash
uv run python scripts/report.py run \
  --run artifacts/<experiment> \
  --data-root /path/to/gtsrb

uv run python scripts/report.py dataset \
  --manifest configs/splits/gtsrb_v1.json

uv run python scripts/report.py conditions \
  --split validation \
  --data-root /path/to/gtsrb

uv run python scripts/report.py compare \
  --type augmentation \
  --runs artifacts/<baseline-run> artifacts/<augmented-run>

uv run python scripts/report.py compare \
  --type depth \
  --runs artifacts/<depth2-run> artifacts/<depth3-run> artifacts/<depth4-run>

uv run python scripts/report.py compare \
  --type depth \
  --runs artifacts/<depth2-run> artifacts/<depth3-run> artifacts/<depth4-run> \
  --conditions clean rotation zoom brightness lowres

uv run python scripts/report.py compare \
  --type augmentation \
  --runs artifacts/<baseline-run> artifacts/<rotation-run> artifacts/<zoom-run> \
    artifacts/<brightness-run> artifacts/<combined-run> \
  --conditions clean rotation zoom brightness lowres
```

`run` writes figures under `artifacts/<experiment>/figures/`: training and validation accuracy, training and validation loss, a count-annotated confusion matrix and a separate row-normalized matrix, and a per-class F1 chart. Nonzero count cells are labeled and zero cells are left blank. The matrix size comes from the saved evaluation. The selected checkpoint epoch is marked on the training curves. `per_class.csv` copies the stored per-class precision, recall, F1, and support. Pass `--split test` only after `scripts/evaluate.py` has written that split. An augmentation example is written only when that run's saved settings enable at least one transform, using `--data-root` and a fixed preview seed. The figure shows the original image and one panel for each nonzero factor. A rotation-only run does not include brightness or zoom. A run with every factor at 0 skips the figure.

`dataset` reads the committed split manifest and writes `class_distribution.png` and `dataset_summary.csv` under `artifacts/dataset/`. It does not use a training run.

`conditions` writes PNG previews of the evaluation transforms under `artifacts/evaluation_previews/<split>/<fingerprint>/`. Each preview shows a fixed set of real images beside the transformed versions and records the parameters in `metadata.json`. A matching preview is reused. Changing a parameter, the image size, the split, or the manifest creates a new directory. This is separate from the training-augmentation example, and it does not run the model. Pass `--condition rotation` to preview one condition.

`compare --type augmentation` is RQ1. Augmentation settings may differ. Architecture, split manifest, and training protocol must match, and one run must be the unaugmented baseline. `compare --type depth` is RQ2. Convolutional depth may differ. Augmentation, split manifest, and training protocol must match. Both commands use validation metrics from the selected checkpoint. A configuration with two or more distinct seeds shows the mean and sample standard deviation. A single seed does not get a standard deviation. Each comparison writes `comparison.csv`, `comparison.tex`, and `comparison.png` under `artifacts/comparisons/<type>/<comparison-id>/`. The CSV keeps one row per run. The LaTeX file is a `table` environment for `\input{}` and needs `\usepackage{booktabs}`.

`--conditions` reads evaluation files that `scripts/evaluate.py` already wrote. It does not train or evaluate. It works for both `--type depth` and `--type augmentation`. Every run needs one validation result for every requested condition, with the same split, image ids, and parameters. If one condition name has two different parameter sets, the comparison stops instead of choosing one. The LaTeX table shows validation accuracy and is a \texttt{tabular*} of width \texttt{\textbackslash textwidth}, so it spans the page inside the margins. For depth it shows the number of convolutional layers. For augmentation it shows the training strategy. Trainable parameter counts stay in the CSV. The CSV keeps macro-F1, validation loss, the selected checkpoint epoch, the configuration, and the accuracy drop from the clean condition when that condition is included. Omit `--conditions` for the clean summary table. Training augmentation and these evaluation transforms stay separate.

The clean RQ1 summary lists the augmentation setting, validation accuracy, macro-F1, and the validation loss at the selected checkpoint. The selected checkpoint epoch stays in the CSV and in the clean depth table.

Those generated files stay under `artifacts/` and are not copied into `reports/`. After review, copy the table and figure you want to keep:

```text
reports/tables/rq1_comparison.tex
reports/tables/rq2_comparison.tex
reports/figures/rq1_comparison.png
reports/figures/rq2_comparison.png
```

## Configuration

```yaml
experiment: baseline
seed: 42
device: auto

data:
  split_manifest: configs/splits/gtsrb_v1.json
  image_size: 48
  num_classes: 43

model:
  channels: [64, 64]
  kernel_size: 3
  dropout: 0.2
  dense_units: 128
  min_spatial: 12

training:
  epochs: 42
  batch_size: 64
  learning_rate: 0.001
  optimizer: adam

augmentation:
  rotation_degrees: 0
  brightness: 0
  zoom: 0
```

`model.channels` is one convolutional block per entry. `[64, 64]` is the shared two-layer baseline. `[64, 64, 128]` is three layers, and `[64, 64, 128, 128]` is four. There is one model definition. For a three-layer run, copy `baseline.yaml`, set `experiment` to `depth3`, and set `model.channels` to `[64, 64, 128]`. For a rotation run, set `experiment` to `rotation10` and change `augmentation.rotation_degrees`. Keep the split manifest, image size, optimizer, learning rate, batch size, and epoch count the same across runs you plan to compare.

Each block is a same-padded convolution, batch normalization, and ReLU. The convolution has no bias because batch normalization supplies the shift. Pooling is not tied to depth. On a 48×48 input with `min_spatial: 12`, every depth uses two 2×2 pooling stages and ends at a 12×12 map. The pools are spaced through the network. Global average pooling, dropout, a 128-unit ReLU layer, dropout, and a 43-way softmax follow. The saved run metadata includes the pooling schedule and the final map size.

A setting of 0 disables that augmentation. Positive values apply only to training batches:

- `rotation_degrees` is the maximum rotation, in degrees, in either direction. The allowed range is 0 to 20.
- `brightness` is an additive shift on pixels already scaled to `[0, 1]`. `0.1` means a shift of at most ±0.1. The allowed range is 0 to 0.2.
- `zoom` draws one scale from `[1 - zoom, 1 + zoom]` and applies it to both axes, so the sign is rescaled without being stretched. The allowed range is 0 to 0.2.

Values are clipped back to `[0, 1]`. Validation and test images are only converted to RGB, resized, and divided by 255. The example file leaves every augmentation factor at 0.

Unknown YAML keys are errors, so a typo cannot silently drop a setting. If `dense_units`, `min_spatial`, `optimizer`, or `num_classes` is omitted, the resolved configuration uses 128, 12, `adam`, and 43. The copy written to the run directory always includes those values. The optimizer must be `adam`.

## Logs and artifacts

Logs use the standard library and record the experiment name, timestamp, resolved configuration, seeds, split hash, framework versions, device, epoch metrics, warnings, and errors. They do not record every batch. Retraining an experiment replaces `logs/train/<experiment>.log`.

```text
logs/
├── prepare/
├── train/
├── evaluate/
└── predict/
```

A training run:

```text
artifacts/<experiment>/
├── config.yaml      # resolved configuration used for this run
├── metadata.json    # experiment name, seeds, versions, device, git commit, manifest hash, status
├── model.keras      # highest clean-validation-accuracy checkpoint
├── metrics.json     # that checkpoint's validation accuracy and macro-F1
├── history.csv      # one row per completed epoch
└── evaluations/     # later evaluation files; metrics.json is not rewritten
```

`metadata.json` is written at the start with status `running` and updated at the end. If training fails, status is `failed`, the error is stored, and the log contains the traceback. Epochs completed before the failure remain in `history.csv`. The manifest hash is SHA-256 of the manifest file bytes. Macro-F1 is the unweighted mean of the per-class F1 scores over all classes the model can predict, including a class with no images.

Downloaded images, training runs, and logs stay on each machine and are not committed. `baseline.yaml`, `evaluation.yaml`, both example files, and the split manifest are meant to be committed. The `config.yaml` inside a run directory is that run's resolved settings and stays with the run.

## Sharing a run

`shared/` is the drop folder for a finished experiment. Copy `artifacts/<experiment>/` into `shared/<experiment>/` and commit that folder. A teammate points the same commands at `shared/<experiment>/`. See [shared/README.md](shared/README.md) for the files to include. They still pass their own `--data-root`.

Selected comparison files can later be copied into `reports/tables/` and `reports/figures/`. Nothing here publishes or syncs them.

## Reproducing a teammate's experiment

1. Check out the git commit recorded as `git_commit` in that run's `metadata.json`. `git_dirty: true` means the work tree had extra changes, so the commit alone is not the full code state.
2. Use the committed `configs/splits/gtsrb_v1.json`. Its SHA-256 must match `split_manifest_sha256`. Do not regenerate the manifest unless the team has agreed to replace it.
3. Use a GTSRB extract with the same layout the manifest describes. `scripts/prepare.py` without `--overwrite` will refuse to replace the committed manifest.
4. Run `uv sync`, then `scripts/train.py` with the same settings and `--data-root` pointed at your own extract. The path is required. The `config.yaml` inside `artifacts/<experiment>/` records the path used for that run, and another machine still passes its own `--data-root`. Training the same experiment name replaces that folder.
5. Compare `metrics.json`. The training seed and the split seed are separate fields. CPU and GPU runs can differ slightly. For a closer comparison, set `device: cpu` in a copy of the configuration.

## Tests

```bash
uv run pytest
```

The tests cover configuration validation, the example configuration, CNN output shape at several depths, the shared pooling schedule, deterministic track splits, preprocessing range, checkpoint reload, metric arithmetic, report figures, comparison tables, and a one-epoch smoke training run on synthetic images. They do not download the dataset or train on the real images.

## Reports

Proposal and report sources are in [`reports/`](reports/README.md). Leave those files on their own review path. LaTeX build products stay out of Git.
