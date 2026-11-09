"""Paper figures drawn from saved run artifacts.

Training curves, confusion matrices, per-class scores, class counts, and
augmentation examples are read from files the training and evaluation
commands already write. This module does not retrain a model or recompute
precision, recall, or F1.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from traffic_signs.data.labels import class_name_for_model
from traffic_signs.data.splits import load_manifest
from traffic_signs.reporting import ReportError
from traffic_signs.utils.config import load_config, override_data_root
from traffic_signs.utils.runs import read_json, resolve_path

TRAIN_COLOR = "#1f4e79"
VAL_COLOR = "#d4760a"
TEST_COLOR = "#2f6f4e"
HISTORY_FIELDS = ["epoch", "loss", "accuracy", "val_loss", "val_accuracy"]
AUGMENTATION_KEYS = ("rotation_degrees", "brightness", "zoom")


def read_history(path: Path) -> list[dict]:
    """Read ``history.csv`` written by training."""
    path = Path(path)
    if not path.is_file():
        raise ReportError(f"Training history not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != HISTORY_FIELDS:
            found = ", ".join(reader.fieldnames or [])
            raise ReportError(
                f"{path} must have columns {', '.join(HISTORY_FIELDS)}. Found: {found or 'none'}."
            )
        rows = []
        for line_number, row in enumerate(reader, start=2):
            parsed = {"epoch": _require_int(row["epoch"], path, line_number, "epoch")}
            for field in HISTORY_FIELDS[1:]:
                parsed[field] = _require_float(row[field], path, line_number, field)
            rows.append(parsed)
    if not rows:
        raise ReportError(f"{path} has no completed epochs.")
    return rows


def row_normalize_confusion(matrix) -> np.ndarray:
    """Scale each true-class row to sum to 1. A row with no support stays 0."""
    counts = np.asarray(matrix, dtype=np.float64)
    if counts.ndim != 2 or counts.shape[0] != counts.shape[1] or counts.shape[0] == 0:
        raise ReportError("A confusion matrix must be a non-empty square array.")
    if np.any(counts < 0):
        raise ReportError("A confusion matrix cannot contain negative counts.")
    support = counts.sum(axis=1, keepdims=True)
    normalized = np.zeros_like(counts, dtype=np.float64)
    np.divide(counts, support, out=normalized, where=support > 0)
    return normalized


def load_split_evaluation(run_dir: Path, split: str) -> dict:
    """Load the newest evaluation JSON for ``split``, or the training metrics file."""
    matches = sorted((Path(run_dir) / "evaluations").glob(f"*_{split}.json"))
    if matches:
        return read_json(matches[-1])
    metrics_path = Path(run_dir) / "metrics.json"
    if metrics_path.is_file():
        metrics = read_json(metrics_path)
        if metrics.get("split") == split:
            return metrics
    raise ReportError(
        f"No {split} evaluation found in {run_dir}. "
        "Training writes validation scores to metrics.json. "
        "Score another split with scripts/evaluate.py before reporting it."
    )


def class_counts(manifest: dict) -> list[dict]:
    """Training, validation, and test image counts in class-id order."""
    per_class = manifest.get("per_class")
    if per_class:
        return [
            {
                "label": int(row["label"]),
                "train": int(row["train_images"]),
                "validation": int(row["validation_images"]),
                "test": int(row["test_images"]),
            }
            for row in per_class
        ]
    labels = range(int(manifest["num_classes"]))
    counts = []
    samples = manifest.get("samples") or {}
    for label in labels:
        counts.append(
            {
                "label": label,
                "train": _count_label(samples.get("train", []), label),
                "validation": _count_label(samples.get("validation", []), label),
                "test": _count_label(samples.get("test", []), label),
            }
        )
    return counts


def write_per_class_csv(evaluation: dict, path: Path) -> Path:
    """Write the per-class scores already stored in an evaluation file."""
    rows = evaluation.get("per_class")
    if not isinstance(rows, list) or not rows:
        raise ReportError("The evaluation file has no per-class scores.")
    num_classes = int(evaluation.get("num_classes") or len(rows))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class_id", "name", "support", "precision", "recall", "f1"])
        for row in rows:
            writer.writerow(
                [
                    int(row["class_id"]),
                    _class_name(row, num_classes),
                    int(row["support"]),
                    f"{float(row['precision']):.6f}",
                    f"{float(row['recall']):.6f}",
                    f"{float(row['f1']):.6f}",
                ]
            )
    return path


def plot_training_curves(history: list[dict], checkpoint_epoch: int | None, output_dir: Path, title: str) -> list[Path]:
    """Plot accuracy and loss, and mark the checkpoint epoch."""
    epochs = [row["epoch"] for row in history]
    written = []
    written += _curve(
        epochs,
        [row["accuracy"] for row in history],
        [row["val_accuracy"] for row in history],
        checkpoint_epoch,
        output_dir / "training_accuracy",
        title=title,
        ylabel="Accuracy",
    )
    written += _curve(
        epochs,
        [row["loss"] for row in history],
        [row["val_loss"] for row in history],
        checkpoint_epoch,
        output_dir / "training_loss",
        title=title,
        ylabel="Loss",
    )
    return written


def plot_confusion_matrices(evaluation: dict, output_dir: Path, title: str) -> list[Path]:
    """Draw the stored count matrix and its row-normalized form."""
    if "num_classes" not in evaluation:
        raise ReportError("The evaluation file does not record num_classes.")
    counts = np.asarray(evaluation["confusion_matrix"], dtype=np.float64)
    num_classes = int(evaluation["num_classes"])
    if counts.shape != (num_classes, num_classes):
        raise ReportError(
            f"The confusion matrix has shape {counts.shape}, expected {(num_classes, num_classes)}."
        )
    labels = [str(class_id) for class_id in range(num_classes)]
    written = _heatmap(
        counts,
        labels,
        output_dir / "confusion_matrix",
        title=f"{title}\nConfusion matrix (counts)",
        colorbar="Images",
        integer=True,
    )
    written += _heatmap(
        row_normalize_confusion(counts),
        labels,
        output_dir / "confusion_matrix_normalized",
        title=f"{title}\nConfusion matrix (row-normalized)",
        colorbar="Share of true class",
        integer=False,
    )
    return written


def plot_per_class_f1(evaluation: dict, output_dir: Path, title: str) -> list[Path]:
    """Horizontal bars of the stored per-class F1 scores, in class-id order."""
    rows = sorted(evaluation["per_class"], key=lambda row: int(row["class_id"]))
    num_classes = int(evaluation.get("num_classes") or len(rows))
    labels = [f"{int(row['class_id'])} {_class_name(row, num_classes)}".rstrip() for row in rows]
    scores = [float(row["f1"]) for row in rows]
    figure, axis = plt.subplots(figsize=(8, max(4.0, 0.28 * len(rows) + 1.2)))
    positions = np.arange(len(rows))
    axis.barh(positions, scores, color=TRAIN_COLOR)
    axis.set_yticks(positions, labels, fontsize=7 if len(rows) > 20 else 9)
    axis.invert_yaxis()
    axis.set_xlim(0, 1)
    axis.set_xlabel("F1")
    axis.set_title(f"{title}\nPer-class F1")
    axis.axvline(float(np.mean(scores)), color=VAL_COLOR, linestyle="--", linewidth=1, label="Macro-F1")
    axis.legend(frameon=False)
    figure.tight_layout()
    return save_figure(figure, output_dir / "per_class_f1")


def plot_class_distribution(manifest: dict, output_dir: Path) -> list[Path]:
    """Grouped bars of training, validation, and test counts from the split manifest."""
    counts = class_counts(manifest)
    labels = [str(row["label"]) for row in counts]
    x = np.arange(len(counts))
    width = 0.25
    figure, axis = plt.subplots(figsize=(max(8, 0.28 * len(counts) + 2), 4.8))
    axis.bar(x - width, [row["train"] for row in counts], width, label="Training", color=TRAIN_COLOR)
    axis.bar(x, [row["validation"] for row in counts], width, label="Validation", color=VAL_COLOR)
    axis.bar(x + width, [row["test"] for row in counts], width, label="Test", color=TEST_COLOR)
    axis.set_xticks(x, labels, rotation=90, fontsize=6 if len(counts) > 20 else 9)
    axis.set_xlabel("Class")
    axis.set_ylabel("Images")
    axis.set_title("Class distribution")
    axis.legend(frameon=False)
    figure.tight_layout()
    return save_figure(figure, output_dir / "class_distribution")


def enabled_augmentations(augmentation: dict) -> list[tuple[dict, str]]:
    """Return one isolated setting and title for each nonzero augmentation factor."""
    rotation = float(augmentation["rotation_degrees"])
    brightness = float(augmentation["brightness"])
    zoom = float(augmentation["zoom"])
    enabled = []
    if rotation:
        enabled.append(
            (
                {"rotation_degrees": rotation, "brightness": 0.0, "zoom": 0.0},
                f"Rotation {rotation:g}°",
            )
        )
    if brightness:
        enabled.append(
            (
                {"rotation_degrees": 0.0, "brightness": brightness, "zoom": 0.0},
                f"Brightness {brightness:g}",
            )
        )
    if zoom:
        enabled.append(
            (
                {"rotation_degrees": 0.0, "brightness": 0.0, "zoom": zoom},
                f"Zoom {zoom:g}",
            )
        )
    return enabled


def augmentation_panels(image: np.ndarray, image_size: int, augmentation: dict, seed: int = 0) -> list[np.ndarray]:
    """Return the original image and one view for each enabled augmentation.

    A rotation-only run returns the original and the rotated image. Brightness
    and zoom are included only when those factors are nonzero. Each transform
    is applied on its own with ``seed``.
    """
    import tensorflow as tf

    from traffic_signs.data.transforms import apply_augmentation, build_augmentor, preprocess_array

    original = preprocess_array(np.asarray(image), image_size)
    panels = [original]
    for settings, _title in enabled_augmentations(augmentation):
        tf.keras.utils.set_random_seed(seed)
        augmentor = build_augmentor(settings, seed)
        batch = apply_augmentation(original[None, ...], augmentor, training=True)
        panels.append(np.asarray(batch)[0])
    return panels


def plot_augmentation_examples(
    image_path: Path,
    image_size: int,
    augmentation: dict,
    output_dir: Path,
    *,
    seed: int = 0,
    source_note: str,
) -> list[Path]:
    """Save the original image and one panel for each enabled augmentation."""
    from traffic_signs.data.transforms import load_rgb

    image = load_rgb(image_path)
    panels = augmentation_panels(image, image_size, augmentation, seed)
    titles = ["Original", *[title for _settings, title in enabled_augmentations(augmentation)]]
    figure, axes = plt.subplots(1, len(panels), figsize=(2.6 * len(panels), 3.2), squeeze=False)
    for axis, panel, panel_title in zip(axes.ravel(), panels, titles, strict=True):
        axis.imshow(np.clip(panel, 0.0, 1.0))
        axis.set_title(panel_title, fontsize=10)
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(f"Augmentation examples ({source_note}, seed {seed})", fontsize=11)
    figure.tight_layout()
    return save_figure(figure, output_dir / "augmentation_examples")


def generate_run_report(
    run_dir: Path,
    *,
    output_dir: Path | None = None,
    split: str = "validation",
    data_root: str | None = None,
    project_dir: Path | None = None,
    augmentation_seed: int = 0,
) -> dict:
    """Write the figures and per-class table for one completed run."""
    run_dir = Path(run_dir)
    project_dir = Path(project_dir or Path.cwd())
    output_dir = Path(output_dir or run_dir / "figures")
    config_path = run_dir / "config.yaml"
    metadata_path = run_dir / "metadata.json"
    if not config_path.is_file() or not metadata_path.is_file():
        raise ReportError(f"{run_dir} is missing config.yaml or metadata.json.")
    config = load_config(config_path)
    if data_root is not None:
        override_data_root(config, data_root)
    metadata = read_json(metadata_path)
    if metadata.get("status") != "completed":
        raise ReportError(
            f"Run {metadata.get('run_id', run_dir.name)} has status "
            f"{metadata.get('status')!r}. Reporting uses a completed run."
        )
    history = read_history(run_dir / "history.csv")
    evaluation = load_split_evaluation(run_dir, split)
    title = _run_title(metadata, evaluation, split)
    written = []
    written += plot_training_curves(history, _checkpoint_epoch(metadata), output_dir, title)
    written += plot_confusion_matrices(evaluation, output_dir, title)
    written.append(write_per_class_csv(evaluation, output_dir / "per_class.csv"))
    written += plot_per_class_f1(evaluation, output_dir, title)
    warnings = []
    if augmentation_enabled(config["augmentation"]):
        try:
            manifest = load_manifest(resolve_path(config["data"]["split_manifest"], project_dir))
            written += _augmentation_figure(
                config,
                manifest,
                output_dir,
                project_dir,
                seed=augmentation_seed,
            )
        except Exception as exc:
            warnings.append(f"Skipped the augmentation figure: {exc}")
    return {"output_dir": output_dir, "written": written, "warnings": warnings}


def augmentation_enabled(augmentation: dict) -> bool:
    """True when the saved run applies at least one nonzero augmentation."""
    return any(float(augmentation[key]) != 0 for key in AUGMENTATION_KEYS)


def generate_dataset_report(manifest_path: Path, output_dir: Path) -> dict:
    """Write the class distribution and summary counts for one split manifest."""
    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)
    manifest = load_manifest(manifest_path)
    written = plot_class_distribution(manifest, output_dir)
    summary_path = output_dir / "dataset_summary.csv"
    _write_dataset_summary(manifest, summary_path)
    written.append(summary_path)
    return {"output_dir": output_dir, "written": written, "manifest": manifest_path}


def save_figure(figure, path: Path) -> list[Path]:
    """Write one PNG figure at publication resolution, then close it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    destination = path.with_suffix(".png")
    figure.savefig(destination, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return [destination]


def _curve(epochs, train, validation, checkpoint_epoch, path, *, title, ylabel) -> list[Path]:
    figure, axis = plt.subplots(figsize=(6.2, 4.0))
    axis.plot(epochs, train, color=TRAIN_COLOR, marker="o", markersize=3, label="Training")
    axis.plot(epochs, validation, color=VAL_COLOR, marker="o", markersize=3, label="Validation")
    if checkpoint_epoch in epochs:
        marker = validation[epochs.index(checkpoint_epoch)]
        axis.axvline(checkpoint_epoch, color="0.35", linestyle="--", linewidth=1, label="Selected checkpoint")
        axis.scatter([checkpoint_epoch], [marker], color="0.15", zorder=3)
    axis.set_xlabel("Epoch")
    axis.set_ylabel(ylabel)
    axis.set_title(f"{title}\n{ylabel}")
    axis.legend(frameon=False)
    figure.tight_layout()
    return save_figure(figure, path)


def count_annotations(matrix) -> list[tuple[int, int, str]]:
    """Integer labels for cells that contain at least one image."""
    values = np.asarray(matrix)
    annotations = []
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            count = int(values[row, column])
            if count:
                annotations.append((row, column, str(count)))
    return annotations


def _heatmap(matrix, labels, path, *, title, colorbar, integer: bool) -> list[Path]:
    side = max(6.0, 0.22 * len(labels) + 2.0)
    if integer and len(labels) > 20:
        side = max(side, 0.38 * len(labels) + 2.0)
    figure, axis = plt.subplots(figsize=(side, side * 0.92))
    image = axis.imshow(matrix, cmap="Blues", vmin=0, interpolation="nearest")
    bar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    bar.set_label(colorbar)
    tick_size = 6 if len(labels) > 20 else 8
    axis.set_xticks(np.arange(len(labels)), labels, rotation=90, fontsize=tick_size)
    axis.set_yticks(np.arange(len(labels)), labels, fontsize=tick_size)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title(title)
    annotate = integer or len(labels) <= 12
    if annotate:
        limit = float(np.max(matrix)) if matrix.size else 1.0
        font_size = 4.5 if len(labels) > 20 else 7
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix[row, column]
                if integer:
                    if int(value) == 0:
                        continue
                    text = str(int(value))
                elif value == 0:
                    continue
                else:
                    text = f"{value:.2f}"
                color = "white" if limit and value > limit * 0.6 else "black"
                axis.text(column, row, text, ha="center", va="center", fontsize=font_size, color=color)
    figure.tight_layout()
    return save_figure(figure, path)


def _write_dataset_summary(manifest: dict, path: Path) -> None:
    counts = class_counts(manifest)
    names = manifest.get("class_names") or []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "class_id",
                "name",
                "train_images",
                "validation_images",
                "test_images",
            ]
        )
        train_total = validation_total = test_total = 0
        for row in counts:
            label = int(row["label"])
            name = names[label] if label < len(names) and names[label] else _manifest_class_name(label, manifest)
            writer.writerow([label, name, row["train"], row["validation"], row["test"]])
            train_total += row["train"]
            validation_total += row["validation"]
            test_total += row["test"]
        writer.writerow(["total", "", train_total, validation_total, test_total])


def _manifest_class_name(label: int, manifest: dict) -> str:
    from traffic_signs.data.labels import class_name_for_model

    return class_name_for_model(label, int(manifest["num_classes"])) or ""


def _example_image(manifest: dict, data_root: Path) -> Path | None:
    samples = sorted(manifest["samples"]["train"], key=lambda sample: sample["id"])
    for sample in samples:
        path = data_root / sample["id"]
        if path.is_file():
            return path
    return None


def _augmentation_figure(
    config: dict,
    manifest: dict,
    output_dir: Path,
    project_dir: Path,
    *,
    seed: int,
) -> list[Path]:
    """Preview the augmentation settings stored in this run's configuration."""
    image_path = _example_image(manifest, resolve_path(config["data"]["root"], project_dir))
    if image_path is None:
        raise ReportError("no training image was found under data.root.")
    factors = {key: float(config["augmentation"][key]) for key in AUGMENTATION_KEYS}
    return plot_augmentation_examples(
        image_path,
        int(config["data"]["image_size"]),
        factors,
        output_dir,
        seed=seed,
        source_note=f"factors from this run; {image_path.name}",
    )


def _run_title(metadata: dict, evaluation: dict, split: str) -> str:
    run_id = metadata.get("run_id") or evaluation.get("run_id") or "run"
    experiment = metadata.get("experiment") or evaluation.get("experiment") or ""
    checkpoint = evaluation.get("checkpoint") or metadata.get("checkpoint") or ""
    checkpoint_name = Path(str(checkpoint)).name if checkpoint else "checkpoint"
    return f"{experiment} ({run_id}), {split}, {checkpoint_name}"


def _checkpoint_epoch(metadata: dict) -> int | None:
    value = metadata.get("checkpoint_epoch")
    if value is None:
        return None
    return int(value)


def _class_name(row: dict, num_classes: int) -> str:
    name = row.get("name")
    if isinstance(name, str) and name.strip():
        return name
    return class_name_for_model(int(row["class_id"]), num_classes) or ""


def _count_label(samples: list[dict], label: int) -> int:
    return sum(int(sample["label"]) == label for sample in samples)


def _require_int(value, path: Path, line_number: int, field: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ReportError(f"{path}:{line_number} has a non-integer {field}.") from exc


def _require_float(value, path: Path, line_number: int, field: str) -> float:
    if value is None or str(value).strip() == "":
        raise ReportError(f"{path}:{line_number} is missing {field}.")
    try:
        return float(value)
    except ValueError as exc:
        raise ReportError(f"{path}:{line_number} has a non-numeric {field}.") from exc
