"""Accuracy, macro averages, and the confusion matrix.

Macro precision, recall, and F1 are unweighted means over every class the
model can predict. A class with no support or no predictions contributes 0.
That keeps runs with the same ``num_classes`` comparable. Rows of the
confusion matrix are true classes and columns are predicted classes.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def evaluate_predictions(y_true, y_pred, num_classes: int) -> dict:
    """Score integer predictions and return JSON-ready metrics."""
    if num_classes < 1:
        raise ValueError("num_classes must be positive.")
    true = np.asarray(y_true, dtype=np.int64)
    pred = np.asarray(y_pred, dtype=np.int64)
    if true.ndim != 1 or pred.ndim != 1 or true.shape != pred.shape:
        raise ValueError("y_true and y_pred must be 1-D arrays of the same length.")
    if true.size == 0:
        raise ValueError("Cannot score an empty prediction set.")
    if np.any(true < 0) or np.any(true >= num_classes):
        raise ValueError("A label is outside the configured class range.")
    if np.any(pred < 0) or np.any(pred >= num_classes):
        raise ValueError("A prediction is outside the configured class range.")

    matrix = confusion_matrix(true, pred, num_classes)
    per_class = per_class_metrics(matrix)
    return {
        "accuracy": accuracy_from_confusion(matrix),
        "macro_precision": macro_average(per_class, "precision"),
        "macro_recall": macro_average(per_class, "recall"),
        "macro_f1": macro_f1(per_class),
        "num_samples": int(matrix.sum()),
        "num_classes": int(num_classes),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }


def confusion_matrix(y_true, y_pred, num_classes: int) -> np.ndarray:
    """Return a ``[true, predicted]`` count matrix."""
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(matrix, (np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64)), 1)
    return matrix


def per_class_metrics(matrix: np.ndarray) -> list[dict]:
    """Precision, recall, F1, and support for each row of a confusion matrix."""
    matrix = np.asarray(matrix, dtype=np.int64)
    true_positive = np.diag(matrix).astype(np.float64)
    support = matrix.sum(axis=1).astype(np.float64)
    predicted = matrix.sum(axis=0).astype(np.float64)
    false_positive = predicted - true_positive
    false_negative = support - true_positive
    precision = _safe_divide(true_positive, true_positive + false_positive)
    recall = _safe_divide(true_positive, true_positive + false_negative)
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    rows = []
    for class_id in range(matrix.shape[0]):
        rows.append(
            {
                "class_id": class_id,
                "precision": float(precision[class_id]),
                "recall": float(recall[class_id]),
                "f1": float(f1[class_id]),
                "support": int(support[class_id]),
            }
        )
    return rows


def accuracy_from_confusion(matrix: np.ndarray) -> float:
    total = float(np.asarray(matrix).sum())
    if total == 0:
        raise ValueError("Cannot compute accuracy from an empty confusion matrix.")
    return float(np.trace(matrix) / total)


def macro_f1(per_class: list[dict]) -> float:
    if not per_class:
        raise ValueError("Cannot compute macro-F1 without classes.")
    return macro_average(per_class, "f1")


def macro_average(per_class: list[dict], key: str) -> float:
    """Unweighted mean of one per-class score, including zeros."""
    if not per_class:
        raise ValueError(f"Cannot compute macro {key} without classes.")
    return float(np.mean([row[key] for row in per_class]))


def write_predictions_csv(path: Path, samples: list[dict], probabilities) -> None:
    """Write sample id, true class, predicted class, and predicted-class confidence."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape (samples, classes).")
    if len(samples) != probabilities.shape[0]:
        raise ValueError("Each sample needs one probability row.")
    predicted = np.argmax(probabilities, axis=1)
    confidence = probabilities[np.arange(probabilities.shape[0]), predicted]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "true_class", "predicted_class", "confidence"])
        for sample, prediction, score in zip(samples, predicted, confidence, strict=True):
            writer.writerow(
                [sample["id"], int(sample["label"]), int(prediction), f"{float(score):.6f}"]
            )


def _safe_divide(numerator, denominator):
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator > 0,
    )
