"""Batches of preprocessed traffic-sign images for Keras."""

from __future__ import annotations

import inspect
import math
from pathlib import Path

import numpy as np
from tensorflow import keras

from traffic_signs.data.splits import DatasetError
from traffic_signs.data.transforms import apply_augmentation, build_augmentor, load_rgb, preprocess_array


def make_sequence(
    samples: list[dict],
    root: Path,
    image_size: int,
    batch_size: int,
    training: bool,
    augmentation: dict | None,
    seed: int,
    condition: dict | None = None,
) -> "TrafficSignSequence":
    """Build a dataset sequence. Stochastic augmentation is used only for training.

    ``condition`` is a deterministic evaluation transform. It is ignored while
    training so a robustness check cannot leak into a fit.
    """
    augmentor = None
    if training and augmentation is not None:
        augmentor = build_augmentor(augmentation, seed)
    return TrafficSignSequence(
        samples,
        root=root,
        image_size=image_size,
        batch_size=batch_size,
        training=training,
        augmentor=augmentor,
        seed=seed,
        condition=None if training else condition,
    )


def ensure_files_exist(root: Path, samples: list[dict]) -> None:
    """Fail before training when the manifest points at missing images."""
    missing = [sample["id"] for sample in samples if not (root / sample["id"]).is_file()]
    if not missing:
        return
    preview = ", ".join(missing[:5])
    raise DatasetError(
        f"{len(missing)} manifest images are missing under {root}. Examples: {preview}. "
        "Check data.root and regenerate the manifest only if this machine's layout differs."
    )


_DatasetBase = getattr(keras.utils, "PyDataset", None) or keras.utils.Sequence


class TrafficSignSequence(_DatasetBase):
    """Yield preprocessed batches. Training order is shuffled once per epoch."""

    def __init__(
        self,
        samples: list[dict],
        root: Path,
        image_size: int,
        batch_size: int,
        training: bool,
        augmentor,
        seed: int,
        condition: dict | None = None,
    ):
        super().__init__(**_dataset_init_kwargs())
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        self.samples = list(samples)
        self.root = Path(root)
        self.image_size = image_size
        self.batch_size = batch_size
        self.training = training
        self.augmentor = augmentor if training else None
        self.condition = None if training else condition
        self.indices = np.arange(len(self.samples))
        self.rng = np.random.default_rng(seed)
        if self.training:
            self.rng.shuffle(self.indices)

    def __len__(self) -> int:
        if not self.samples:
            return 0
        return math.ceil(len(self.samples) / self.batch_size)

    def __getitem__(self, index):
        start = int(index) * self.batch_size
        batch_indices = self.indices[start : start + self.batch_size]
        images = []
        labels = []
        for sample_index in batch_indices:
            sample = self.samples[int(sample_index)]
            path = self.root / sample["id"]
            if not path.is_file():
                raise DatasetError(f"Missing image file: {path}")
            images.append(preprocess_array(load_rgb(path), self.image_size))
            labels.append(int(sample["label"]))
        batch_images = np.stack(images).astype(np.float32)
        batch_images = apply_augmentation(batch_images, self.augmentor, self.training)
        if self.condition is not None:
            from traffic_signs.evaluation.conditions import apply_condition_batch

            batch_images = apply_condition_batch(batch_images, self.condition, self.image_size)
        batch_labels = np.asarray(labels, dtype=np.int32)
        return batch_images, batch_labels

    def on_epoch_end(self):
        parent = getattr(super(), "on_epoch_end", None)
        if callable(parent):
            parent()
        if self.training:
            self.rng.shuffle(self.indices)


def _dataset_init_kwargs() -> dict:
    """Ask the dataset base class for a single worker. Multiprocessing breaks the augmentor."""
    signature = inspect.signature(_DatasetBase.__init__)
    accepts_keywords = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    names = set(signature.parameters)
    kwargs = {}
    if "workers" in names or accepts_keywords:
        kwargs["workers"] = 1
    if "use_multiprocessing" in names or accepts_keywords:
        kwargs["use_multiprocessing"] = False
    return kwargs


def predict_probabilities(model: keras.Model, sequence: TrafficSignSequence) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(probabilities, true_labels)`` in manifest order."""
    if sequence.training:
        raise ValueError("Evaluation must use a sequence with training=False.")
    if len(sequence) == 0:
        raise DatasetError("Cannot evaluate an empty split.")
    probabilities = np.asarray(model.predict(sequence, verbose=0))
    labels = np.asarray([int(sample["label"]) for sample in sequence.samples], dtype=np.int64)
    if probabilities.shape[0] != labels.shape[0]:
        raise RuntimeError(
            f"The model returned {probabilities.shape[0]} predictions for {labels.shape[0]} images."
        )
    return probabilities, labels


def sparse_categorical_crossentropy(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Mean negative log likelihood of the true class. Matches the training loss."""
    clipped = np.clip(probabilities, 1e-7, 1.0)
    chosen = clipped[np.arange(labels.shape[0]), labels]
    return float(-np.mean(np.log(chosen)))
