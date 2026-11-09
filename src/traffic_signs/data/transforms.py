"""Deterministic preprocessing and optional training augmentation.

Validation and test images are only converted to RGB, resized, and scaled to
``[0, 1]``. Rotation, brightness, and zoom are applied later, and only to
training batches. A setting of 0 disables that transform.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers

from traffic_signs.data.splits import DatasetError


def load_rgb(path: Path) -> np.ndarray:
    """Open an image file and return a uint8 RGB array."""
    path = Path(path)
    try:
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            array = np.asarray(rgb, dtype=np.uint8)
    except (OSError, ValueError) as exc:
        raise DatasetError(f"Could not read image {path}: {exc}") from exc
    if array.ndim != 3 or array.shape[2] != 3 or array.size == 0:
        raise DatasetError(f"{path} did not produce an RGB image. Got shape {array.shape}.")
    return array


def preprocess_array(image: np.ndarray, image_size: int) -> np.ndarray:
    """Resize a uint8 RGB image and scale pixels into ``[0, 1]``."""
    if image_size < 1:
        raise ValueError("image_size must be positive.")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected an RGB image, got shape {image.shape}.")
    if image.dtype != np.uint8:
        raise ValueError("preprocess_array expects a uint8 RGB image.")
    pixels = tf.cast(tf.convert_to_tensor(image), tf.float32)
    resized = tf.image.resize(
        pixels,
        (image_size, image_size),
        method="bilinear",
        antialias=True,
    )
    normalized = tf.clip_by_value(resized / 255.0, 0.0, 1.0)
    array = normalized.numpy().astype(np.float32)
    if array.shape != (image_size, image_size, 3):
        raise RuntimeError(f"Preprocessing produced shape {array.shape}.")
    return array


def build_augmentor(augmentation: dict, seed: int) -> keras.Sequential | None:
    """Build a Keras augmentation stack, or return None when every factor is 0.

    Rotation is ±``rotation_degrees``. Brightness adds at most ±``brightness``
    to pixels already scaled to ``[0, 1]``. Zoom draws one scale from
    ``[1 - zoom, 1 + zoom]`` and uses it on both axes. Outputs are clipped
    back to ``[0, 1]`` by :func:`apply_augmentation`.
    """
    stack = []
    rotation = float(augmentation["rotation_degrees"])
    brightness = float(augmentation["brightness"])
    zoom = float(augmentation["zoom"])
    if rotation:
        stack.append(
            layers.RandomRotation(
                factor=rotation / 360.0,
                fill_mode="nearest",
                seed=seed,
                name="rotation",
            )
        )
    if brightness:
        stack.append(_brightness_layer(brightness, seed))
    if zoom:
        # A single float is the symmetric range [1 - zoom, 1 + zoom].
        # Width is left unset so that scale is reused and the sign is not stretched.
        stack.append(
            layers.RandomZoom(
                height_factor=zoom,
                fill_mode="nearest",
                seed=seed,
                name="zoom",
            )
        )
    if not stack:
        return None
    return keras.Sequential(stack, name="training_augmentation")


def apply_augmentation(
    images: np.ndarray,
    augmentor: keras.Sequential | None,
    training: bool,
) -> np.ndarray:
    """Augment a training batch. Validation and test batches are returned unchanged."""
    if not training or augmentor is None:
        return images
    augmented = augmentor(images, training=True)
    augmented = tf.clip_by_value(augmented, 0.0, 1.0)
    return augmented.numpy().astype(np.float32)


def _brightness_layer(brightness: float, seed: int) -> keras.layers.Layer:
    """Construct RandomBrightness for images scaled to ``[0, 1]``."""
    signature = inspect.signature(layers.RandomBrightness.__init__)
    accepts_keywords = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    kwargs = {"factor": brightness, "seed": seed, "name": "brightness"}
    if "value_range" in signature.parameters or accepts_keywords:
        kwargs["value_range"] = (0.0, 1.0)
    try:
        return layers.RandomBrightness(**kwargs)
    except TypeError:
        kwargs.pop("value_range", None)
        return layers.RandomBrightness(**kwargs)
