"""Deterministic visual conditions for evaluation and their previews.

Training augmentation is a separate setting. These transforms are applied
after the standard preprocessing and do not change labels or image dimensions.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml

CONDITION_NAME = re.compile(r"^[a-z][a-z-]*$")
CONDITION_TYPES = ("none", "rotation", "brightness", "downsample")


class ConditionError(ValueError):
    """An evaluation condition file is missing or inconsistent."""


def default_conditions_path(project_dir: Path) -> Path:
    """Use a local ``evaluation.yaml`` when it exists, otherwise the example."""
    project_dir = Path(project_dir)
    local = project_dir / "evaluation.yaml"
    if local.is_file():
        return local
    return project_dir / "evaluation.yaml.example"


def load_conditions(path: Path | str) -> dict[str, dict]:
    """Read an evaluation conditions file and return conditions keyed by name."""
    config_path = Path(path)
    if not config_path.is_file():
        raise ConditionError(f"Evaluation conditions file not found: {config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConditionError(f"Could not parse {config_path}: {exc}") from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("conditions"), dict):
        raise ConditionError(f"{config_path} must contain a conditions mapping.")
    unknown = sorted(set(raw) - {"conditions"})
    if unknown:
        raise ConditionError(f"Unknown evaluation setting(s): {', '.join(unknown)}.")
    conditions = {}
    for name, body in raw["conditions"].items():
        conditions[name] = normalize_condition(name, body)
    if "clean" not in conditions:
        raise ConditionError("Evaluation conditions must include clean.")
    return conditions


def builtin_clean_condition() -> dict:
    """The identity condition used when no conditions file is supplied."""
    return normalize_condition("clean", {"type": "none"})


def normalize_condition(name, body) -> dict:
    """Validate one condition and return its resolved parameters."""
    if not isinstance(name, str) or CONDITION_NAME.fullmatch(name) is None:
        raise ConditionError(
            f"Condition name {name!r} must be lowercase words, without digits. "
            "Put the rotation angle or brightness delta in the condition values."
        )
    if not isinstance(body, dict):
        raise ConditionError(f"Condition {name!r} must be a mapping.")
    condition_type = body.get("type")
    if condition_type not in CONDITION_TYPES:
        raise ConditionError(
            f"Condition {name!r} type must be one of {', '.join(CONDITION_TYPES)}. "
            f"Got {condition_type!r}."
        )
    allowed = {"type", *_parameter_keys(condition_type)}
    unknown = sorted(set(body) - allowed)
    if unknown:
        raise ConditionError(
            f"Condition {name!r} has unknown setting(s): {', '.join(unknown)}."
        )
    parameters = _parameters(name, condition_type, body)
    return {"name": name, "type": condition_type, "parameters": parameters}


def condition_label(condition: dict, image_size: int | None = None) -> str:
    """Short label that includes the configured parameter values."""
    parameters = condition["parameters"]
    if condition["type"] == "none":
        return "Clean"
    if condition["type"] == "rotation":
        return f"Rotation: {parameters['degrees']:+g}°"
    if condition["type"] == "brightness":
        return f"Brightness: {parameters['delta']:+g}"
    low = int(parameters["image_size"])
    if image_size is None:
        return f"Low resolution: {low}×{low}"
    return f"Low resolution: {low}×{low} → {int(image_size)}×{int(image_size)}"


def condition_notes(condition: dict) -> str:
    """Describe the interpolation and fill behavior for this condition."""
    if condition["type"] == "none":
        return "Standard preprocessing only: RGB, bilinear resize, scale to [0, 1]."
    if condition["type"] == "rotation":
        return (
            "Counterclockwise rotation of the preprocessed image. "
            "Bilinear sampling. Corners repeat the nearest border pixel."
        )
    if condition["type"] == "brightness":
        return "Adds delta to normalized pixel intensities and clips to [0, 1]."
    return (
        "Bilinear downsample with antialiasing, then bilinear resize back to "
        "the model input, also with antialiasing."
    )


def condition_fingerprint(condition: dict, **fields) -> str:
    """Hash the condition and any extra fields that must invalidate a result."""
    payload = {"condition": _canonical_condition(condition), **fields}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def apply_condition(image: np.ndarray, condition: dict, image_size: int) -> np.ndarray:
    """Transform one preprocessed HWC image. Shape and label stay unchanged."""
    batch = apply_condition_batch(image[None, ...], condition, image_size)
    return batch[0]


def apply_condition_batch(images: np.ndarray, condition: dict, image_size: int) -> np.ndarray:
    """Transform an NHWC batch. A ``none`` condition returns the batch unchanged."""
    values = np.asarray(images, dtype=np.float32)
    if values.ndim != 4 or values.shape[-1] != 3:
        raise ConditionError(f"Expected an NHWC RGB batch. Got shape {values.shape}.")
    if values.shape[1] != image_size or values.shape[2] != image_size:
        raise ConditionError(
            f"Condition input is {values.shape[1]}×{values.shape[2]}, "
            f"but the model image size is {image_size}."
        )
    condition_type = condition["type"]
    parameters = condition["parameters"]
    if condition_type == "none":
        transformed = values
    elif condition_type == "rotation":
        transformed = _rotate(values, float(parameters["degrees"]))
    elif condition_type == "brightness":
        transformed = values + np.float32(parameters["delta"])
    elif condition_type == "downsample":
        transformed = _downsample(values, int(parameters["image_size"]), image_size)
    else:
        raise ConditionError(f"Unknown condition type {condition_type!r}.")
    transformed = np.clip(transformed, 0.0, 1.0).astype(np.float32)
    if transformed.shape != values.shape:
        raise ConditionError(
            f"Condition {condition['name']!r} changed the image shape from "
            f"{values.shape} to {transformed.shape}."
        )
    return transformed


def _canonical_condition(condition: dict) -> dict:
    return {
        "name": condition["name"],
        "type": condition["type"],
        "parameters": {key: condition["parameters"][key] for key in sorted(condition["parameters"])},
    }


def _parameter_keys(condition_type: str) -> set[str]:
    if condition_type == "rotation":
        return {"degrees"}
    if condition_type == "brightness":
        return {"delta"}
    if condition_type == "downsample":
        return {"image_size"}
    return set()


def _parameters(name: str, condition_type: str, body: dict) -> dict:
    if condition_type == "none":
        return {}
    if condition_type == "rotation":
        return {"degrees": _number(body, "degrees", name, minimum=-180.0, maximum=180.0)}
    if condition_type == "brightness":
        return {"delta": _number(body, "delta", name, minimum=-1.0, maximum=1.0)}
    image_size = body.get("image_size")
    if isinstance(image_size, bool) or not isinstance(image_size, int):
        raise ConditionError(f"Condition {name!r} image_size must be an integer.")
    if image_size < 8 or image_size > 128:
        raise ConditionError(f"Condition {name!r} image_size must be from 8 to 128. Got {image_size}.")
    return {"image_size": image_size}


def _number(body: dict, key: str, name: str, *, minimum: float, maximum: float) -> float:
    value = body.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConditionError(f"Condition {name!r} {key} must be a number. Got {value!r}.")
    number = float(value)
    if number < minimum or number > maximum:
        raise ConditionError(
            f"Condition {name!r} {key} must be between {minimum:g} and {maximum:g}. Got {number:g}."
        )
    return number


def _rotate(images: np.ndarray, degrees: float) -> np.ndarray:
    """Rotate counterclockwise. Bilinear sampling, nearest-border fill."""
    if degrees == 0:
        return images
    theta = float(np.deg2rad(degrees))
    height = int(images.shape[1])
    width = int(images.shape[2])
    center_x = (width - 1) / 2.0
    center_y = (height - 1) / 2.0
    cosine = float(np.cos(theta))
    sine = float(np.sin(theta))
    transform = np.array(
        [
            cosine,
            sine,
            center_x - cosine * center_x - sine * center_y,
            -sine,
            cosine,
            center_y + sine * center_x - cosine * center_y,
            0.0,
            0.0,
        ],
        dtype=np.float32,
    )
    transforms = np.tile(transform, (images.shape[0], 1))
    rotated = tf.raw_ops.ImageProjectiveTransformV3(
        images=tf.convert_to_tensor(images, dtype=tf.float32),
        transforms=tf.convert_to_tensor(transforms),
        output_shape=tf.constant([height, width], dtype=tf.int32),
        fill_value=tf.constant(0.0, dtype=tf.float32),
        interpolation="BILINEAR",
        fill_mode="NEAREST",
    )
    return rotated.numpy().astype(np.float32)


def _downsample(images: np.ndarray, low_size: int, image_size: int) -> np.ndarray:
    """Shrink with antialiased bilinear sampling, then restore the model size."""
    if low_size >= image_size:
        raise ConditionError(
            f"Downsample size {low_size} must be smaller than the model input {image_size}."
        )
    small = tf.image.resize(images, (low_size, low_size), method="bilinear", antialias=True)
    restored = tf.image.resize(small, (image_size, image_size), method="bilinear", antialias=True)
    return restored.numpy().astype(np.float32)
