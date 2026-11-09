"""Load and validate experiment YAML.

Experimental fields have no hidden defaults: a missing learning rate or
augmentation block is an error. A few architectural fields that are constant
across the supplied experiments are filled in when omitted, and the resolved
configuration written to the run directory always includes those values.
"""

from __future__ import annotations

from pathlib import Path

import yaml

ALLOWED_DEVICES = ("auto", "cpu", "gpu")
ALLOWED_OPTIMIZERS = ("adam",)
MAX_CONV_LAYERS = 8
DEFAULT_NUM_CLASSES = 43
DEFAULT_DENSE_UNITS = 128
DEFAULT_MIN_SPATIAL = 12
DEFAULT_OPTIMIZER = "adam"


class ConfigError(ValueError):
    """The experiment file is missing, unreadable, or inconsistent."""


def load_config(path: Path | str) -> dict:
    """Read a YAML file and return the resolved configuration."""
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"Configuration file not found: {config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Could not parse {config_path}: {exc}") from exc
    if raw is None:
        raise ConfigError(f"{config_path} is empty.")
    if not isinstance(raw, dict):
        raise ConfigError(f"{config_path} must contain a YAML mapping.")
    try:
        return normalize_config(raw)
    except ConfigError as exc:
        raise ConfigError(f"{config_path}: {exc}") from exc


def normalize_config(raw: dict) -> dict:
    """Validate a parsed configuration and fill documented architectural defaults."""
    if not isinstance(raw, dict):
        raise ConfigError("The configuration must be a mapping.")
    _reject_unknown(
        raw,
        ("experiment", "seed", "device", "data", "model", "training", "augmentation"),
        "top-level",
    )
    experiment = _require_experiment(_require(raw, "experiment"))
    seed = _require_int(_require(raw, "seed"), "seed", minimum=0, maximum=2**31 - 1)
    device = _require_choice(_require(raw, "device"), "device", ALLOWED_DEVICES)
    data = _normalize_data(_require_mapping(_require(raw, "data"), "data"))
    model = _normalize_model(_require_mapping(_require(raw, "model"), "model"), data["image_size"])
    training = _normalize_training(_require_mapping(_require(raw, "training"), "training"))
    augmentation = _normalize_augmentation(
        _require_mapping(_require(raw, "augmentation"), "augmentation")
    )
    return {
        "experiment": experiment,
        "seed": seed,
        "device": device,
        "data": data,
        "model": model,
        "training": training,
        "augmentation": augmentation,
    }


def _normalize_data(raw: dict) -> dict:
    _reject_unknown(raw, ("root", "split_manifest", "image_size", "num_classes"), "data")
    manifest = _require_relative_string(_require(raw, "split_manifest"), "data.split_manifest")
    image_size = _require_int(
        _require(raw, "image_size"),
        "data.image_size",
        minimum=16,
        maximum=128,
    )
    num_classes = _require_int(
        raw.get("num_classes", DEFAULT_NUM_CLASSES),
        "data.num_classes",
        minimum=2,
        maximum=43,
    )
    data = {
        "split_manifest": manifest,
        "image_size": image_size,
        "num_classes": num_classes,
    }
    if "root" in raw:
        data["root"] = _require_data_root(raw["root"])
    return data


def _normalize_model(raw: dict, image_size: int) -> dict:
    _reject_unknown(
        raw,
        ("channels", "kernel_size", "dropout", "dense_units", "min_spatial"),
        "model",
    )
    channels = _require_channels(_require(raw, "channels"))
    kernel_size = _require_int(
        _require(raw, "kernel_size"),
        "model.kernel_size",
        minimum=1,
        maximum=7,
    )
    if kernel_size % 2 == 0:
        raise ConfigError(
            "model.kernel_size must be odd so same-padding keeps the feature map centered. "
            f"Got {kernel_size}."
        )
    dropout = _require_number(
        _require(raw, "dropout"),
        "model.dropout",
        minimum=0.0,
        maximum=0.5,
    )
    dense_units = _require_int(
        raw.get("dense_units", DEFAULT_DENSE_UNITS),
        "model.dense_units",
        minimum=1,
        maximum=1024,
    )
    min_spatial = _require_int(
        raw.get("min_spatial", DEFAULT_MIN_SPATIAL),
        "model.min_spatial",
        minimum=8,
        maximum=image_size,
    )
    if image_size // 2 < min_spatial:
        raise ConfigError(
            "model.min_spatial leaves no room to pool. "
            f"image_size {image_size} with min_spatial {min_spatial} would keep the "
            "original resolution. Lower min_spatial or increase data.image_size."
        )
    return {
        "channels": channels,
        "kernel_size": kernel_size,
        "dropout": dropout,
        "dense_units": dense_units,
        "min_spatial": min_spatial,
    }


def _normalize_training(raw: dict) -> dict:
    _reject_unknown(
        raw,
        ("epochs", "batch_size", "learning_rate", "optimizer"),
        "training",
    )
    return {
        "epochs": _require_int(_require(raw, "epochs"), "training.epochs", minimum=1, maximum=200),
        "batch_size": _require_int(
            _require(raw, "batch_size"),
            "training.batch_size",
            minimum=1,
            maximum=512,
        ),
        "learning_rate": _require_number(
            _require(raw, "learning_rate"),
            "training.learning_rate",
            minimum=1e-6,
            maximum=0.1,
        ),
        "optimizer": _require_choice(
            raw.get("optimizer", DEFAULT_OPTIMIZER),
            "training.optimizer",
            ALLOWED_OPTIMIZERS,
        ),
    }


def _normalize_augmentation(raw: dict) -> dict:
    _reject_unknown(raw, ("rotation_degrees", "brightness", "zoom"), "augmentation")
    rotation = _require_number(
        _require(raw, "rotation_degrees"),
        "augmentation.rotation_degrees",
        minimum=0.0,
        maximum=20.0,
    )
    brightness = _require_number(
        _require(raw, "brightness"),
        "augmentation.brightness",
        minimum=0.0,
        maximum=0.2,
    )
    zoom = _require_number(
        _require(raw, "zoom"),
        "augmentation.zoom",
        minimum=0.0,
        maximum=0.2,
    )
    return {
        "rotation_degrees": rotation,
        "brightness": brightness,
        "zoom": zoom,
    }


def _require(mapping: dict, key: str):
    if key not in mapping:
        raise ConfigError(f"Missing required setting '{key}'.")
    return mapping[key]


def _require_mapping(value, name: str) -> dict:
    if not isinstance(value, dict):
        raise ConfigError(f"{name} must be a mapping.")
    return value


def _reject_unknown(mapping: dict, allowed: tuple[str, ...], name: str) -> None:
    unknown = sorted(set(mapping) - set(allowed))
    if unknown:
        raise ConfigError(
            f"Unknown {name} setting(s): {', '.join(unknown)}. "
            f"Allowed settings: {', '.join(allowed)}."
        )


def _require_experiment(value, name: str = "experiment") -> str:
    if not isinstance(value, str):
        raise ConfigError(f"{name} must be a string.")
    if not value or not all(character.isalnum() or character in "-_" for character in value):
        raise ConfigError(
            f"{name} must contain only letters, numbers, hyphens, and underscores, "
            "so it can be used in a run directory name."
        )
    if len(value) > 40:
        raise ConfigError(f"{name} must be at most 40 characters.")
    return value


def _require_choice(value, name: str, allowed: tuple[str, ...]) -> str:
    if not isinstance(value, str) or value not in allowed:
        choices = ", ".join(repr(item) for item in allowed)
        raise ConfigError(f"{name} must be one of {choices}. Got {value!r}.")
    return value


def _require_int(value, name: str, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{name} must be an integer. Got {value!r}.")
    if value < minimum or value > maximum:
        raise ConfigError(f"{name} must be between {minimum} and {maximum}. Got {value}.")
    return value


def _require_number(value, name: str, *, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be a number. Got {value!r}.")
    number = float(value)
    if number < minimum or number > maximum:
        raise ConfigError(f"{name} must be between {minimum} and {maximum}. Got {number}.")
    return number


def _require_channels(value) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ConfigError(
            "model.channels must be a non-empty list of filter counts, one per "
            "convolutional layer. Example: [32, 64]."
        )
    if len(value) > MAX_CONV_LAYERS:
        raise ConfigError(
            f"model.channels can contain at most {MAX_CONV_LAYERS} layers. Got {len(value)}."
        )
    channels = []
    for index, filters in enumerate(value, start=1):
        parsed = _require_int(filters, f"model.channels[{index}]", minimum=1, maximum=512)
        channels.append(parsed)
    return channels


def _require_relative_string(value, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{name} must be a non-empty path string.")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ConfigError(
            f"{name} must be a relative path without '..', so the configuration "
            f"stays portable across machines. Got {value!r}."
        )
    return path.as_posix()


def override_data_root(config: dict, data_root: str) -> None:
    """Point this loaded configuration at a dataset directory.

    The YAML file on disk is left unchanged. Training stores the replacement
    in the run's ``config.yaml``.
    """
    data = config.get("data")
    if not isinstance(data, dict):
        raise ConfigError("The configuration is missing the data section.")
    data["root"] = _require_data_root(data_root)


def _require_data_root(value) -> str:
    """Accept a relative dataset path, ``~``, or an absolute local path."""
    if not isinstance(value, str) or not value.strip():
        raise ConfigError("data.root must be a non-empty path string.")
    expanded = Path(value).expanduser()
    if ".." in Path(value).parts or ".." in expanded.parts:
        raise ConfigError(f"data.root must not contain '..'. Got {value!r}.")
    if expanded.is_absolute():
        return value
    return Path(value).as_posix()
