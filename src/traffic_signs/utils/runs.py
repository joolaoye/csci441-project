"""Run identifiers, logs, device selection, and artifact IO."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml


class DeviceError(RuntimeError):
    """The requested compute device is not available."""


def utc_now() -> str:
    """Return the current UTC time as an ISO-8601 string without microseconds."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def utc_stamp() -> str:
    """Return a UTC timestamp safe to use in file names."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def configure_logging(operation: str, name: str, project_dir: Path | None = None) -> Path:
    """Send ``traffic_signs`` logs to stderr and ``logs/<operation>/<name>.log``."""
    root = Path(project_dir or Path.cwd())
    log_path = root / "logs" / operation / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("traffic_signs")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    stream_handler = logging.StreamHandler()
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return log_path


def file_sha256(path: Path) -> str:
    """Hash file bytes with SHA-256."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    """Atomically write JSON with a trailing newline."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_yaml(path: Path, payload: dict) -> None:
    """Write a resolved configuration without reordering keys."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def read_json(path: Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def resolve_path(path: str | Path, project_dir: Path) -> Path:
    """Resolve a config path against the project directory."""
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return Path(project_dir) / candidate


def relative_to_project(path: Path, project_dir: Path) -> str:
    """Return a forward-slash path relative to the project when possible."""
    try:
        return Path(path).resolve().relative_to(Path(project_dir).resolve()).as_posix()
    except ValueError:
        return str(path)


def git_info() -> dict:
    """Return the current commit and whether the work tree is dirty.

    The search starts at the current directory and at this source file, so a
    run records the code version even when artifacts are written elsewhere.
    """
    root = _find_git_root()
    if root is None:
        return {"commit": None, "dirty": None}
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": None, "dirty": None}
    return {"commit": commit or None, "dirty": bool(status.strip())}


def framework_versions() -> dict:
    """Return library versions used for a training or evaluation record."""
    import numpy
    import PIL
    import tensorflow as tf

    return {
        "python_version": platform.python_version(),
        "tensorflow_version": tf.__version__,
        "keras_version": tf.keras.__version__,
        "numpy_version": numpy.__version__,
        "pillow_version": PIL.__version__,
    }


def configure_device(choice: str) -> dict:
    """Select CPU or GPU and return a description for the run metadata.

    ``auto`` uses a GPU when TensorFlow lists one, including Apple Silicon
    when the installed TensorFlow build exposes it, and otherwise uses the CPU.
    ``gpu`` fails when no GPU is visible so a run does not silently change devices.
    """
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if choice == "cpu":
        _hide_gpus(tf)
        selected = "cpu"
        name = _cpu_name(tf)
    elif choice == "gpu":
        if not gpus:
            raise DeviceError(
                "device is 'gpu' but TensorFlow found no GPU. Training was not "
                "started, so this run did not fall back to CPU. Check devices with "
                "`uv run python -c \"import tensorflow as tf; "
                "print(tf.config.list_physical_devices())\"`. On Apple Silicon, GPU "
                "acceleration requires a TensorFlow build with Metal support. "
                "Use device: auto or device: cpu to train on the CPU."
            )
        _enable_memory_growth(tf, gpus)
        selected = "gpu"
        name = ", ".join(device.name for device in gpus)
    elif choice == "auto":
        if gpus:
            _enable_memory_growth(tf, gpus)
            selected = "gpu"
            name = ", ".join(device.name for device in gpus)
        else:
            selected = "cpu"
            name = _cpu_name(tf)
    else:
        raise DeviceError(f"Unknown device {choice!r}. Expected auto, cpu, or gpu.")
    return {
        "requested": choice,
        "selected": selected,
        "name": name,
        "gpu_devices": [device.name for device in gpus],
    }


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and TensorFlow.

    ``PYTHONHASHSEED`` only takes effect when it is set before the interpreter
    starts. The split manifest, not this seed, decides which images are in
    training and validation.
    """
    import random

    import numpy as np
    import tensorflow as tf

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def _hide_gpus(tf) -> None:
    try:
        tf.config.set_visible_devices([], "GPU")
    except RuntimeError as exc:
        if tf.config.get_visible_devices("GPU"):
            raise DeviceError(
                "Could not force CPU execution because TensorFlow has already "
                "initialized a GPU in this process."
            ) from exc


def _enable_memory_growth(tf, gpus) -> None:
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            # Metal devices and already-initialized GPUs may reject this.
            continue


def _cpu_name(tf) -> str:
    cpus = tf.config.list_physical_devices("CPU")
    if not cpus:
        return "cpu"
    return ", ".join(device.name for device in cpus)


def _find_git_root() -> Path | None:
    starts = [Path.cwd(), Path(__file__).resolve()]
    seen: set[Path] = set()
    for start in starts:
        for candidate in [start, *start.parents]:
            if candidate in seen:
                continue
            seen.add(candidate)
            if (candidate / ".git").exists():
                return candidate
    return None
