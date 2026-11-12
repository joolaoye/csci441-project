"""Evaluate a saved ``.keras`` checkpoint without training it."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
from tensorflow import keras

from traffic_signs.data.dataset import ensure_files_exist, make_sequence, predict_probabilities, sparse_categorical_crossentropy
from traffic_signs.data.labels import class_name_for_model
from traffic_signs.data.splits import DatasetError, load_manifest
from traffic_signs.evaluation.conditions import (
    builtin_clean_condition,
    condition_fingerprint,
    condition_label,
    condition_notes,
    load_conditions,
)
from traffic_signs.evaluation.metrics import evaluate_predictions, write_predictions_csv
from traffic_signs.utils.config import load_config, override_data_root
from traffic_signs.utils.runs import (
    configure_logging,
    file_sha256,
    framework_versions,
    read_json,
    relative_to_project,
    resolve_path,
    utc_now,
    utc_stamp,
    write_json,
)

logger = logging.getLogger(__name__)
EVALUATION_SPLITS = ("validation", "test")


def requested_condition_names(condition: str | None, conditions: list[str] | None) -> list[str]:
    """Return the conditions to score, preserving the order they were given.

    ``--conditions`` scores several transforms in one command. A repeated name
    is scored once. ``--condition`` remains the single-condition form and
    defaults to clean when neither flag is set.
    """
    if conditions and condition:
        raise ValueError("Pass --condition or --conditions, not both.")
    chosen = list(conditions) if conditions else [condition or "clean"]
    names = list(dict.fromkeys(chosen))
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("Pass at least one evaluation condition.")
    return names


def evaluate_checkpoint(
    checkpoint: Path,
    split: str,
    *,
    project_dir: Path | None = None,
    data_root: str | None = None,
    condition: str = "clean",
    conditions_path: Path | None = None,
) -> dict:
    """Score one saved model on validation or test data and write a new JSON file.

    The training ``metrics.json`` is not modified. The model is not trained.
    Each call writes a new evaluation file, including two evaluations that share
    a condition name but use different parameters. The dataset TEST partition is
    available for the final report; the log warns when it is used.
    """
    results = evaluate_conditions(
        checkpoint,
        split,
        conditions=[condition],
        project_dir=project_dir,
        data_root=data_root,
        conditions_path=conditions_path,
    )
    return results[0]


def evaluate_conditions(
    checkpoint: Path,
    split: str,
    conditions: list[str],
    *,
    project_dir: Path | None = None,
    data_root: str | None = None,
    conditions_path: Path | None = None,
) -> list[dict]:
    """Score one checkpoint on each named condition and write one file per condition.

    The checkpoint is loaded once. Every condition uses that same checkpoint,
    the same split, and the same image ids. Altered conditions do not choose
    a new checkpoint.
    """
    if split not in EVALUATION_SPLITS:
        raise ValueError(
            "split must be 'validation' or 'test'. "
            "The training partition is not an evaluation split."
        )
    if not conditions:
        raise ValueError("Pass at least one evaluation condition.")
    names = requested_condition_names(None, conditions)
    project_dir = Path(project_dir or Path.cwd())
    checkpoint = Path(checkpoint)
    if checkpoint.suffix != ".keras":
        raise ValueError(f"The checkpoint must be a .keras file. Got {checkpoint.name}.")
    resolved_conditions = [_resolve_condition(name, conditions_path) for name in names]
    run_dir, config, metadata = load_run(checkpoint)
    if data_root is not None:
        override_data_root(config, data_root)
    stamp = utc_stamp()
    log_path = configure_logging("evaluate", f"{stamp}_{split}", project_dir)
    started = utc_now()
    try:
        if split == "test":
            logger.warning(
                "Evaluating the held-out TEST partition. Reserve it for the final report "
                "rather than using it to choose augmentation or depth."
            )
        manifest_path = resolve_path(config["data"]["split_manifest"], project_dir)
        manifest = load_manifest(manifest_path)
        current_hash = file_sha256(manifest_path)
        trained_hash = metadata.get("split_manifest_sha256")
        if trained_hash and trained_hash != current_hash:
            logger.warning(
                "The split manifest hash changed since training. "
                "Training recorded %s and the file now hashes to %s.",
                trained_hash,
                current_hash,
            )
        if manifest["num_classes"] != config["data"]["num_classes"]:
            raise DatasetError(
                "The manifest num_classes does not match the training configuration "
                f"({manifest['num_classes']} vs {config['data']['num_classes']})."
            )
        samples = manifest["samples"][split]
        if not samples:
            raise DatasetError(f"The {split} split in {manifest_path} is empty.")
        image_size = int(config["data"]["image_size"])
        sample_hash = _sample_ids_sha256([sample["id"] for sample in samples])
        resolved_root = resolve_path(config["data"]["root"], project_dir)
        ensure_files_exist(resolved_root, samples)
        versions = framework_versions()
        logger.info("source_run_id=%s experiment=%s", metadata.get("run_id"), metadata.get("experiment"))
        logger.info("checkpoint=%s", checkpoint)
        logger.info("split=%s images=%d conditions=%s", split, len(samples), ", ".join(names))
        logger.info("started_at=%s", started)
        logger.info(
            "tensorflow=%s keras=%s",
            versions["tensorflow_version"],
            versions["keras_version"],
        )
        logger.info("split_manifest=%s sha256=%s", manifest_path, current_hash)
        logger.info("training_seed=%s split_seed=%s", config["seed"], manifest.get("seed"))
        model = keras.models.load_model(checkpoint)
        _check_model_shapes(model, image_size, config["data"]["num_classes"])
        written = []
        for resolved in resolved_conditions:
            fingerprint = condition_fingerprint(
                resolved,
                image_size=image_size,
                split=split,
                split_manifest_sha256=current_hash,
                sample_ids_sha256=sample_hash,
            )
            logger.info(
                "condition=%s parameters=%s",
                condition_label(resolved, image_size),
                resolved["parameters"],
            )
            results = score_split(
                checkpoint,
                samples,
                data_root=resolved_root,
                image_size=image_size,
                num_classes=config["data"]["num_classes"],
                batch_size=config["training"]["batch_size"],
                condition=resolved,
                model=model,
            )
            probabilities = results.pop("probabilities")
            payload = {
                "source": "evaluation",
                "created_at": started,
                "run_id": metadata.get("run_id"),
                "experiment": metadata.get("experiment"),
                "checkpoint": relative_to_project(checkpoint, project_dir),
                "split": split,
                "split_manifest": config["data"]["split_manifest"],
                "split_manifest_sha256": current_hash,
                "training_split_manifest_sha256": trained_hash,
                "sample_ids_sha256": sample_hash,
                "image_size": image_size,
                "condition": {
                    **resolved,
                    "label": condition_label(resolved, image_size),
                    "notes": condition_notes(resolved),
                    "fingerprint": fingerprint,
                },
                **results,
            }
            output_dir = run_dir / "evaluations"
            output_dir.mkdir(parents=True, exist_ok=True)
            stem = f"{stamp}_{split}_{resolved['name']}_{fingerprint}"
            output_path = _unique_path(output_dir, stem, ".json")
            payload["evaluation_id"] = output_path.stem
            write_json(output_path, payload)
            write_predictions_csv(
                output_dir / f"{output_path.stem}_predictions.csv",
                samples,
                probabilities,
            )
            logger.info(
                "condition=%s accuracy=%.4f macro_f1=%.4f loss=%.4f samples=%d",
                resolved["name"],
                payload["accuracy"],
                payload["macro_f1"],
                payload["loss"],
                payload["num_samples"],
            )
            logger.info("wrote %s", output_path)
            written.append(
                {
                    "results_path": output_path,
                    "log_path": log_path,
                    "metrics": payload,
                    "run_id": metadata.get("run_id"),
                }
            )
        return written
    except Exception:
        logger.exception("Evaluation failed")
        raise


def score_split(
    checkpoint: Path,
    samples: list[dict],
    *,
    data_root: Path,
    image_size: int,
    num_classes: int,
    batch_size: int,
    condition: dict | None = None,
    model: keras.Model | None = None,
) -> dict:
    """Load a checkpoint and score it. This does not write training artifacts.

    Pass ``model`` to reuse a checkpoint that has already been loaded. The
    condition changes the images, not which checkpoint is scored.
    """
    if model is None:
        model = keras.models.load_model(checkpoint)
        _check_model_shapes(model, image_size, num_classes)
    sequence = make_sequence(
        samples,
        root=data_root,
        image_size=image_size,
        batch_size=batch_size,
        training=False,
        augmentation=None,
        seed=0,
        condition=condition,
    )
    probabilities, labels = predict_probabilities(model, sequence)
    predictions = np.argmax(probabilities, axis=-1)
    results = evaluate_predictions(labels, predictions, num_classes)
    results["loss"] = sparse_categorical_crossentropy(probabilities, labels)
    results["probabilities"] = probabilities
    for row in results["per_class"]:
        name = class_name_for_model(row["class_id"], num_classes)
        if name is not None:
            row["name"] = name
    return results


def _resolve_condition(name: str, conditions_path: Path | None) -> dict:
    if conditions_path is None:
        if name != "clean":
            raise ValueError(
                "Pass the evaluation conditions file when selecting a condition other than clean."
            )
        return builtin_clean_condition()
    conditions = load_conditions(conditions_path)
    if name not in conditions:
        available = ", ".join(sorted(conditions))
        raise ValueError(f"Unknown condition {name!r}. Available conditions: {available}.")
    return conditions[name]


def _sample_ids_sha256(sample_ids: list[str]) -> str:
    digest = hashlib.sha256()
    digest.update("\n".join(sample_ids).encode("utf-8"))
    return digest.hexdigest()


def _unique_path(directory: Path, stem: str, suffix: str) -> Path:
    path = directory / f"{stem}{suffix}"
    counter = 2
    while path.exists():
        path = directory / f"{stem}_{counter}{suffix}"
        counter += 1
    return path


def load_run(checkpoint: Path) -> tuple[Path, dict, dict]:
    """Load the configuration and metadata stored beside a training checkpoint."""
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    run_dir = checkpoint.parent
    config_path = run_dir / "config.yaml"
    metadata_path = run_dir / "metadata.json"
    if not config_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            f"Could not find config.yaml and metadata.json next to {checkpoint}. "
            "Evaluate a model.keras file written by scripts/train.py."
        )
    config = load_config(config_path)
    try:
        metadata = read_json(metadata_path)
    except ValueError as exc:
        raise FileNotFoundError(f"Could not read {metadata_path}: {exc}") from exc
    return run_dir, config, metadata


def _check_model_shapes(model: keras.Model, image_size: int, num_classes: int) -> None:
    input_shape = tuple(model.input_shape)
    output_shape = tuple(model.output_shape)
    if input_shape[-3:] != (image_size, image_size, 3):
        raise ValueError(
            f"The checkpoint expects input {input_shape}, but the training configuration "
            f"uses {image_size}x{image_size} RGB images."
        )
    if output_shape[-1] != num_classes:
        raise ValueError(
            f"The checkpoint has {output_shape[-1]} outputs, but the configuration "
            f"has {num_classes} classes."
        )
