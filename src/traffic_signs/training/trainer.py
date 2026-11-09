"""Train one configured CNN and write an inspectable run directory."""

from __future__ import annotations

import csv
import logging
import shutil
import time
from pathlib import Path

from tensorflow import keras

from traffic_signs.data.dataset import ensure_files_exist, make_sequence
from traffic_signs.data.splits import DatasetError, load_manifest
from traffic_signs.evaluation.evaluate import score_split
from traffic_signs.evaluation.metrics import write_predictions_csv
from traffic_signs.models.cnn import build_model_from_config, final_spatial_size, parameter_counts, pooling_schedule
from traffic_signs.utils.config import normalize_config
from traffic_signs.utils.runs import (
    configure_device,
    configure_logging,
    file_sha256,
    framework_versions,
    git_info,
    relative_to_project,
    resolve_path,
    set_seed,
    utc_now,
    write_json,
    write_yaml,
)

logger = logging.getLogger(__name__)


def train(config: dict, *, project_dir: Path | None = None) -> Path:
    """Train one run and return its artifact directory.

    The artifact directory is ``artifacts/<experiment>``. Training the same
    experiment again replaces that directory. A failed run still leaves
    ``config.yaml``, ``metadata.json``, the log, and any epoch history written
    before the failure. ``model.keras`` is the best validation-loss checkpoint
    and is only present after at least one epoch improved on that metric.
    """
    project_dir = Path(project_dir or Path.cwd())
    config = normalize_config(config)
    if "root" not in config["data"]:
        raise DatasetError(
            "Pass --data-root with the extracted GTSRB folder. "
            "The example configuration does not choose a dataset directory."
        )
    run_id = config["experiment"]
    run_dir = project_dir / "artifacts" / run_id
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    log_path = configure_logging("train", run_id, project_dir)
    started_at = utc_now()
    started = time.perf_counter()
    metadata = _initial_metadata(
        config,
        run_id=run_id,
        run_dir=run_dir,
        log_path=log_path,
        project_dir=project_dir,
        started_at=started_at,
    )
    write_yaml(run_dir / "config.yaml", config)
    write_json(run_dir / "metadata.json", metadata)
    history_writer = None
    status = "failed"
    error = None
    try:
        device = configure_device(config["device"])
        set_seed(config["seed"])
        versions = framework_versions()
        metadata["device_selected"] = device["selected"]
        metadata["device_name"] = device["name"]
        metadata["gpu_devices"] = device["gpu_devices"]
        metadata.update(versions)
        _log_run_header(config, metadata, device, versions)

        manifest_path = resolve_path(config["data"]["split_manifest"], project_dir)
        manifest = load_manifest(manifest_path)
        if manifest["num_classes"] != config["data"]["num_classes"]:
            raise DatasetError(
                "data.num_classes does not match the split manifest "
                f"({config['data']['num_classes']} vs {manifest['num_classes']})."
            )
        manifest_hash = file_sha256(manifest_path)
        train_samples = manifest["samples"]["train"]
        validation_samples = manifest["samples"]["validation"]
        if not train_samples or not validation_samples:
            raise DatasetError("The training and validation splits must both contain images.")
        data_root = resolve_path(config["data"]["root"], project_dir)
        ensure_files_exist(data_root, train_samples + validation_samples)
        metadata["split_manifest_sha256"] = manifest_hash
        metadata["split_seed"] = manifest["seed"]
        metadata["split_counts"] = {
            "train": len(train_samples),
            "validation": len(validation_samples),
            "test": len(manifest["samples"]["test"]),
        }
        logger.info("split_manifest=%s", manifest_path)
        logger.info("split_manifest_sha256=%s", manifest_hash)
        logger.info("split_seed=%s training_seed=%s", manifest["seed"], config["seed"])
        logger.info("split_counts=%s", metadata["split_counts"])

        schedule = pooling_schedule(
            config["data"]["image_size"],
            len(config["model"]["channels"]),
            config["model"]["min_spatial"],
        )
        spatial = final_spatial_size(
            config["data"]["image_size"],
            len(config["model"]["channels"]),
            config["model"]["min_spatial"],
        )
        metadata["architecture"] = {
            "conv_layers": len(config["model"]["channels"]),
            "channels": list(config["model"]["channels"]),
            "pooling_schedule": schedule,
            "final_spatial_size": spatial,
            "head": "global_average_pooling",
        }
        logger.info("architecture=%s", metadata["architecture"])

        model = build_model_from_config(config)
        counts = parameter_counts(model)
        metadata.update(counts)
        logger.info(
            "trainable_parameters=%d total_parameters=%d",
            counts["trainable_parameters"],
            counts["total_parameters"],
        )
        model.summary(print_fn=lambda line: logger.info("%s", line))
        optimizer_name = config["training"]["optimizer"]
        if optimizer_name != "adam":
            raise DatasetError(f"Unsupported optimizer {optimizer_name!r}.")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=config["training"]["learning_rate"]),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        train_sequence = make_sequence(
            train_samples,
            root=data_root,
            image_size=config["data"]["image_size"],
            batch_size=config["training"]["batch_size"],
            training=True,
            augmentation=config["augmentation"],
            seed=config["seed"],
        )
        validation_sequence = make_sequence(
            validation_samples,
            root=data_root,
            image_size=config["data"]["image_size"],
            batch_size=config["training"]["batch_size"],
            training=False,
            augmentation=None,
            seed=config["seed"],
        )
        checkpoint_path = run_dir / "model.keras"
        history_writer = HistoryWriter(run_dir / "history.csv")
        best_epoch = BestEpochTracker(metadata)
        callbacks = [
            history_writer,
            best_epoch,
            keras.callbacks.ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor="val_loss",
                save_best_only=True,
                mode="min",
                save_weights_only=False,
                verbose=0,
            ),
        ]
        model.fit(
            train_sequence,
            validation_data=validation_sequence,
            epochs=config["training"]["epochs"],
            callbacks=callbacks,
            verbose=0,
        )
        if not checkpoint_path.is_file():
            raise RuntimeError(
                "Training finished but no validation checkpoint was saved. "
                "val_loss was missing or never improved."
            )
        metrics = score_split(
            checkpoint_path,
            validation_samples,
            data_root=data_root,
            image_size=config["data"]["image_size"],
            num_classes=config["data"]["num_classes"],
            batch_size=config["training"]["batch_size"],
        )
        write_predictions_csv(
            run_dir / "predictions.csv",
            validation_samples,
            metrics.pop("probabilities"),
        )
        metrics["source"] = "training_checkpoint"
        metrics["split"] = "validation"
        metrics["checkpoint_epoch"] = metadata["checkpoint_epoch"]
        write_json(run_dir / "metrics.json", metrics)
        metadata["validation_accuracy"] = metrics["accuracy"]
        metadata["validation_macro_f1"] = metrics["macro_f1"]
        status = "completed"
        logger.info(
            "selected epoch=%s val_loss=%s accuracy=%.4f macro_f1=%.4f",
            metadata["checkpoint_epoch"],
            metadata["best_val_loss"],
            metrics["accuracy"],
            metrics["macro_f1"],
        )
    except KeyboardInterrupt:
        status = "interrupted"
        error = "Training was interrupted."
        logger.error(error)
        raise
    except Exception as exc:
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
        logger.exception("Training failed")
        raise
    finally:
        if history_writer is not None:
            history_writer.close()
        metadata["status"] = status
        metadata["error"] = error
        metadata["finished_at"] = utc_now()
        metadata["duration_seconds"] = round(time.perf_counter() - started, 3)
        write_json(run_dir / "metadata.json", metadata)
        logger.info(
            "status=%s duration_seconds=%s run_dir=%s",
            status,
            metadata["duration_seconds"],
            run_dir,
        )
    return run_dir


class HistoryWriter(keras.callbacks.Callback):
    """Append one CSV row per completed epoch and flush it immediately."""

    def __init__(self, path: Path):
        super().__init__()
        self.path = Path(path)
        self._handle = None
        self._writer = None

    def on_train_begin(self, logs=None):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._handle)
        self._writer.writerow(["epoch", "loss", "accuracy", "val_loss", "val_accuracy"])
        self._handle.flush()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if self._writer is None or self._handle is None:
            return
        self._writer.writerow(
            [
                epoch + 1,
                _format_metric(logs, "loss"),
                _format_metric(logs, "accuracy", "sparse_categorical_accuracy"),
                _format_metric(logs, "val_loss"),
                _format_metric(logs, "val_accuracy", "val_sparse_categorical_accuracy"),
            ]
        )
        self._handle.flush()

    def close(self):
        if self._handle is not None:
            self._handle.close()
            self._handle = None


class BestEpochTracker(keras.callbacks.Callback):
    """Record the epoch selected by the same rule as ``save_best_only`` on val_loss."""

    def __init__(self, metadata: dict):
        super().__init__()
        self.metadata = metadata
        self.best = float("inf")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        value = logs.get("val_loss")
        if value is None:
            return
        value = float(value)
        improved = value < self.best
        logger.info(
            "epoch %d loss=%.4f accuracy=%.4f val_loss=%.4f val_accuracy=%.4f",
            epoch + 1,
            float(logs.get("loss", float("nan"))),
            float(_first(logs, "accuracy", "sparse_categorical_accuracy", default=float("nan"))),
            value,
            float(_first(logs, "val_accuracy", "val_sparse_categorical_accuracy", default=float("nan"))),
        )
        if improved:
            self.best = value
            self.metadata["checkpoint_epoch"] = epoch + 1
            self.metadata["best_val_loss"] = value
            logger.info("saved checkpoint at epoch %d (val_loss=%.4f)", epoch + 1, value)


def _initial_metadata(config, *, run_id, run_dir, log_path, project_dir, started_at) -> dict:
    git = git_info()
    return {
        "run_id": run_id,
        "experiment": config["experiment"],
        "status": "running",
        "started_at": started_at,
        "finished_at": None,
        "duration_seconds": None,
        "seed": config["seed"],
        "split_seed": None,
        "python_version": None,
        "tensorflow_version": None,
        "keras_version": None,
        "numpy_version": None,
        "pillow_version": None,
        "device_requested": config["device"],
        "device_selected": None,
        "device_name": None,
        "gpu_devices": [],
        "git_commit": git["commit"],
        "git_dirty": git["dirty"],
        "split_manifest": config["data"]["split_manifest"],
        "split_manifest_sha256": None,
        "split_counts": None,
        "architecture": None,
        "trainable_parameters": None,
        "total_parameters": None,
        "checkpoint_epoch": None,
        "best_val_loss": None,
        "validation_accuracy": None,
        "validation_macro_f1": None,
        "checkpoint": relative_to_project(run_dir / "model.keras", project_dir),
        "log": relative_to_project(log_path, project_dir),
        "error": None,
    }


def _log_run_header(config, metadata, device, versions) -> None:
    logger.info("run_id=%s", metadata["run_id"])
    logger.info("experiment=%s", config["experiment"])
    logger.info("started_at=%s", metadata["started_at"])
    logger.info("training_seed=%s", config["seed"])
    logger.info(
        "tensorflow=%s keras=%s numpy=%s pillow=%s python=%s",
        versions["tensorflow_version"],
        versions["keras_version"],
        versions["numpy_version"],
        versions["pillow_version"],
        versions["python_version"],
    )
    logger.info(
        "device_requested=%s device_selected=%s device_name=%s",
        device["requested"],
        device["selected"],
        device["name"],
    )
    logger.info("git_commit=%s git_dirty=%s", metadata["git_commit"], metadata["git_dirty"])
    logger.info("resolved configuration follows")
    for line in _yaml_lines(config):
        logger.info("%s", line)


def _yaml_lines(config: dict) -> list[str]:
    import yaml

    text = yaml.safe_dump(config, sort_keys=False).rstrip().splitlines()
    return text or [""]


def _format_metric(logs: dict, *names: str) -> str:
    value = _first(logs, *names, default=None)
    if value is None:
        return ""
    return f"{float(value):.6f}"


def _first(logs: dict, *names: str, default):
    for name in names:
        if name in logs and logs[name] is not None:
            return logs[name]
    return default
