"""A one-epoch smoke run, and a failed run that still leaves metadata."""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from traffic_signs.data.splits import DatasetError
from traffic_signs.evaluation.evaluate import evaluate_checkpoint
from traffic_signs.inference.predict import predict_image
from traffic_signs.training.trainer import train
from traffic_signs.utils.runs import file_sha256, write_json


def test_training_rejects_a_config_with_no_dataset_directory(tmp_path):
    config = _config()
    del config["data"]["root"]
    with pytest.raises(DatasetError, match="--data-root"):
        train(config, project_dir=tmp_path)
    assert not (tmp_path / "artifacts").exists()


def test_smoke_training_writes_checkpoint_metrics_and_can_be_evaluated(tmp_path):
    _write_images(tmp_path / "images")
    manifest_path = tmp_path / "splits" / "mini.json"
    write_json(manifest_path, _manifest())
    run_dir = train(_config(), project_dir=tmp_path)

    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert run_dir == tmp_path / "artifacts" / "smoke"
    assert metadata["status"] == "completed"
    assert metadata["run_id"] == "smoke"
    assert "uuid" not in metadata
    assert metadata["error"] is None
    assert metadata["experiment"] == "smoke"
    assert metadata["seed"] == 1
    assert metadata["split_seed"] == 7
    assert metadata["split_manifest_sha256"] == file_sha256(manifest_path)
    assert metadata["checkpoint_epoch"] == 1
    assert metadata["trainable_parameters"] > 0
    assert metadata["duration_seconds"] >= 0
    assert metadata["device_selected"] in {"cpu", "gpu"}
    assert metadata["tensorflow_version"]
    assert (run_dir / "config.yaml").is_file()
    assert (run_dir / "model.keras").is_file()

    history = (run_dir / "history.csv").read_text(encoding="utf-8").strip().splitlines()
    assert history[0] == "epoch,loss,accuracy,val_loss,val_accuracy"
    assert len(history) == 2
    epoch, loss, accuracy, val_loss, val_accuracy = history[1].split(",")
    assert epoch == "1"
    for value in (loss, accuracy, val_loss, val_accuracy):
        float(value)

    metrics_path = run_dir / "metrics.json"
    original_metrics = metrics_path.read_bytes()
    metrics = json.loads(original_metrics)
    assert metrics["split"] == "validation"
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert metrics["num_samples"] == 4
    assert len(metrics["confusion_matrix"]) == 2

    evaluation = evaluate_checkpoint(run_dir / "model.keras", "validation", project_dir=tmp_path)
    assert metrics_path.read_bytes() == original_metrics
    assert evaluation["metrics"]["split"] == "validation"
    assert evaluation["results_path"].parent == run_dir / "evaluations"
    assert not (run_dir / "evaluations").joinpath("metrics.json").exists()
    np.testing.assert_allclose(evaluation["metrics"]["accuracy"], metrics["accuracy"], atol=1e-5)

    image = next((tmp_path / "images").glob("*.png"))
    prediction = predict_image(run_dir / "model.keras", image, project_dir=tmp_path)
    assert prediction["class_id"] in (0, 1)
    assert prediction["class_name"] is None
    assert 0.0 < prediction["confidence"] <= 1.0


def test_retraining_an_experiment_replaces_its_artifact_directory(tmp_path):
    manifest = _manifest()
    manifest["samples"]["train"][0]["id"] = "missing.png"
    write_json(tmp_path / "splits" / "mini.json", manifest)
    with pytest.raises(DatasetError, match="missing"):
        train(_config(), project_dir=tmp_path)
    stale = tmp_path / "artifacts" / "smoke" / "old.txt"
    stale.write_text("stale", encoding="utf-8")
    with pytest.raises(DatasetError, match="missing"):
        train(_config(), project_dir=tmp_path)
    assert [path.name for path in (tmp_path / "artifacts").iterdir()] == ["smoke"]
    assert not stale.exists()
    metadata = json.loads((tmp_path / "artifacts" / "smoke" / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["run_id"] == "smoke"
    assert metadata["experiment"] == "smoke"


def test_failed_run_records_the_error_and_keeps_its_config(tmp_path):
    manifest = _manifest()
    manifest["samples"]["train"][0]["id"] = "missing.png"
    manifest_path = tmp_path / "splits" / "mini.json"
    write_json(manifest_path, manifest)
    with pytest.raises(DatasetError, match="missing"):
        train(_config(), project_dir=tmp_path)

    runs = list((tmp_path / "artifacts").iterdir())
    assert len(runs) == 1
    metadata = json.loads((runs[0] / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "failed"
    assert "DatasetError" in metadata["error"]
    assert (runs[0] / "config.yaml").is_file()
    assert not (runs[0] / "model.keras").exists()
    log_path = tmp_path / metadata["log"]
    assert "Training failed" in log_path.read_text(encoding="utf-8")


def _config():
    return {
        "experiment": "smoke",
        "seed": 1,
        "device": "cpu",
        "data": {
            "root": "images",
            "split_manifest": "splits/mini.json",
            "image_size": 16,
            "num_classes": 2,
        },
        "model": {
            "channels": [4, 8],
            "kernel_size": 3,
            "dropout": 0.0,
            "dense_units": 8,
            "min_spatial": 8,
        },
        "training": {
            "epochs": 1,
            "batch_size": 4,
            "learning_rate": 0.01,
            "optimizer": "adam",
        },
        "augmentation": {"rotation_degrees": 0, "brightness": 0, "zoom": 0},
    }


def _manifest():
    def sample(name, label, track):
        return {"id": name, "label": label, "track": track}

    return {
        "version": 1,
        "dataset": "gtsrb",
        "seed": 7,
        "validation_fraction": 0.5,
        "num_classes": 2,
        "samples": {
            "train": [
                sample("a0.png", 0, "00000/00000"),
                sample("a1.png", 0, "00000/00001"),
                sample("b0.png", 1, "00001/00000"),
                sample("b1.png", 1, "00001/00001"),
            ],
            "validation": [
                sample("c0.png", 0, "00000/00002"),
                sample("c1.png", 0, "00000/00003"),
                sample("d0.png", 1, "00001/00002"),
                sample("d1.png", 1, "00001/00003"),
            ],
            "test": [
                {"id": "e0.png", "label": 0},
                {"id": "e1.png", "label": 1},
            ],
        },
    }


def _write_images(root: Path) -> None:
    root.mkdir(parents=True)
    colors = {
        "a0.png": (220, 20, 20),
        "a1.png": (200, 40, 40),
        "b0.png": (20, 20, 220),
        "b1.png": (40, 40, 200),
        "c0.png": (210, 30, 30),
        "c1.png": (190, 50, 50),
        "d0.png": (30, 30, 210),
        "d1.png": (50, 50, 190),
        "e0.png": (180, 20, 20),
        "e1.png": (20, 20, 180),
    }
    for name, color in colors.items():
        Image.new("RGB", (16, 16), color).save(root / name)
