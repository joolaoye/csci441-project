"""Saving a Keras checkpoint and predicting with the saved file."""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from tensorflow import keras

from traffic_signs.data.labels import class_name_for_model
from traffic_signs.evaluation.evaluate import evaluate_checkpoint
from traffic_signs.inference.predict import predict_image
from traffic_signs.models.cnn import build_cnn
from traffic_signs.utils.runs import write_yaml


def test_saved_keras_file_reloads_with_the_same_predictions(tmp_path):
    model = build_cnn(
        image_size=16,
        num_classes=43,
        channels=[4, 4],
        kernel_size=3,
        dropout=0.0,
        dense_units=8,
        min_spatial=8,
    )
    inputs = np.random.default_rng(0).random((2, 16, 16, 3), dtype=np.float32)
    before = model.predict(inputs, verbose=0)
    checkpoint = tmp_path / "run" / "model.keras"
    checkpoint.parent.mkdir()
    model.save(checkpoint)
    loaded = keras.models.load_model(checkpoint)
    after = loaded.predict(inputs, verbose=0)
    np.testing.assert_allclose(before, after, atol=1e-5)

    _write_run_files(checkpoint.parent)
    image_path = tmp_path / "sign.png"
    Image.new("RGB", (20, 18), (180, 40, 40)).save(image_path)
    prediction = predict_image(checkpoint, image_path, project_dir=tmp_path)
    assert prediction["class_id"] in range(43)
    assert prediction["class_name"] == class_name_for_model(prediction["class_id"], 43)
    assert prediction["class_name"] is not None
    assert 0.0 < prediction["confidence"] <= 1.0


def test_evaluate_rejects_the_training_split():
    with pytest.raises(ValueError, match="validation"):
        evaluate_checkpoint(Path("missing.keras"), "train")


def _write_run_files(run_dir: Path) -> None:
    write_yaml(
        run_dir / "config.yaml",
        {
            "experiment": "reload",
            "seed": 1,
            "device": "cpu",
            "data": {
                "root": "images",
                "split_manifest": "splits/mini.json",
                "image_size": 16,
                "num_classes": 43,
            },
            "model": {
                "channels": [4, 4],
                "kernel_size": 3,
                "dropout": 0.0,
                "dense_units": 8,
                "min_spatial": 8,
            },
            "training": {
                "epochs": 1,
                "batch_size": 2,
                "learning_rate": 0.01,
                "optimizer": "adam",
            },
            "augmentation": {"rotation_degrees": 0, "brightness": 0, "zoom": 0},
        },
    )
    (run_dir / "metadata.json").write_text(
        json.dumps({"run_id": "reload_test", "experiment": "reload"}) + "\n",
        encoding="utf-8",
    )
