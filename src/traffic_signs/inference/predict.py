"""Predict the class of one traffic-sign image."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from tensorflow import keras

from traffic_signs.data.labels import class_name_for_model
from traffic_signs.data.transforms import load_rgb, preprocess_array
from traffic_signs.evaluation.evaluate import load_run
from traffic_signs.utils.runs import configure_logging, utc_stamp

logger = logging.getLogger(__name__)


def predict_image(
    checkpoint: Path,
    image_path: Path,
    *,
    project_dir: Path | None = None,
) -> dict:
    """Load a checkpoint, preprocess one image, and return the top class."""
    project_dir = Path(project_dir or Path.cwd())
    checkpoint = Path(checkpoint)
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    log_path = configure_logging("predict", utc_stamp(), project_dir)
    try:
        _run_dir, config, metadata = load_run(checkpoint)
        image_size = config["data"]["image_size"]
        num_classes = config["data"]["num_classes"]
        model = keras.models.load_model(checkpoint)
        image = preprocess_array(load_rgb(image_path), image_size)
        probabilities = model.predict(image[None, ...], verbose=0)[0]
        class_id = int(np.argmax(probabilities))
        confidence = float(probabilities[class_id])
        result = {
            "class_id": class_id,
            "class_name": class_name_for_model(class_id, num_classes),
            "confidence": confidence,
            "run_id": metadata.get("run_id"),
            "log_path": log_path,
        }
        name = result["class_name"] or "unavailable"
        logger.info("image=%s", image_path)
        logger.info("source_run_id=%s checkpoint=%s", result["run_id"], checkpoint)
        logger.info("class_id=%d class_name=%s confidence=%.4f", class_id, name, confidence)
        return result
    except Exception:
        logger.exception("Prediction failed")
        raise
