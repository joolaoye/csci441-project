"""Preprocessing shape, value range, and where augmentation is applied."""

from pathlib import Path

import numpy as np
from PIL import Image

from traffic_signs.data.dataset import TrafficSignSequence
from traffic_signs.data.transforms import (
    apply_augmentation,
    build_augmentor,
    load_rgb,
    preprocess_array,
)


def test_preprocess_scales_uint8_rgb_into_the_unit_interval():
    black = np.zeros((20, 30, 3), dtype=np.uint8)
    white = np.full((20, 30, 3), 255, dtype=np.uint8)
    black_out = preprocess_array(black, 48)
    white_out = preprocess_array(white, 48)
    assert black_out.shape == (48, 48, 3)
    assert white_out.shape == (48, 48, 3)
    assert black_out.dtype == np.float32
    assert black_out.min() == 0
    assert black_out.max() == 0
    assert white_out.min() == pytest_approx_one()
    assert white_out.max() == pytest_approx_one()

    pattern = np.zeros((25, 40, 3), dtype=np.uint8)
    pattern[:, :20] = (255, 128, 0)
    processed = preprocess_array(pattern, 16)
    assert processed.shape == (16, 16, 3)
    assert processed.min() >= 0
    assert processed.max() <= 1


def test_grayscale_file_is_converted_to_rgb(tmp_path):
    path = tmp_path / "gray.png"
    Image.new("L", (8, 8), 128).save(path)
    array = load_rgb(path)
    assert array.shape == (8, 8, 3)
    assert array.dtype == np.uint8


def test_zero_augmentation_is_disabled_and_validation_is_deterministic(tmp_path):
    assert build_augmentor({"rotation_degrees": 0, "brightness": 0, "zoom": 0}, seed=1) is None
    path = _pattern_image(tmp_path / "sign.png")
    samples = [{"id": "sign.png", "label": 3, "track": "00003/00001"}]
    sequence = TrafficSignSequence(
        samples,
        root=tmp_path,
        image_size=16,
        batch_size=2,
        training=False,
        augmentor=build_augmentor(
            {"rotation_degrees": 15, "brightness": 0.2, "zoom": 0.1},
            seed=1,
        ),
        seed=0,
    )
    images, labels = sequence[0]
    expected = preprocess_array(load_rgb(path), 16)
    np.testing.assert_array_equal(images[0], expected)
    assert labels.tolist() == [3]
    images_again, _ = sequence[0]
    np.testing.assert_array_equal(images, images_again)

    unchanged = apply_augmentation(
        images,
        build_augmentor({"rotation_degrees": 15, "brightness": 0.2, "zoom": 0.1}, seed=1),
        training=False,
    )
    np.testing.assert_array_equal(unchanged, images)


def test_training_augmentation_stays_in_range_and_changes_the_batch(tmp_path):
    _pattern_image(tmp_path / "sign.png")
    samples = [{"id": "sign.png", "label": 1, "track": "00001/00000"}]
    plain = TrafficSignSequence(
        samples,
        root=tmp_path,
        image_size=32,
        batch_size=1,
        training=False,
        augmentor=None,
        seed=2,
    )
    augmentor = build_augmentor({"rotation_degrees": 15, "brightness": 0.1, "zoom": 0.1}, seed=2)
    augmented_sequence = TrafficSignSequence(
        samples,
        root=tmp_path,
        image_size=32,
        batch_size=1,
        training=True,
        augmentor=augmentor,
        seed=2,
    )
    original, _ = plain[0]
    augmented, labels = augmented_sequence[0]
    assert augmented.shape == original.shape
    assert labels.tolist() == [1]
    assert augmented.min() >= 0
    assert augmented.max() <= 1
    assert not np.allclose(augmented, original)


def _pattern_image(path: Path) -> Path:
    array = np.zeros((30, 20, 3), dtype=np.uint8)
    array[:15, :] = (220, 30, 30)
    array[15:, :10] = (30, 30, 220)
    Image.fromarray(array, mode="RGB").save(path)
    return path


def pytest_approx_one():
    import pytest

    return pytest.approx(1.0)
