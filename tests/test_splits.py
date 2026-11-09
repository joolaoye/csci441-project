"""Track-aware splitting and manifest preparation on a tiny Kaggle-shaped tree."""

import csv
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from traffic_signs.data.labels import CLASS_NAMES
from traffic_signs.data.splits import (
    DatasetError,
    OFFICIAL_TEST_IMAGES,
    OFFICIAL_TRAIN_IMAGES,
    prepare_dataset,
    split_samples,
    track_token,
    validate_manifest,
    validation_track_count,
)
from traffic_signs.data.transforms import load_rgb, preprocess_array


def test_validation_track_count_rounds_half_up_and_keeps_both_sides():
    assert validation_track_count(10, 0.2) == 2
    assert validation_track_count(8, 0.25) == 2
    assert validation_track_count(3, 0.2) == 1
    assert validation_track_count(2, 0.2) == 1
    assert validation_track_count(1, 0.2) == 0


def test_gtsrb_track_token_uses_the_documented_sequence_id():
    assert track_token("00020_00004_00002.png", 20) == "00004"
    assert track_token("00000_00001.ppm", 0) == "00000"
    assert len(CLASS_NAMES) == 43
    assert OFFICIAL_TRAIN_IMAGES == 39209
    assert OFFICIAL_TEST_IMAGES == 12630
    with pytest.raises(DatasetError, match="annotation says class 3"):
        track_token("00020_00004_00002.png", 3)


def test_split_is_deterministic_and_ignores_global_numpy_state():
    samples = _synthetic_samples()
    np.random.seed(0)
    first_train, first_val = split_samples(samples, 0.25, seed=42)
    np.random.seed(12345)
    second_train, second_val = split_samples(samples, 0.25, seed=42)
    assert [sample["id"] for sample in first_train] == [sample["id"] for sample in second_train]
    assert [sample["id"] for sample in first_val] == [sample["id"] for sample in second_val]


def test_tracks_are_disjoint_and_frames_stay_together():
    samples = _synthetic_samples()
    train, validation = split_samples(samples, 0.25, seed=42)
    train_tracks = {sample["track"] for sample in train}
    validation_tracks = {sample["track"] for sample in validation}
    assert train_tracks.isdisjoint(validation_tracks)

    split_by_track = {}
    for name, group in (("train", train), ("validation", validation)):
        for sample in group:
            split_by_track.setdefault(sample["track"], set()).add(name)
    assert all(len(names) == 1 for names in split_by_track.values())
    assert all(
        sum(sample["track"] == track for sample in train + validation) == 2
        for track in split_by_track
    )

    train_ids = [sample["id"] for sample in train]
    validation_ids = [sample["id"] for sample in validation]
    assert train_ids == sorted(train_ids)
    assert validation_ids == sorted(validation_ids)
    assert len(train_ids + validation_ids) == len(set(train_ids + validation_ids)) == len(samples)
    for label in (0, 1):
        assert len({sample["track"] for sample in validation if sample["label"] == label}) == 1
        assert len({sample["track"] for sample in train if sample["label"] == label}) == 3


def test_existing_manifest_is_not_regenerated(tmp_path):
    manifest = tmp_path / "gtsrb_v1.json"
    manifest.write_text('{"keep": true}\n', encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to regenerate"):
        prepare_dataset(tmp_path / "missing", manifest, overwrite=False)
    assert manifest.read_text(encoding="utf-8") == '{"keep": true}\n'


def test_prepare_writes_a_deterministic_track_split(tmp_path):
    first_root = tmp_path / "one"
    second_root = tmp_path / "two"
    _build_mini_dataset(first_root)
    _build_mini_dataset(second_root)
    first_manifest = tmp_path / "first.json"
    second_manifest = tmp_path / "second.json"
    first = prepare_dataset(
        first_root,
        first_manifest,
        seed=42,
        validation_fraction=0.25,
        enforce_counts=False,
    )
    second = prepare_dataset(
        second_root,
        second_manifest,
        seed=42,
        validation_fraction=0.25,
        enforce_counts=False,
    )
    assert first_manifest.read_text(encoding="utf-8") == second_manifest.read_text(encoding="utf-8")
    manifest = first["manifest_body"]
    assert manifest["dataset"] == "gtsrb"
    assert manifest["seed"] == 42
    assert manifest["num_classes"] == 2
    assert "class_names" not in manifest
    counts = manifest["counts"]
    assert counts["train_images"] + counts["validation_images"] == 9
    assert counts["test_images"] == 2
    assert counts["train_tracks"] == 6
    assert counts["validation_tracks"] == 2
    train_tracks = {sample["track"] for sample in manifest["samples"]["train"]}
    validation_tracks = {sample["track"] for sample in manifest["samples"]["validation"]}
    assert train_tracks.isdisjoint(validation_tracks)

    alias_splits = {
        sample["id"]: split
        for split, samples in manifest["samples"].items()
        if split != "test"
        for sample in samples
        if sample["track"] == "0/00000"
    }
    assert set(alias_splits) == {
        "Train/0/00000_00000_00000.png",
        "Train/0/00000_00000_00001.png",
    }
    assert len(set(alias_splits.values())) == 1

    test_ids = [sample["id"] for sample in manifest["samples"]["test"]]
    assert test_ids == ["Test/00000.png", "Test/00001.png"]
    used = {sample["id"] for sample in manifest["samples"]["train"] + manifest["samples"]["validation"]}
    assert used.isdisjoint(test_ids)
    for sample in manifest["samples"]["train"] + manifest["samples"]["validation"] + manifest["samples"]["test"]:
        assert (first_root / sample["id"]).is_file()
        assert not Path(sample["id"]).is_absolute()
    pixels = preprocess_array(load_rgb(first_root / "Train/0/00000_00000_00000.png"), image_size=8)
    assert pixels.shape == (8, 8, 3)
    assert float(pixels.min()) >= 0.0
    assert float(pixels.max()) <= 1.0
    with pytest.raises(FileExistsError, match="Refusing to regenerate"):
        prepare_dataset(first_root, first_manifest, enforce_counts=False)


def test_partial_extract_does_not_claim_the_official_counts(tmp_path):
    root = tmp_path / "partial"
    _build_mini_dataset(root)
    with pytest.raises(DatasetError, match="39209"):
        prepare_dataset(root, tmp_path / "manifest.json", enforce_counts=True)


def test_fifty_two_class_extract_is_rejected(tmp_path):
    root = tmp_path / "old"
    (root / "DATA" / "0").mkdir(parents=True)
    (root / "labels.csv").write_text("ClassId,Name\n0,Speed\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="52-class"):
        prepare_dataset(root, tmp_path / "manifest.json", enforce_counts=False)


def test_official_archive_keeps_a_track_together_and_holds_out_the_test_set(tmp_path):
    root = tmp_path / "official"
    _build_official_mini(root)
    result = prepare_dataset(root, tmp_path / "manifest.json", seed=42, validation_fraction=0.25, enforce_counts=False)
    manifest = result["manifest_body"]
    grouped = [
        sample["id"]
        for split in ("train", "validation")
        for sample in manifest["samples"][split]
        if sample["track"] == "0/00000"
    ]
    assert grouped == [
        "Final_Training/Images/00000/00000_00000.ppm",
        "Final_Training/Images/00000/00000_00001.ppm",
    ]
    test_ids = {sample["id"] for sample in manifest["samples"]["test"]}
    assert test_ids == {"Final_Test/Images/00000.ppm", "Final_Test/Images/00001.ppm"}
    used = {sample["id"] for sample in manifest["samples"]["train"] + manifest["samples"]["validation"]}
    assert used.isdisjoint(test_ids)
    assert manifest["counts"]["test_images"] == 2


def test_manifest_rejects_ids_that_escape_the_dataset_root():
    manifest = {
        "version": 1,
        "dataset": "gtsrb",
        "seed": 1,
        "validation_fraction": 0.2,
        "num_classes": 2,
        "samples": {
            "train": [{"id": "../secret.png", "label": 0, "track": "0/0001"}],
            "validation": [{"id": "DATA/1/001_0001.png", "label": 1, "track": "1/0001"}],
            "test": [],
        },
    }
    with pytest.raises(DatasetError, match="relative"):
        validate_manifest(manifest)


def _synthetic_samples(n_classes=2, n_tracks=4, n_frames=2):
    samples = []
    for label in range(n_classes):
        for track in range(n_tracks):
            for frame in range(n_frames):
                samples.append(
                    {
                        "id": f"DATA/{label}/{track:03d}_{frame:04d}.png",
                        "label": label,
                        "track": f"{label}/{track:04d}",
                    }
                )
    return samples


def _build_mini_dataset(root: Path) -> None:
    root.mkdir(parents=True)
    train_rows = []
    for class_id, tracks in (
        (0, [("00000", 2), ("00001", 1), ("00002", 1), ("00003", 1)]),
        (1, [("00000", 1), ("00001", 1), ("00002", 1), ("00003", 1)]),
    ):
        for track, frames in tracks:
            for frame in range(frames):
                name = f"{class_id:05d}_{track}_{frame:05d}.png"
                relative = f"Train/{class_id}/{name}"
                _write_image(root / relative, (class_id * 40, int(track), frame * 40))
                train_rows.append([8, 8, 1, 1, 6, 6, class_id, relative])
    _write_index(root / "Train.csv", train_rows)
    test_rows = []
    for index, class_id in enumerate((0, 1)):
        relative = f"Test/{index:05d}.png"
        _write_image(root / relative, (0, 0, 40 + class_id))
        test_rows.append([8, 8, 1, 1, 6, 6, class_id, relative])
    _write_index(root / "Test.csv", test_rows)


def _build_official_mini(root: Path) -> None:
    rows = []
    for track, frames in (("00000", 2), ("00001", 1), ("00002", 1), ("00003", 1)):
        for frame in range(frames):
            name = f"{track}_{frame:05d}.ppm"
            path = root / "Final_Training" / "Images" / "00000" / name
            _write_image(path, (10, int(track[-1]), frame * 20))
            rows.append([name, 8, 8, 1, 1, 6, 6, 0])
    _write_official_csv(root / "Final_Training" / "Images" / "00000" / "GT-00000.csv", rows)
    rows = []
    for track in range(4):
        name = f"{track:05d}_00000.ppm"
        path = root / "Final_Training" / "Images" / "00001" / name
        _write_image(path, (0, track * 20, 0))
        rows.append([name, 8, 8, 1, 1, 6, 6, 1])
    _write_official_csv(root / "Final_Training" / "Images" / "00001" / "GT-00001.csv", rows)
    test_rows = []
    for index, class_id in enumerate((0, 1)):
        name = f"{index:05d}.ppm"
        _write_image(root / "Final_Test" / "Images" / name, (0, 0, 30 + class_id))
        test_rows.append([name, 8, 8, 1, 1, 6, 6, class_id])
    _write_official_csv(root / "GT-final_test.csv", test_rows)


def _write_index(path: Path, rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Width", "Height", "Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2", "ClassId", "Path"])
        writer.writerows(rows)


def _write_official_csv(path: Path, rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(["Filename", "Width", "Height", "Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2", "ClassId"])
        writer.writerows(rows)


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color).save(path)
