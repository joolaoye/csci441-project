"""Evaluation conditions, previews, and depth robustness comparisons."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from traffic_signs.data.dataset import make_sequence
from traffic_signs.data.transforms import preprocess_array
from traffic_signs.evaluation.conditions import (
    ConditionError,
    apply_condition,
    apply_condition_batch,
    condition_fingerprint,
    default_conditions_path,
    load_conditions,
    normalize_condition,
)
from traffic_signs.evaluation.evaluate import (
    evaluate_checkpoint,
    evaluate_conditions,
    requested_condition_names,
    score_split,
)
from traffic_signs.models.cnn import build_cnn
from traffic_signs.reporting.comparisons import compare_runs
from traffic_signs.reporting.previews import generate_condition_previews, preview_panels, select_preview_samples
from traffic_signs.reporting import ReportError
from traffic_signs.utils.runs import write_json, write_yaml

ROOT = Path(__file__).resolve().parents[1]


def test_shipped_evaluation_conditions_keep_generic_names():
    conditions = load_conditions(ROOT / "evaluation.yaml.example")
    assert load_conditions(ROOT / "evaluation.yaml") == conditions
    assert set(conditions) == {"clean", "rotation", "zoom", "brightness", "lowres"}
    assert conditions["clean"]["type"] == "none"
    assert conditions["rotation"]["parameters"] == {"degrees": 10.0}
    assert conditions["zoom"]["parameters"] == {"factor": 0.1}
    assert conditions["zoom"]["type"] == "zoom"
    assert conditions["brightness"]["parameters"] == {"delta": 0.1}
    assert conditions["lowres"]["parameters"] == {"image_size": 24}
    with pytest.raises(ConditionError, match="without digits"):
        normalize_condition("rotation10", {"type": "rotation", "degrees": 10})


def test_zoom_is_a_deterministic_center_zoom_in():
    image = np.zeros((16, 16, 3), dtype=np.float32)
    image[4:12, 4:12] = 1.0
    identity = normalize_condition("zoom", {"type": "zoom", "factor": 0})
    np.testing.assert_array_equal(apply_condition(image, identity, 16), image)
    doubled = normalize_condition("zoom", {"type": "zoom", "factor": 1})
    once = apply_condition(image, doubled, 16)
    twice = apply_condition(image, doubled, 16)
    np.testing.assert_array_equal(once, twice)
    assert once.shape == (16, 16, 3)
    assert once.min() >= 0 and once.max() <= 1
    assert once[8, 8, 0] == pytest.approx(1.0)
    assert once.mean() > 0.9
    assert once.mean() > image.mean()
    mild = normalize_condition("zoom", {"type": "zoom", "factor": 0.1})
    changed = apply_condition(image, mild, 16)
    assert changed.shape == image.shape
    assert not np.array_equal(changed, image)
    assert changed[8, 8, 0] > 0.9
    assert changed.mean() > image.mean()
    with pytest.raises(ConditionError, match="factor"):
        normalize_condition("zoom", {"type": "zoom", "factor": -0.1})


def test_conditions_are_deterministic_and_keep_shape_range_and_labels(tmp_path):
    image = _pattern()
    clean = normalize_condition("clean", {"type": "none"})
    rotation = normalize_condition("rotation", {"type": "rotation", "degrees": 10})
    brighter = normalize_condition("brightness", {"type": "brightness", "delta": 0.1})
    lowres = normalize_condition("lowres", {"type": "downsample", "image_size": 8})
    zoom = normalize_condition("zoom", {"type": "zoom", "factor": 0.1})
    original = preprocess_array(image, 16)
    assert np.array_equal(apply_condition(original, clean, 16), original)
    rotated_a = apply_condition(original, rotation, 16)
    rotated_b = apply_condition(original, rotation, 16)
    np.testing.assert_array_equal(rotated_a, rotated_b)
    assert rotated_a.shape == original.shape
    assert not np.array_equal(rotated_a, original)
    assert rotated_a.min() >= 0 and rotated_a.max() <= 1
    bright = apply_condition(original, brighter, 16)
    assert bright.mean() > original.mean()
    assert bright.max() <= 1
    restored = apply_condition(original, lowres, 16)
    assert restored.shape == (16, 16, 3)
    assert not np.allclose(restored, original)
    zoomed_a = apply_condition(original, zoom, 16)
    zoomed_b = apply_condition(original, zoom, 16)
    np.testing.assert_array_equal(zoomed_a, zoomed_b)
    assert zoomed_a.shape == original.shape
    assert not np.array_equal(zoomed_a, original)
    turned = _half_turn(original)
    np.testing.assert_allclose(turned, np.rot90(original, 2), atol=0.08)

    path = _save(tmp_path / "images" / "a.png", image)
    sample = {"id": "a.png", "label": 1, "track": "track-a"}
    sequence = make_sequence(
        [sample],
        root=tmp_path / "images",
        image_size=16,
        batch_size=1,
        training=False,
        augmentation=None,
        seed=0,
        condition=rotation,
    )
    batch, labels = sequence[0]
    np.testing.assert_array_equal(batch[0], apply_condition(preprocess_array(np.asarray(Image.open(path)), 16), rotation, 16))
    assert int(labels[0]) == 1
    again = make_sequence(
        [sample],
        root=tmp_path / "images",
        image_size=16,
        batch_size=1,
        training=False,
        augmentation=None,
        seed=0,
        condition=rotation,
    )
    np.testing.assert_array_equal(again[0][0], batch)


def test_clean_evaluation_matches_an_unconditioned_score_and_keeps_both_parameter_sets(tmp_path):
    images = tmp_path / "images"
    _save(images / "0_0.png", _pattern())
    _save(images / "1_1.png", _pattern()[:, ::-1])
    manifest = tmp_path / "splits" / "mini.json"
    write_json(manifest, _manifest())
    run_dir = tmp_path / "artifacts" / "depth2"
    run_dir.mkdir(parents=True)
    write_yaml(run_dir / "config.yaml", _run_config(channels=[4, 8]))
    write_json(run_dir / "metadata.json", {"run_id": "depth2", "experiment": "depth2"})
    model = build_cnn(16, 2, [4, 8], kernel_size=3, dropout=0.0, dense_units=8, min_spatial=8)
    model.save(run_dir / "model.keras")
    samples = _manifest()["samples"]["validation"]
    plain = score_split(
        run_dir / "model.keras",
        samples,
        data_root=images,
        image_size=16,
        num_classes=2,
        batch_size=2,
    )
    clean_path = tmp_path / "conditions.yaml"
    clean_path.write_text("conditions:\n  clean:\n    type: none\n  rotation:\n    type: rotation\n    degrees: 10\n", encoding="utf-8")
    clean = evaluate_checkpoint(
        run_dir / "model.keras",
        "validation",
        project_dir=tmp_path,
        data_root=str(images),
        condition="clean",
        conditions_path=clean_path,
    )
    assert clean["metrics"]["condition"]["type"] == "none"
    assert clean["metrics"]["accuracy"] == pytest.approx(plain["accuracy"])
    first = evaluate_checkpoint(
        run_dir / "model.keras",
        "validation",
        project_dir=tmp_path,
        data_root=str(images),
        condition="rotation",
        conditions_path=clean_path,
    )
    changed = tmp_path / "rotation15.yaml"
    changed.write_text("conditions:\n  clean:\n    type: none\n  rotation:\n    type: rotation\n    degrees: 15\n", encoding="utf-8")
    second = evaluate_checkpoint(
        run_dir / "model.keras",
        "validation",
        project_dir=tmp_path,
        data_root=str(images),
        condition="rotation",
        conditions_path=changed,
    )
    assert first["results_path"] != second["results_path"]
    assert first["results_path"].is_file() and second["results_path"].is_file()
    assert first["metrics"]["condition"]["name"] == second["metrics"]["condition"]["name"] == "rotation"
    assert first["metrics"]["condition"]["parameters"]["degrees"] == 10
    assert second["metrics"]["condition"]["parameters"]["degrees"] == 15
    assert first["metrics"]["condition"]["fingerprint"] != second["metrics"]["condition"]["fingerprint"]
    zoom = normalize_condition("zoom", {"type": "zoom", "factor": 0.1})
    first_zoom = score_split(
        run_dir / "model.keras",
        samples,
        data_root=images,
        image_size=16,
        num_classes=2,
        batch_size=2,
        condition=zoom,
    )
    second_zoom = score_split(
        run_dir / "model.keras",
        samples,
        data_root=images,
        image_size=16,
        num_classes=2,
        batch_size=2,
        condition=zoom,
    )
    assert first_zoom["accuracy"] == second_zoom["accuracy"]
    np.testing.assert_array_equal(first_zoom["probabilities"], second_zoom["probabilities"])


def test_preview_uses_the_evaluation_transform_and_reuses_a_matching_fingerprint(tmp_path):
    root = tmp_path / "images"
    for index, label in enumerate((0, 1, 0, 1)):
        _save(root / f"{label}_{index}.png", _pattern() if label == 0 else _pattern()[:, ::-1])
    manifest = tmp_path / "splits" / "mini.json"
    write_json(manifest, _manifest(extra=True))
    config = tmp_path / "conditions.yaml"
    config.write_text(
        "conditions:\n  clean:\n    type: none\n  rotation:\n    type: rotation\n    degrees: 10\n",
        encoding="utf-8",
    )
    from traffic_signs.data.splits import load_manifest

    loaded = load_manifest(manifest)
    chosen = select_preview_samples(loaded["samples"]["validation"], 4)
    prepared = [(sample, preprocess_array(np.asarray(Image.open(root / sample["id"])), 16)) for sample in chosen]
    rotation = normalize_condition("rotation", {"type": "rotation", "degrees": 10})
    for _sample, original, transformed in preview_panels(prepared, rotation, 16):
        np.testing.assert_array_equal(transformed, apply_condition(original, rotation, 16))
    first = generate_condition_previews(
        manifest_path=manifest,
        data_root=root,
        output_root=tmp_path / "previews",
        split="validation",
        image_size=16,
        conditions_path=config,
        condition_names=["rotation"],
        project_dir=tmp_path,
    )
    assert len(first["written"]) == 1
    assert first["written"][0].read_bytes().startswith(b"\x89PNG")
    saved = first["written"][0].read_bytes()
    second = generate_condition_previews(
        manifest_path=manifest,
        data_root=root,
        output_root=tmp_path / "previews",
        split="validation",
        image_size=16,
        conditions_path=config,
        condition_names=["rotation"],
        project_dir=tmp_path,
    )
    assert second["written"] == []
    assert second["reused"] == first["written"]
    assert first["written"][0].read_bytes() == saved
    config.write_text(
        "conditions:\n  clean:\n    type: none\n  rotation:\n    type: rotation\n    degrees: 15\n",
        encoding="utf-8",
    )
    third = generate_condition_previews(
        manifest_path=manifest,
        data_root=root,
        output_root=tmp_path / "previews",
        split="validation",
        image_size=16,
        conditions_path=config,
        condition_names=["rotation"],
        project_dir=tmp_path,
    )
    assert len(third["written"]) == 1
    assert third["written"][0] != first["written"][0]
    assert first["written"][0].is_file()


def test_condition_comparison_reports_accuracy_drop_and_rejects_ambiguous_runs(tmp_path):
    shallow = _comparison_run(tmp_path, "depth2", [4, 8], 100)
    deep = _comparison_run(tmp_path, "depth4", [4, 8, 8, 8], 240)
    _evaluation(shallow, "clean", {}, 0.80, "same-images")
    _evaluation(shallow, "rotation", {"degrees": 10}, 0.55, "same-images")
    _evaluation(deep, "clean", {}, 0.90, "same-images")
    _evaluation(deep, "rotation", {"degrees": 10}, 0.70, "same-images")
    result = compare_runs([shallow, deep], "depth", tmp_path / "out", conditions=["clean", "rotation"])
    text = result["latex"].read_text(encoding="utf-8")
    assert r"\begin{table}" in text
    assert not text.lstrip().startswith("%")
    assert "CNN depth" in text
    assert "Parameters" not in text
    assert r"\textwidth" in text
    assert "Rotation: +10" in text
    assert result["png"].read_bytes().startswith(b"\x89PNG")
    assert result["output_dir"].parent.name == "depth"
    rows = list(csv.DictReader(result["csv"].open(encoding="utf-8")))
    drop = next(row for row in rows if row["run_id"] == "depth2" and row["condition"] == "rotation")
    assert drop["condition_accuracy"] == "0.550000"
    assert drop["accuracy_drop_from_clean"] == "0.250000"
    assert drop["evaluation_id"]
    plain = compare_runs([shallow, deep], "depth", tmp_path / "plain")
    assert "Macro-F1" in plain["latex"].read_text(encoding="utf-8")
    with pytest.raises(ReportError, match="architecture"):
        compare_runs([shallow, deep], "augmentation", tmp_path / "rq1", conditions=["clean"])
    _evaluation(shallow, "rotation", {"degrees": 15}, 0.40, "same-images", tag="other")
    with pytest.raises(ReportError, match="different parameters"):
        compare_runs([shallow, deep], "depth", tmp_path / "ambiguous", conditions=["rotation"])
    other = _comparison_run(tmp_path, "depth3", [4, 8, 16], 180)
    _evaluation(other, "rotation", {"degrees": 10}, 0.60, "different-images")
    with pytest.raises(ReportError, match="same image ids"):
        compare_runs([deep, other], "depth", tmp_path / "ids", conditions=["rotation"])


def test_five_condition_tables_cover_depth_and_augmentation(tmp_path):
    names = ("clean", "rotation", "zoom", "brightness", "lowres")
    parameters = {
        "clean": {},
        "rotation": {"degrees": 10},
        "zoom": {"factor": 0.1},
        "brightness": {"delta": 0.1},
        "lowres": {"image_size": 24},
    }
    accuracies = {
        "clean": 0.80,
        "rotation": 0.70,
        "zoom": 0.66,
        "brightness": 0.74,
        "lowres": 0.60,
    }
    shallow = _comparison_run(tmp_path, "depth2", [4, 8], 100)
    middle = _comparison_run(tmp_path, "depth3", [4, 8, 16], 180)
    deep = _comparison_run(tmp_path, "depth4", [4, 8, 8, 8], 240)
    for run_dir, offset in ((shallow, 0.0), (middle, 0.05), (deep, 0.10)):
        for name in names:
            _evaluation(run_dir, name, parameters[name], accuracies[name] + offset, "same-images")
    depth = compare_runs(
        [shallow, middle, deep],
        "depth",
        tmp_path / "depth",
        conditions=list(names),
    )
    depth_text = depth["latex"].read_text(encoding="utf-8")
    assert r"\begin{tabular*}{\textwidth}" in depth_text
    header = next(line for line in depth_text.splitlines() if line.startswith("CNN depth"))
    assert header.startswith("CNN depth & Clean & Rotation & Zoom & Brightness & Low-res")
    assert "Parameters" not in header
    assert "Selected checkpoint epoch" not in depth_text
    depth_csv = list(csv.DictReader(depth["csv"].open(encoding="utf-8")))
    zoom_row = next(row for row in depth_csv if row["run_id"] == "depth2" and row["condition"] == "zoom")
    assert zoom_row["condition_accuracy"] == "0.660000"
    assert zoom_row["condition_macro_f1"] == "0.560000"
    assert zoom_row["condition_loss"] == "1.200000"
    assert zoom_row["checkpoint_epoch"] == "1"
    assert zoom_row["trainable_parameters"] == "100"
    assert depth["png"].read_bytes().startswith(b"\x89PNG")

    baseline = _comparison_run(tmp_path, "baseline", [4, 8], 100)
    rotation = _comparison_run(
        tmp_path,
        "rotation10",
        [4, 8],
        100,
        augmentation={"rotation_degrees": 10, "brightness": 0, "zoom": 0},
    )
    for run_dir in (baseline, rotation):
        for name in names:
            _evaluation(run_dir, name, parameters[name], accuracies[name], "same-images")
    augmentation = compare_runs(
        [baseline, rotation],
        "augmentation",
        tmp_path / "augmentation",
        conditions=list(names),
    )
    text = augmentation["latex"].read_text(encoding="utf-8")
    assert "Training augmentation" in text
    assert "None (baseline)" in text
    assert "Rotation only" in text
    assert "Parameters" not in text.split("Training augmentation", 1)[1].split(r"\midrule", 1)[0]
    for header in ("Clean", "Rotation", "Zoom", "Brightness", "Low-res"):
        assert header in text
    assert augmentation["output_dir"].parent.name == "augmentation"
    rows = list(csv.DictReader(augmentation["csv"].open(encoding="utf-8")))
    assert {row["configuration"] for row in rows} == {"None (baseline)", "Rotation only"}
    assert all(row["checkpoint_epoch"] == "1" for row in rows)
    assert all(row["condition_macro_f1"] for row in rows)
    changed = _comparison_run(
        tmp_path,
        "other-depth",
        [4, 8, 16],
        180,
        augmentation={"rotation_degrees": 10, "brightness": 0, "zoom": 0},
    )
    for name in names:
        _evaluation(changed, name, parameters[name], 0.5, "same-images")
    with pytest.raises(ReportError, match="architecture"):
        compare_runs([baseline, changed], "augmentation", tmp_path / "bad-arch", conditions=["clean", "zoom"])
    _evaluation(baseline, "zoom", {"factor": 0.2}, 0.4, "same-images", tag="other")
    with pytest.raises(ReportError, match="different parameters"):
        compare_runs([baseline, rotation], "augmentation", tmp_path / "bad-zoom", conditions=["zoom"])


def test_local_evaluation_file_is_preferred_over_the_example(tmp_path):
    example = tmp_path / "evaluation.yaml.example"
    example.write_text("conditions:\n  clean:\n    type: none\n", encoding="utf-8")
    assert default_conditions_path(tmp_path) == example
    local = tmp_path / "evaluation.yaml"
    local.write_text("conditions:\n  clean:\n    type: none\n", encoding="utf-8")
    assert default_conditions_path(tmp_path) == local


def test_one_evaluation_scores_every_requested_condition(tmp_path, monkeypatch):
    images = tmp_path / "images"
    _save(images / "0_0.png", _pattern())
    _save(images / "1_1.png", _pattern()[:, ::-1])
    write_json(tmp_path / "splits" / "mini.json", _manifest())
    run_dir = tmp_path / "artifacts" / "depth2"
    run_dir.mkdir(parents=True)
    write_yaml(run_dir / "config.yaml", _run_config(channels=[4, 8]))
    write_json(run_dir / "metadata.json", {"run_id": "depth2", "experiment": "depth2"})
    model = build_cnn(16, 2, [4, 8], kernel_size=3, dropout=0.0, dense_units=8, min_spatial=8)
    model.save(run_dir / "model.keras")
    config = tmp_path / "conditions.yaml"
    config.write_text(
        "\n".join(
            [
                "conditions:",
                "  clean:",
                "    type: none",
                "  rotation:",
                "    type: rotation",
                "    degrees: 10",
                "  zoom:",
                "    type: zoom",
                "    factor: 0.1",
                "  brightness:",
                "    type: brightness",
                "    delta: 0.1",
                "  lowres:",
                "    type: downsample",
                "    image_size: 8",
            ]
        ),
        encoding="utf-8",
    )
    loads = {"count": 0}
    from tensorflow import keras

    real_load = keras.models.load_model

    def spy(path, *args, **kwargs):
        loads["count"] += 1
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr("traffic_signs.evaluation.evaluate.keras.models.load_model", spy)
    names = ["clean", "rotation", "zoom", "brightness", "lowres"]
    results = evaluate_conditions(
        run_dir / "model.keras",
        "validation",
        names,
        project_dir=tmp_path,
        data_root=str(images),
        conditions_path=config,
    )
    assert loads["count"] == 1
    assert [item["metrics"]["condition"]["name"] for item in results] == names
    assert len({item["results_path"] for item in results}) == 5
    assert all(item["results_path"].is_file() for item in results)
    assert results[0]["metrics"]["checkpoint"].endswith("model.keras")
    with pytest.raises(ValueError, match="Unknown condition"):
        evaluate_conditions(
            run_dir / "model.keras",
            "validation",
            ["clean", "missing"],
            project_dir=tmp_path,
            data_root=str(images),
            conditions_path=config,
        )


def test_condition_names_default_to_clean_and_reject_both_flags():
    assert requested_condition_names(None, None) == ["clean"]
    assert requested_condition_names(None, ["clean", "zoom", "clean"]) == ["clean", "zoom"]
    with pytest.raises(ValueError, match="not both"):
        requested_condition_names("rotation", ["clean"])


def test_condition_fingerprint_changes_with_parameters():
    ten = normalize_condition("rotation", {"type": "rotation", "degrees": 10})
    fifteen = normalize_condition("rotation", {"type": "rotation", "degrees": 15})
    assert condition_fingerprint(ten, image_size=48) != condition_fingerprint(fifteen, image_size=48)


def _half_turn(image):
    return apply_condition_batch(image[None, ...], normalize_condition("rotation", {"type": "rotation", "degrees": 180}), 16)[0]


def _pattern():
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[:8, :8] = (220, 30, 30)
    image[:8, 8:] = (30, 30, 220)
    image[8:, :] = (240, 220, 40)
    return image


def _save(path: Path, array: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)
    return path


def _manifest(extra: bool = False):
    validation = [
        {"id": "0_0.png", "label": 0, "track": "v0"},
        {"id": "1_1.png", "label": 1, "track": "v1"},
    ]
    if extra:
        validation.extend(
            [
                {"id": "0_2.png", "label": 0, "track": "v2"},
                {"id": "1_3.png", "label": 1, "track": "v3"},
            ]
        )
    return {
        "version": 1,
        "dataset": "gtsrb",
        "seed": 1,
        "validation_fraction": 0.5,
        "num_classes": 2,
        "class_names": ["Alpha", "Beta"],
        "samples": {
            "train": [{"id": "train.png", "label": 0, "track": "t0"}],
            "validation": validation,
            "test": [{"id": "test.png", "label": 1}],
        },
    }


def _run_config(channels):
    return {
        "experiment": "depth",
        "seed": 1,
        "device": "cpu",
        "data": {
            "root": "images",
            "split_manifest": "splits/mini.json",
            "image_size": 16,
            "num_classes": 2,
        },
        "model": {
            "channels": channels,
            "kernel_size": 3,
            "dropout": 0.0,
            "dense_units": 8,
            "min_spatial": 8,
        },
        "training": {"epochs": 1, "batch_size": 2, "learning_rate": 0.01, "optimizer": "adam"},
        "augmentation": {"rotation_degrees": 0, "brightness": 0, "zoom": 0},
    }


def _comparison_run(
    root: Path,
    name: str,
    channels: list[int],
    parameters: int,
    augmentation: dict | None = None,
) -> Path:
    run_dir = root / "runs" / name
    run_dir.mkdir(parents=True)
    config = _run_config(channels)
    config["experiment"] = name
    if augmentation is not None:
        config["augmentation"] = augmentation
    write_yaml(run_dir / "config.yaml", config)
    write_json(
        run_dir / "metadata.json",
        {
            "run_id": name,
            "experiment": name,
            "status": "completed",
            "checkpoint_epoch": 1,
            "best_val_loss": 0.5,
            "duration_seconds": 3,
            "trainable_parameters": parameters,
            "split_manifest_sha256": "abc",
        },
    )
    write_json(
        run_dir / "metrics.json",
        {"accuracy": 0.5, "macro_f1": 0.4, "split": "validation", "num_classes": 2, "per_class": []},
    )
    return run_dir


def _condition_label(name: str, parameters: dict) -> str:
    if name == "clean":
        return "Clean"
    if "degrees" in parameters:
        return f"Rotation: +{parameters['degrees']}°"
    if "factor" in parameters:
        return f"Zoom in: {1 + float(parameters['factor']):g}×"
    if "delta" in parameters:
        return f"Brightness: +{parameters['delta']}"
    if "image_size" in parameters:
        return f"Low resolution: {parameters['image_size']}"
    return name


def _evaluation(run_dir: Path, name: str, parameters: dict, accuracy: float, sample_hash: str, tag: str = "a") -> None:
    condition_type = {
        "degrees": "rotation",
        "factor": "zoom",
        "delta": "brightness",
        "image_size": "downsample",
    }
    kind = "none"
    for key, value in condition_type.items():
        if key in parameters:
            kind = value
            break
    payload = {
        "split": "validation",
        "accuracy": accuracy,
        "macro_f1": accuracy - 0.1,
        "loss": 1.2,
        "image_size": 16,
        "split_manifest_sha256": "abc",
        "sample_ids_sha256": sample_hash,
        "evaluation_id": f"{name}-{tag}",
        "created_at": "2026-11-08T00:00:00+00:00",
        "condition": {
            "name": name,
            "type": kind,
            "parameters": parameters,
            "label": _condition_label(name, parameters),
        },
    }
    directory = run_dir / "evaluations"
    directory.mkdir(exist_ok=True)
    (directory / f"{name}-{tag}.json").write_text(json.dumps(payload), encoding="utf-8")
