"""Figures and comparison tables from saved artifacts, without training."""

import csv
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from traffic_signs.evaluation.metrics import evaluate_predictions
from traffic_signs.reporting import ReportError
from traffic_signs.utils.config import ConfigError, load_config
from traffic_signs.reporting.comparisons import compare_runs
from traffic_signs.reporting.plots import (
    augmentation_panels,
    count_annotations,
    enabled_augmentations,
    generate_dataset_report,
    generate_run_report,
    plot_confusion_matrices,
    read_history,
    row_normalize_confusion,
    write_per_class_csv,
)
from traffic_signs.utils.runs import write_json, write_yaml


def test_count_annotations_leave_zero_cells_blank():
    assert count_annotations([[4, 0], [0, 2]]) == [(0, 0, "4"), (1, 1, "2")]


def test_confusion_matrix_size_comes_from_the_evaluation(tmp_path):
    evaluation = {
        "num_classes": 3,
        "split": "validation",
        "confusion_matrix": [[2, 0, 1], [0, 0, 0], [0, 1, 3]],
    }
    written = plot_confusion_matrices(evaluation, tmp_path, "mini")
    names = {path.name for path in written}
    assert "confusion_matrix.png" in names
    assert "confusion_matrix_normalized.png" in names
    assert "confusion_matrix.pdf" not in names
    assert (tmp_path / "confusion_matrix.png").read_bytes().startswith(b"\x89PNG")
    with pytest.raises(ReportError, match="num_classes"):
        plot_confusion_matrices({"confusion_matrix": [[1]], "split": "validation"}, tmp_path, "mini")


def test_exploratory_52_class_run_cannot_be_compared(tmp_path):
    run = tmp_path / "old52"
    run.mkdir()
    write_yaml(
        run / "config.yaml",
        {
            "experiment": "baseline",
            "seed": 1,
            "device": "cpu",
            "data": {
                "root": "images",
                "split_manifest": "splits/old.json",
                "image_size": 16,
                "num_classes": 52,
            },
            "model": {
                "channels": [4, 8],
                "kernel_size": 3,
                "dropout": 0.0,
                "dense_units": 8,
                "min_spatial": 8,
            },
            "training": {"epochs": 2, "batch_size": 4, "learning_rate": 0.01, "optimizer": "adam"},
            "augmentation": {"rotation_degrees": 0, "brightness": 0, "zoom": 0},
        },
    )
    write_json(run / "metadata.json", {"status": "completed", "checkpoint_epoch": 1, "best_val_loss": 1.0})
    write_json(run / "metrics.json", {"accuracy": 0.5, "macro_f1": 0.4, "split": "validation"})
    with pytest.raises(ReportError, match="52-class"):
        compare_runs([run], "augmentation", tmp_path / "out")


def test_row_normalization_keeps_true_class_rows_and_zero_support():
    scored = evaluate_predictions([0, 1], [1, 1], num_classes=2)
    matrix = np.asarray(scored["confusion_matrix"])
    assert matrix.tolist() == [[0, 1], [0, 1]]
    normalized = row_normalize_confusion(matrix)
    assert normalized.tolist() == [[0.0, 1.0], [0.0, 1.0]]
    with_empty = row_normalize_confusion([[2, 0], [0, 0]])
    assert with_empty[0].tolist() == [1.0, 0.0]
    assert with_empty[1].tolist() == [0.0, 0.0]


def test_per_class_table_copies_stored_scores_including_zero_support(tmp_path):
    evaluation = {
        "num_classes": 2,
        "per_class": [
            {"class_id": 0, "name": "Alpha", "support": 2, "precision": 1.0, "recall": 0.5, "f1": 2 / 3},
            {"class_id": 1, "name": "Beta", "support": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0},
        ],
    }
    path = write_per_class_csv(evaluation, tmp_path / "per_class.csv")
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert rows[1]["class_id"] == "1"
    assert rows[1]["name"] == "Beta"
    assert rows[1]["support"] == "0"
    assert rows[1]["precision"] == "0.000000"
    assert rows[1]["f1"] == "0.000000"


def test_history_reader_rejects_a_missing_metric(tmp_path):
    path = tmp_path / "history.csv"
    path.write_text("epoch,loss,accuracy,val_loss,val_accuracy\n1,0.5,0.4,,0.2\n", encoding="utf-8")
    with pytest.raises(ReportError, match="val_loss"):
        read_history(path)


def test_run_report_writes_figures_from_saved_metrics(tmp_path):
    image_path = _write_image(tmp_path / "images" / "sign.png")
    run_dir = _write_run(tmp_path, name="baseline", seed=1, accuracy=0.5, macro_f1=0.4, manifest_hash="abc")
    result = generate_run_report(run_dir, project_dir=tmp_path, augmentation_seed=0)
    names = {path.name for path in result["written"]}
    assert "training_accuracy.png" in names
    assert "training_loss.png" in names
    assert "confusion_matrix.png" in names
    assert "confusion_matrix_normalized.png" in names
    assert "per_class.csv" in names
    assert "per_class_f1.png" in names
    assert "class_distribution.png" not in names
    assert "augmentation_examples.png" in names
    assert not any(path.suffix == ".pdf" for path in result["written"])
    for path in result["written"]:
        assert path.stat().st_size > 0
    assert (result["output_dir"] / "training_accuracy.png").read_bytes().startswith(b"\x89PNG")
    panels_a = augmentation_panels(np.asarray(Image.open(image_path)), 16, _factors(), seed=0)
    panels_b = augmentation_panels(np.asarray(Image.open(image_path)), 16, _factors(), seed=0)
    for left, right in zip(panels_a, panels_b, strict=True):
        np.testing.assert_array_equal(left, right)
    assert any(not np.array_equal(panels_a[0], panel) for panel in panels_a[1:])
    assert len(panels_a) == 4
    rotation_only = {"rotation_degrees": 10, "brightness": 0, "zoom": 0}
    assert [title for _settings, title in enabled_augmentations(rotation_only)] == ["Rotation 10°"]
    rotated = augmentation_panels(np.asarray(Image.open(image_path)), 16, rotation_only, seed=0)
    assert len(rotated) == 2
    assert not np.array_equal(rotated[0], rotated[1])


def test_run_report_uses_saved_augmentation_and_ignores_other_configs(tmp_path, monkeypatch):
    _write_invalid_augmentation_config(tmp_path)
    with pytest.raises(ConfigError, match="rotation_degrees"):
        load_config(tmp_path / "configs" / "augmentation.yaml")
    run_dir = _write_run(
        tmp_path,
        name="baseline",
        seed=1,
        accuracy=0.5,
        macro_f1=0.4,
        manifest_hash="abc",
        augmentation={"rotation_degrees": 0, "brightness": 0, "zoom": 0},
        tag="saved",
    )
    captured = {}

    def spy(image_path, image_size, augmentation, output_dir, **kwargs):
        captured["augmentation"] = dict(augmentation)
        captured["image"] = Path(image_path).name
        return []

    monkeypatch.setattr("traffic_signs.reporting.plots.plot_augmentation_examples", spy)
    result = generate_run_report(run_dir, project_dir=tmp_path)
    names = {path.name for path in result["written"]}
    assert captured == {}
    assert "training_accuracy.png" in names
    assert "confusion_matrix.png" in names
    assert "augmentation_examples.png" not in names
    assert "class_distribution.png" not in names
    assert result["warnings"] == []


def test_augmentation_figure_failure_keeps_the_other_figures(tmp_path, monkeypatch):
    _write_invalid_augmentation_config(tmp_path)
    run_dir = _write_run(tmp_path, name="baseline", seed=1, accuracy=0.5, macro_f1=0.4, manifest_hash="abc", tag="boom")

    def fail(*args, **kwargs):
        raise RuntimeError("preview failed")

    monkeypatch.setattr("traffic_signs.reporting.plots.plot_augmentation_examples", fail)
    result = generate_run_report(run_dir, project_dir=tmp_path)
    names = {path.name for path in result["written"]}
    assert "training_accuracy.png" in names
    assert "training_loss.png" in names
    assert "confusion_matrix.png" in names
    assert "per_class.csv" in names
    assert "class_distribution.png" not in names
    assert "augmentation_examples.png" not in names
    assert any("preview failed" in warning for warning in result["warnings"])


def test_compare_averages_distinct_seeds_and_leaves_single_runs_without_std(tmp_path):
    zeros = {"rotation_degrees": 0, "brightness": 0, "zoom": 0}
    first = _write_run(
        tmp_path, name="baseline", seed=1, accuracy=0.5, macro_f1=0.4, manifest_hash="abc", augmentation=zeros
    )
    second = _write_run(
        tmp_path, name="baseline", seed=2, accuracy=0.7, macro_f1=0.6, manifest_hash="abc", augmentation=zeros
    )
    alone = _write_run(
        tmp_path,
        name="augmentation",
        seed=1,
        accuracy=0.8,
        macro_f1=0.7,
        manifest_hash="abc",
        augmentation={"rotation_degrees": 10, "brightness": 0.1, "zoom": 0.1},
    )
    result = compare_runs([first, second, alone], "augmentation", tmp_path / "comparisons")
    rows = {row["label"]: row for row in result["rows"]}
    assert rows["none"]["n_runs"] == 2
    assert rows["none"]["validation_accuracy"] == pytest.approx(0.6)
    assert rows["none"]["validation_accuracy_std"] == pytest.approx(float(np.std([0.5, 0.7], ddof=1)))
    assert rows["none"]["macro_f1"] == pytest.approx(0.5)
    augmented = next(row for row in result["rows"] if row["label"] != "none")
    assert augmented["validation_accuracy_std"] is None
    assert augmented["n_runs"] == 1
    text = result["latex"].read_text(encoding="utf-8")
    assert r"\begin{table}" in text
    assert r"\caption{" in text
    assert r"\label{tab:augmentation-comparison-" in text
    assert r"\pm" in text
    assert r"\toprule" in text
    assert not text.lstrip().startswith("%")
    assert "Augmentation configuration" in text
    assert "Selected checkpoint epoch" not in text
    assert result["png"].read_bytes().startswith(b"\x89PNG")
    assert result["output_dir"].parent.name == "augmentation"
    csv_rows = list(csv.DictReader(result["csv"].open(encoding="utf-8")))
    assert len(csv_rows) == 3
    assert "validation_accuracy_std" not in csv_rows[0]
    augmented_row = next(row for row in csv_rows if row["configuration"] != "none")
    assert augmented_row["validation_accuracy"] == "0.800000"
    assert augmented_row["rotation_degrees"] == "10.000000"
    assert augmented_row["checkpoint_epoch"] == "2"


def test_depth_comparison_allows_different_depths_and_rejects_different_augmentation(tmp_path):
    zeros = {"rotation_degrees": 0, "brightness": 0, "zoom": 0}
    shallow = _write_run(
        tmp_path,
        name="depth2",
        seed=1,
        accuracy=0.4,
        macro_f1=0.3,
        manifest_hash="abc",
        augmentation=zeros,
        channels=[4, 8],
        trainable_parameters=100,
    )
    deep = _write_run(
        tmp_path,
        name="depth4",
        seed=1,
        accuracy=0.6,
        macro_f1=0.5,
        manifest_hash="abc",
        augmentation=zeros,
        channels=[4, 8, 8, 8],
        trainable_parameters=240,
    )
    result = compare_runs([shallow, deep], "depth", tmp_path / "comparisons")
    assert [row["depth"] for row in result["rows"]] == [2, 4]
    text = result["latex"].read_text(encoding="utf-8")
    assert "CNN depth" in text
    assert "Selected checkpoint epoch" in text
    assert "Trainable parameters" in text
    assert "100" in text
    assert result["png"].suffix == ".png"
    changed = _write_run(
        tmp_path,
        name="depth4aug",
        seed=2,
        accuracy=0.6,
        macro_f1=0.5,
        manifest_hash="abc",
        augmentation={"rotation_degrees": 10, "brightness": 0, "zoom": 0},
        channels=[4, 8, 8, 8],
    )
    with pytest.raises(ReportError, match="augmentation"):
        compare_runs([shallow, changed], "depth", tmp_path / "bad")


def test_compare_rejects_incompatible_manifests_and_failed_runs(tmp_path):
    zeros = {"rotation_degrees": 0, "brightness": 0, "zoom": 0}
    matching = _write_run(
        tmp_path, name="baseline", seed=1, accuracy=0.5, macro_f1=0.4, manifest_hash="abc", augmentation=zeros
    )
    other = _write_run(
        tmp_path, name="depth4", seed=1, accuracy=0.5, macro_f1=0.4, manifest_hash="def", augmentation=zeros
    )
    with pytest.raises(ReportError, match="split manifest"):
        compare_runs([matching, other], "depth", tmp_path / "out")
    failed = _write_run(
        tmp_path,
        name="failed",
        seed=1,
        accuracy=0.5,
        macro_f1=0.4,
        manifest_hash="abc",
        status="failed",
        augmentation=zeros,
    )
    with pytest.raises(ReportError, match="failed"):
        compare_runs([failed, matching], "augmentation", tmp_path / "out")


def test_repeated_seed_does_not_invent_a_standard_deviation(tmp_path):
    zeros = {"rotation_degrees": 0, "brightness": 0, "zoom": 0}
    first = _write_run(
        tmp_path,
        name="baseline",
        seed=1,
        accuracy=0.5,
        macro_f1=0.4,
        manifest_hash="abc",
        tag="a",
        augmentation=zeros,
    )
    second = _write_run(
        tmp_path,
        name="baseline",
        seed=1,
        accuracy=0.9,
        macro_f1=0.8,
        manifest_hash="abc",
        tag="b",
        augmentation=zeros,
    )
    augmented = _write_run(
        tmp_path,
        name="augmentation",
        seed=2,
        accuracy=0.7,
        macro_f1=0.6,
        manifest_hash="abc",
        augmentation={"rotation_degrees": 10, "brightness": 0.1, "zoom": 0.1},
    )
    result = compare_runs([first, second, augmented], "augmentation", tmp_path / "repeated")
    baseline = next(row for row in result["rows"] if row["label"] == "none")
    assert baseline["n_runs"] == 2
    assert baseline["validation_accuracy_std"] is None
    assert any("repeats seed" in warning for warning in result["warnings"])


def test_compare_rejects_different_training_protocols(tmp_path):
    zeros = {"rotation_degrees": 0, "brightness": 0, "zoom": 0}
    short = _write_run(
        tmp_path,
        name="baseline",
        seed=1,
        accuracy=0.5,
        macro_f1=0.4,
        manifest_hash="abc",
        epochs=20,
        augmentation=zeros,
    )
    long = _write_run(
        tmp_path,
        name="baseline",
        seed=1,
        accuracy=0.6,
        macro_f1=0.5,
        manifest_hash="abc",
        epochs=100,
        augmentation=zeros,
        channels=[32, 64],
    )
    with pytest.raises(ReportError, match="epoch budgets"):
        compare_runs([short, long], "depth", tmp_path / "budgets")


def test_dataset_report_is_independent_of_a_training_run(tmp_path):
    manifest = tmp_path / "splits" / "mini.json"
    manifest.parent.mkdir()
    write_json(
        manifest,
        {
            "version": 1,
            "dataset": "gtsrb",
            "seed": 42,
            "validation_fraction": 0.2,
            "num_classes": 2,
            "class_names": ["Alpha", "Beta"],
            "samples": {
                "train": [{"id": "a.png", "label": 0, "track": "0/1"}, {"id": "b.png", "label": 1, "track": "1/1"}],
                "validation": [{"id": "c.png", "label": 0, "track": "0/2"}],
                "test": [{"id": "d.png", "label": 1}],
            },
        },
    )
    result = generate_dataset_report(manifest, tmp_path / "dataset")
    names = {path.name for path in result["written"]}
    assert names == {"class_distribution.png", "dataset_summary.csv"}
    assert (tmp_path / "dataset" / "class_distribution.png").read_bytes().startswith(b"\x89PNG")
    rows = list(csv.DictReader((tmp_path / "dataset" / "dataset_summary.csv").open(encoding="utf-8")))
    assert rows[0]["name"] == "Alpha"
    assert rows[0]["train_images"] == "1"
    assert rows[-1]["class_id"] == "total"
    assert rows[-1]["test_images"] == "1"


def _write_invalid_augmentation_config(root: Path) -> None:
    directory = root / "configs"
    directory.mkdir(exist_ok=True)
    (directory / "augmentation.yaml").write_text(
        "experiment: augmentation\n"
        "seed: 1\n"
        "device: cpu\n"
        "data:\n"
        "  root: images\n"
        "  split_manifest: splits/mini.json\n"
        "  image_size: 48\n"
        "  num_classes: 2\n"
        "model:\n"
        "  channels: [32, 64]\n"
        "  kernel_size: 3\n"
        "  dropout: 0.2\n"
        "  min_spatial: 12\n"
        "training:\n"
        "  epochs: 1\n"
        "  batch_size: 4\n"
        "  learning_rate: 0.01\n"
        "  optimizer: adam\n"
        "augmentation:\n"
        "  rotation_degrees: 90\n"
        "  brightness: 0\n"
        "  zoom: 0\n",
        encoding="utf-8",
    )


def _factors():
    return {"rotation_degrees": 10, "brightness": 0.1, "zoom": 0.1}


def _write_image(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[:, :8] = (220, 30, 30)
    image[:, 8:] = (30, 30, 220)
    Image.fromarray(image).save(path)
    return path


def _write_run(
    root: Path,
    *,
    name: str,
    seed: int,
    accuracy: float,
    macro_f1: float,
    manifest_hash: str,
    augmentation: dict | None = None,
    status: str = "completed",
    epochs: int = 20,
    tag: str = "",
    channels: list[int] | None = None,
    trainable_parameters: int = 100,
) -> Path:
    manifest_path = root / "splits" / "mini.json"
    if not manifest_path.exists():
        write_json(
            manifest_path,
            {
                "version": 1,
                "dataset": "mini",
                "seed": 7,
                "validation_fraction": 0.5,
                "num_classes": 2,
                "samples": {
                    "train": [{"id": "sign.png", "label": 0, "track": "0/0001"}],
                    "validation": [{"id": "val.png", "label": 1, "track": "1/0001"}],
                    "test": [{"id": "test.png", "label": 0}],
                },
            },
        )
    run_dir = root / "runs" / f"{name}_{seed}_{epochs}_{manifest_hash}_{tag or accuracy}"
    run_dir.mkdir(parents=True)
    write_yaml(
        run_dir / "config.yaml",
        {
            "experiment": name,
            "seed": seed,
            "device": "cpu",
            "data": {
                "root": "images",
                "split_manifest": "splits/mini.json",
                "image_size": 16,
                "num_classes": 2,
            },
            "model": {
                "channels": channels or [4, 8],
                "kernel_size": 3,
                "dropout": 0.0,
                "dense_units": 8,
                "min_spatial": 8,
            },
            "training": {
                "epochs": epochs,
                "batch_size": 4,
                "learning_rate": 0.01,
                "optimizer": "adam",
            },
            "augmentation": augmentation or {"rotation_degrees": 10, "brightness": 0.1, "zoom": 0.1},
        },
    )
    write_json(
        run_dir / "metadata.json",
        {
            "run_id": run_dir.name,
            "experiment": name,
            "status": status,
            "seed": seed,
            "checkpoint_epoch": 2,
            "best_val_loss": 0.4,
            "duration_seconds": 12.5,
            "trainable_parameters": trainable_parameters,
            "split_manifest_sha256": manifest_hash,
            "checkpoint": "model.keras",
        },
    )
    scored = evaluate_predictions([0, 1, 1], [0, 1, 0], num_classes=2)
    scored["accuracy"] = accuracy
    scored["macro_f1"] = macro_f1
    scored["split"] = "validation"
    scored["source"] = "training_checkpoint"
    write_json(run_dir / "metrics.json", scored)
    (run_dir / "history.csv").write_text(
        "epoch,loss,accuracy,val_loss,val_accuracy\n"
        "1,1.000000,0.200000,0.900000,0.250000\n"
        "2,0.500000,0.600000,0.400000,0.750000\n",
        encoding="utf-8",
    )
    if not (root / "images" / "sign.png").exists():
        _write_image(root / "images" / "sign.png")
    return run_dir
