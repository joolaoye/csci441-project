"""The dataset commands require each person to pass their own extract."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"script_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prepare_train_and_evaluate_require_a_dataset_path():
    prepare = _script("prepare")
    with pytest.raises(SystemExit):
        prepare.parse_args([])
    prepared = prepare.parse_args(["--path", "/path/to/gtsrb"])
    assert prepared.path == "/path/to/gtsrb"

    train = _script("train")
    with pytest.raises(SystemExit):
        train.parse_args(["--config", "training.yaml.example"])
    trained = train.parse_args(["--config", "training.yaml.example", "--data-root", "/path/to/gtsrb"])
    assert trained.data_root == "/path/to/gtsrb"
    with pytest.raises(SystemExit):
        train.parse_args(
            ["--config", "training.yaml.example", "--data-root", "/path/to/gtsrb", "--run-name", "depth2"]
        )

    evaluate = _script("evaluate")
    with pytest.raises(SystemExit):
        evaluate.parse_args(["--checkpoint", "model.keras", "--split", "validation"])
    evaluated = evaluate.parse_args(
        ["--checkpoint", "model.keras", "--split", "validation", "--data-root", "/path/to/gtsrb"]
    )
    assert evaluated.data_root == "/path/to/gtsrb"
    assert evaluated.condition == "clean"

    report = _script("report")
    reported = report.parse_args(["run", "--run", "artifacts/example"])
    assert reported.data_root is None
    with pytest.raises(SystemExit):
        report.parse_args(["compare", "--runs", "artifacts/one", "artifacts/two"])
    compared = report.parse_args(
        ["compare", "--type", "depth", "--runs", "artifacts/one", "artifacts/two"]
    )
    assert compared.type == "depth"
