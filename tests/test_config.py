"""Configuration parsing and the example experiment file."""

from copy import deepcopy
from pathlib import Path

import pytest

from traffic_signs.utils.config import ConfigError, load_config, normalize_config, override_data_root

ROOT = Path(__file__).resolve().parents[1]

BASE = {
    "experiment": "baseline",
    "seed": 42,
    "device": "auto",
    "data": {
        "root": "data/gtsrb",
        "split_manifest": "configs/splits/gtsrb_v1.json",
        "image_size": 48,
        "num_classes": 43,
    },
    "model": {
        "channels": [32, 64],
        "kernel_size": 3,
        "dropout": 0.2,
        "dense_units": 128,
        "min_spatial": 12,
    },
    "training": {
        "epochs": 20,
        "batch_size": 64,
        "learning_rate": 0.001,
        "optimizer": "adam",
    },
    "augmentation": {"rotation_degrees": 0, "brightness": 0, "zoom": 0},
}


def make_config(**overrides):
    config = deepcopy(BASE)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            config[key].update(value)
        else:
            config[key] = value
    return config


def test_valid_config_is_accepted_and_idempotent():
    resolved = normalize_config(make_config())
    assert resolved["model"]["channels"] == [32, 64]
    assert resolved["training"]["optimizer"] == "adam"
    assert normalize_config(resolved) == resolved


def test_documented_defaults_are_filled_and_visible():
    config = make_config()
    del config["model"]["dense_units"]
    del config["model"]["min_spatial"]
    del config["training"]["optimizer"]
    del config["data"]["num_classes"]
    resolved = normalize_config(config)
    assert resolved["model"]["dense_units"] == 128
    assert resolved["model"]["min_spatial"] == 12
    assert resolved["training"]["optimizer"] == "adam"
    assert resolved["data"]["num_classes"] == 43


def test_shared_config_does_not_invent_a_dataset_directory():
    config = make_config()
    del config["data"]["root"]
    resolved = normalize_config(config)
    assert "root" not in resolved["data"]
    assert normalize_config(resolved) == resolved


def test_data_root_override_replaces_the_yaml_path_for_one_run():
    config = make_config()
    del config["data"]["root"]
    override_data_root(config, "~/gtsrb-extract")
    assert config["data"]["root"] == "~/gtsrb-extract"
    assert normalize_config(config)["data"]["root"] == "~/gtsrb-extract"
    with pytest.raises(ConfigError, match="data.root"):
        override_data_root(make_config(), "../outside")


def test_missing_augmentation_block_is_rejected():
    config = make_config()
    del config["augmentation"]
    with pytest.raises(ConfigError, match="augmentation"):
        normalize_config(config)


def test_unknown_setting_is_rejected():
    config = make_config()
    config["model"]["chanels"] = [32]
    with pytest.raises(ConfigError, match="chanels"):
        normalize_config(config)


def test_boolean_is_not_accepted_as_an_integer():
    with pytest.raises(ConfigError, match="epochs"):
        normalize_config(make_config(training={"epochs": True}))


def test_augmentation_limits_reject_transforms_that_would_change_a_sign():
    with pytest.raises(ConfigError, match="rotation_degrees"):
        normalize_config(make_config(augmentation={"rotation_degrees": 21, "brightness": 0, "zoom": 0}))
    allowed = normalize_config(
        make_config(augmentation={"rotation_degrees": 20, "brightness": 0.2, "zoom": 0.2})
    )
    assert allowed["augmentation"]["rotation_degrees"] == 20


def test_device_names_are_limited_to_auto_cpu_and_gpu():
    with pytest.raises(ConfigError, match="device"):
        normalize_config(make_config(device="cuda"))


def test_empty_channel_list_is_rejected():
    with pytest.raises(ConfigError, match="channels"):
        normalize_config(make_config(model={"channels": []}))


def test_example_config_loads_without_a_dataset_directory():
    example = load_config(ROOT / "training.yaml.example")
    assert "root" not in example["data"]
    assert example["experiment"] == "example"
    assert example["augmentation"] == {"rotation_degrees": 0.0, "brightness": 0.0, "zoom": 0.0}
    assert example["model"]["channels"] == [32, 64]
    assert example["data"]["num_classes"] == 43
    assert example["data"]["split_manifest"] == "configs/splits/gtsrb_v1.json"


def test_experiment_name_can_be_an_artifact_folder_name():
    config = normalize_config(make_config(experiment="depth2"))
    assert config["experiment"] == "depth2"
    with pytest.raises(ConfigError, match="experiment"):
        normalize_config(make_config(experiment="depth 2"))
