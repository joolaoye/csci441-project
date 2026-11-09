"""Read GTSRB and build a track-aware split.

The benchmark is the 43-class German Traffic Sign Recognition Benchmark.
Two extracts of that same benchmark are accepted:

* Kaggle ``meowmeowmeowmeowmeow/gtsrb-german-traffic-sign``, with
  ``Train.csv``, ``Test.csv``, ``Train/<class>/Class_Track_Frame.png``,
  and ``Test/<index>.png``.
* The official archive, with ``Final_Training/Images/00000/Track_Frame.ppm``
  and ``GT-final_test.csv``.

A training filename's track is the documented sequence id. Every frame of
that physical sign stays in one split. The official test images are copied
into the manifest and are not used for training or validation.

The split seed is ``--seed`` on ``scripts/prepare.py``. It is independent of
the training seed. Training reads the manifest and does not split again.
"""

from __future__ import annotations

import csv
import logging
import math
from pathlib import Path

import numpy as np
from PIL import Image

from traffic_signs.data.labels import CLASS_NAMES

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".png", ".ppm", ".jpg", ".jpeg"}
OFFICIAL_TRAIN_IMAGES = 39209
OFFICIAL_TEST_IMAGES = 12630


class DatasetError(RuntimeError):
    """The dataset or split manifest cannot be used as-is."""


def prepare_dataset(
    dataset_path: Path,
    manifest_path: Path,
    *,
    seed: int = 42,
    validation_fraction: float = 0.2,
    overwrite: bool = False,
    enforce_counts: bool = True,
) -> dict:
    """Read a local GTSRB extract, write the split manifest, and return a summary."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise DatasetError("--seed must be a non-negative integer.")
    if not 0 < validation_fraction <= 0.5:
        raise DatasetError("--val-fraction must be greater than 0 and at most 0.5.")
    root = Path(dataset_path).expanduser()
    manifest_path = Path(manifest_path)
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(
            f"Split manifest already exists at {manifest_path}. "
            "Refusing to regenerate it. Pass --overwrite to replace it."
        )

    if (root / "labels.csv").is_file() and (root / "DATA").is_dir():
        raise DatasetError(
            f"{root} is the 52-class traffic-sign extract, not GTSRB. "
            "Point --path at the 43-class German Traffic Sign Recognition Benchmark."
        )
    layout = detect_layout(root)
    logger.info("gtsrb_layout=%s", layout)
    if layout == "kaggle":
        training = read_kaggle_index(root, root / "Train.csv", tracked=True)
        testing = read_kaggle_index(root, root / "Test.csv", tracked=False)
    else:
        training = read_official_training(root)
        testing = read_official_test(root)
    if enforce_counts and (
        len(training) != OFFICIAL_TRAIN_IMAGES or len(testing) != OFFICIAL_TEST_IMAGES
    ):
        raise DatasetError(
            f"Expected {OFFICIAL_TRAIN_IMAGES} GTSRB training images and "
            f"{OFFICIAL_TEST_IMAGES} test images. Found {len(training)} training and "
            f"{len(testing)} test."
        )
    num_classes, class_names = _class_table(training, testing, enforce_counts=enforce_counts)

    train_samples, validation_samples = split_samples(
        training,
        validation_fraction=validation_fraction,
        seed=seed,
    )
    if not train_samples or not validation_samples:
        raise DatasetError(
            "The track split produced an empty training or validation set. "
            "Each class needs at least two image indexes, or the validation fraction "
            "needs to leave tracks on both sides."
        )
    manifest = build_manifest(
        train_samples,
        validation_samples,
        testing,
        seed=seed,
        validation_fraction=validation_fraction,
        num_classes=num_classes,
        class_names=class_names,
    )
    write_manifest(manifest_path, manifest, overwrite=overwrite)
    logger.info("dataset_root=%s", root)
    logger.info("example_id=%s", manifest["samples"]["train"][0]["id"])
    logger.info("counts=%s", manifest["counts"])
    for row in manifest["per_class"]:
        if row["train_images"] or row["validation_images"] or row["test_images"]:
            logger.info(
                "class %02d train_tracks=%d validation_tracks=%d train_images=%d validation_images=%d test_images=%d",
                row["label"],
                row["train_tracks"],
                row["validation_tracks"],
                row["train_images"],
                row["validation_images"],
                row["test_images"],
            )
    single_track = [
        row["label"]
        for row in manifest["per_class"]
        if row["train_tracks"] + row["validation_tracks"] == 1
    ]
    if single_track:
        logger.warning(
            "Classes with a single track stayed in training and have no validation tracks: %s",
            ", ".join(f"{label:02d}" for label in single_track),
        )
    return {
        "manifest": manifest_path,
        "dataset_root": root,
        "counts": manifest["counts"],
        "example_id": manifest["samples"]["train"][0]["id"],
        "manifest_body": manifest,
    }


def detect_layout(root: Path) -> str:
    """Return ``kaggle`` or ``official`` for a GTSRB extract."""
    if not root.is_dir():
        raise DatasetError(
            f"Dataset path does not exist: {root}. "
            "Point --path at an extracted GTSRB folder."
        )
    if (root / "Train.csv").is_file() and (root / "Test.csv").is_file():
        return "kaggle"
    if (root / "Final_Training" / "Images").is_dir() and (root / "GT-final_test.csv").is_file():
        return "official"
    raise DatasetError(
        f"{root} is not a GTSRB extract. Expected the Kaggle files Train.csv, Test.csv, "
        "Train/, and Test/, or the official Final_Training/Images directories and "
        "GT-final_test.csv."
    )


def track_token(filename: str, class_id: int) -> str:
    """Return the GTSRB track id encoded in one training filename.

    Kaggle training names are ``ClassId_Track_Frame.png``. Official training
    names are ``Track_Frame.ppm``. The track is only unique inside a class.
    """
    parts = Path(filename).stem.split("_")
    if not parts or not all(part.isdigit() for part in parts):
        raise DatasetError(
            f"Cannot recover a GTSRB track from {filename}. "
            "Expected ClassId_Track_Frame.png or Track_Frame.ppm."
        )
    if len(parts) == 3:
        if int(parts[0]) != class_id:
            raise DatasetError(
                f"{filename} starts with class {int(parts[0])}, but the annotation says class {class_id}."
            )
        return parts[1]
    if len(parts) == 2:
        return parts[0]
    raise DatasetError(
        f"Cannot recover a GTSRB track from {filename}. "
        "Expected ClassId_Track_Frame.png or Track_Frame.ppm."
    )


def read_kaggle_index(root: Path, csv_path: Path, *, tracked: bool) -> list[dict]:
    """Read ``Train.csv`` or ``Test.csv`` from the Kaggle GTSRB extract."""
    fieldnames, rows = _read_csv(csv_path, ",")
    if "ClassId" not in fieldnames or "Path" not in fieldnames:
        raise DatasetError(f"{csv_path} must have ClassId and Path columns.")
    samples = []
    seen = set()
    for row in rows:
        class_id = _class_id(row.get("ClassId"), csv_path)
        relative = _relative_id(row.get("Path"), csv_path)
        path = root / relative
        _require_image(path)
        if relative in seen:
            raise DatasetError(f"{csv_path} lists {relative} more than once.")
        seen.add(relative)
        sample = {"id": relative, "label": class_id}
        if tracked:
            sample["track"] = f"{class_id}/{track_token(Path(relative).name, class_id)}"
        samples.append(sample)
    if not samples:
        raise DatasetError(f"{csv_path} does not list any images.")
    samples.sort(key=lambda sample: sample["id"])
    return samples


def read_official_training(root: Path) -> list[dict]:
    """Read the official ``Final_Training/Images`` class folders."""
    images = root / "Final_Training" / "Images"
    class_dirs = [path for path in images.iterdir() if path.is_dir() and not path.name.startswith(".")]
    samples = []
    seen = set()
    for class_dir in sorted(class_dirs, key=lambda path: path.name):
        if len(class_dir.name) != 5 or not class_dir.name.isdigit():
            raise DatasetError(f"Unexpected training class folder: {class_dir.name}")
        class_id = int(class_dir.name)
        csv_path = class_dir / f"GT-{class_dir.name}.csv"
        fieldnames, rows = _read_csv(csv_path, ";")
        if "Filename" not in fieldnames or "ClassId" not in fieldnames:
            raise DatasetError(f"{csv_path} must have Filename and ClassId columns.")
        for row in rows:
            filename = (row.get("Filename") or "").strip()
            annotated = _class_id(row.get("ClassId"), csv_path)
            if annotated != class_id:
                raise DatasetError(
                    f"{csv_path} assigns {filename} to class {annotated}, but it is in folder {class_dir.name}."
                )
            path = class_dir / filename
            _require_image(path)
            relative = path.resolve().relative_to(root.resolve()).as_posix()
            if relative in seen:
                raise DatasetError(f"Duplicate training image {relative}.")
            seen.add(relative)
            samples.append(
                {
                    "id": relative,
                    "label": class_id,
                    "track": f"{class_id}/{track_token(filename, class_id)}",
                }
            )
    if not samples:
        raise DatasetError(f"{images} does not contain training images.")
    samples.sort(key=lambda sample: sample["id"])
    return samples


def read_official_test(root: Path) -> list[dict]:
    """Read the official held-out test images. They are not assigned to tracks."""
    csv_path = root / "GT-final_test.csv"
    fieldnames, rows = _read_csv(csv_path, ";")
    if "Filename" not in fieldnames or "ClassId" not in fieldnames:
        raise DatasetError(f"{csv_path} must have Filename and ClassId columns.")
    samples = []
    seen = set()
    for row in rows:
        filename = (row.get("Filename") or "").strip()
        class_id = _class_id(row.get("ClassId"), csv_path)
        path = root / "Final_Test" / "Images" / filename
        _require_image(path)
        relative = path.resolve().relative_to(root.resolve()).as_posix()
        if relative in seen:
            raise DatasetError(f"Duplicate test image {relative}.")
        seen.add(relative)
        samples.append({"id": relative, "label": class_id})
    if not samples:
        raise DatasetError(f"{csv_path} does not list any test images.")
    samples.sort(key=lambda sample: sample["id"])
    return samples


def _class_table(training: list[dict], testing: list[dict], *, enforce_counts: bool):
    labels = {int(sample["label"]) for sample in training + testing}
    if not labels or min(labels) < 0 or labels != set(range(max(labels) + 1)):
        raise DatasetError("GTSRB class ids must be contiguous and start at 0.")
    num_classes = max(labels) + 1
    if enforce_counts:
        if num_classes != len(CLASS_NAMES):
            raise DatasetError(f"GTSRB has {len(CLASS_NAMES)} classes. Found {num_classes}.")
        return num_classes, list(CLASS_NAMES)
    return num_classes, None


def _read_csv(path: Path, delimiter: str) -> tuple[list[str], list[dict]]:
    path = Path(path)
    if not path.is_file():
        raise DatasetError(f"Missing annotation file: {path}")
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            fieldnames = list(reader.fieldnames or [])
            rows = list(reader)
    except OSError as exc:
        raise DatasetError(f"Could not read {path}: {exc}") from exc
    return fieldnames, rows


def _class_id(value, source: Path) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise DatasetError(f"{source} has a non-integer ClassId {value!r}.") from exc


def _relative_id(value, source: Path) -> str:
    text = str(value or "").strip().replace("\\", "/")
    path = Path(text)
    if not text or path.is_absolute() or ".." in path.parts:
        raise DatasetError(f"{source} has a path that is not inside the dataset: {value!r}.")
    return path.as_posix()


def _require_image(path: Path) -> None:
    if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
        raise DatasetError(f"Missing or unsupported image file: {path}")
    if path.stat().st_size == 0:
        raise DatasetError(f"Empty image file: {path}")
    _verify_image(path)


def split_samples(
    samples: list[dict],
    validation_fraction: float,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    """Assign whole tracks to training or validation.

    Tracks are shuffled inside each class with ``numpy.random.default_rng(seed)``.
    The global NumPy seed is not used. The number of validation tracks is the
    class track count times ``validation_fraction``, rounded half up, then
    clamped so a class with two or more tracks keeps at least one track on
    each side. A class with a single track stays in training.
    """
    if not 0 < validation_fraction <= 0.5:
        raise DatasetError("validation_fraction must be greater than 0 and at most 0.5.")
    groups: dict[tuple[int, str], list[dict]] = {}
    for sample in samples:
        key = (int(sample["label"]), str(sample["track"]))
        groups.setdefault(key, []).append(sample)

    by_class: dict[int, list[tuple[int, str]]] = {}
    for key in groups:
        by_class.setdefault(key[0], []).append(key)
    for label in by_class:
        by_class[label].sort()

    rng = np.random.default_rng(seed)
    train_keys: list[tuple[int, str]] = []
    validation_keys: list[tuple[int, str]] = []
    for label in sorted(by_class):
        keys = by_class[label]
        order = rng.permutation(len(keys))
        n_validation = validation_track_count(len(keys), validation_fraction)
        chosen = {keys[int(index)] for index in order[:n_validation]}
        for key in keys:
            if key in chosen:
                validation_keys.append(key)
            else:
                train_keys.append(key)

    return _expand(groups, train_keys), _expand(groups, validation_keys)


def validation_track_count(n_tracks: int, validation_fraction: float) -> int:
    """How many tracks from one class go to validation."""
    if n_tracks <= 1:
        return 0
    rounded = int(math.floor(n_tracks * validation_fraction + 0.5))
    return min(max(rounded, 1), n_tracks - 1)


def build_manifest(
    train_samples: list[dict],
    validation_samples: list[dict],
    test_samples: list[dict],
    *,
    seed: int,
    validation_fraction: float,
    num_classes: int,
    class_names: list[str] | None = None,
) -> dict:
    """Assemble the JSON-ready split manifest. Paths stay dataset-relative."""
    _reject_duplicate_ids(train_samples, validation_samples, test_samples)
    _reject_shared_tracks(train_samples, validation_samples)
    if class_names is not None and len(class_names) != num_classes:
        raise DatasetError("class_names must contain one entry per class.")
    manifest = {
        "version": 1,
        "dataset": "gtsrb",
        "seed": seed,
        "validation_fraction": validation_fraction,
        "num_classes": num_classes,
        "split_unit": "track",
        "paths_are_relative_to": "data.root",
        "counts": _counts(train_samples, validation_samples, test_samples),
        "per_class": _per_class(train_samples, validation_samples, test_samples, num_classes),
        "samples": {
            "train": [_public_sample(sample, include_track=True) for sample in train_samples],
            "validation": [
                _public_sample(sample, include_track=True) for sample in validation_samples
            ],
            "test": [_public_sample(sample, include_track=False) for sample in test_samples],
        },
    }
    if class_names is not None:
        ordered = {
            "version": manifest["version"],
            "dataset": manifest["dataset"],
            "seed": manifest["seed"],
            "validation_fraction": manifest["validation_fraction"],
            "num_classes": manifest["num_classes"],
            "class_names": list(class_names),
        }
        ordered.update({key: value for key, value in manifest.items() if key not in ordered})
        manifest = ordered
    return manifest


def write_manifest(path: Path, manifest: dict, overwrite: bool) -> None:
    """Write the manifest. An existing file is left untouched unless overwrite is set."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Split manifest already exists at {path}. "
            "Refusing to regenerate it. Pass --overwrite to replace it."
        )
    from traffic_signs.utils.runs import write_json

    write_json(path, manifest)


def load_manifest(path: Path) -> dict:
    """Load a split manifest and check that its splits do not overlap."""
    path = Path(path)
    if not path.is_file():
        raise DatasetError(
            f"Split manifest not found: {path}. Run scripts/prepare.py before training."
        )
    from traffic_signs.utils.runs import read_json

    try:
        manifest = read_json(path)
    except (OSError, ValueError) as exc:
        raise DatasetError(f"Could not read split manifest {path}: {exc}") from exc
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: dict) -> None:
    """Raise ``DatasetError`` when the manifest schema or split grouping is wrong."""
    if not isinstance(manifest, dict):
        raise DatasetError("The split manifest must be a JSON object.")
    for key in ("version", "dataset", "seed", "validation_fraction", "num_classes", "samples"):
        if key not in manifest:
            raise DatasetError(f"The split manifest is missing '{key}'.")
    if manifest["version"] != 1:
        raise DatasetError(f"Unsupported split manifest version {manifest['version']!r}.")
    num_classes = manifest["num_classes"]
    if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 2:
        raise DatasetError("The split manifest has an invalid num_classes.")
    class_names = manifest.get("class_names")
    if class_names is not None:
        if not isinstance(class_names, list) or len(class_names) != num_classes:
            raise DatasetError("class_names must contain one name per class.")
        if not all(isinstance(name, str) and name for name in class_names):
            raise DatasetError("Every class name must be a non-empty string.")
    samples = manifest["samples"]
    if not isinstance(samples, dict):
        raise DatasetError("The split manifest is missing the samples mapping.")
    for split in ("train", "validation", "test"):
        if split not in samples or not isinstance(samples[split], list):
            raise DatasetError(f"The split manifest is missing the '{split}' sample list.")
        for sample in samples[split]:
            _validate_sample(sample, split, num_classes)
    _reject_duplicate_ids(samples["train"], samples["validation"], samples["test"])
    _reject_shared_tracks(samples["train"], samples["validation"])


def _verify_image(path: Path) -> None:
    try:
        with Image.open(path) as image:
            image.verify()
    except (OSError, ValueError) as exc:
        raise DatasetError(f"Cannot read image {path}: {exc}") from exc


def _expand(groups: dict, keys: list) -> list[dict]:
    selected = []
    for key in keys:
        selected.extend(groups[key])
    selected.sort(key=lambda sample: sample["id"])
    return selected


def _public_sample(sample: dict, include_track: bool) -> dict:
    record = {"id": sample["id"], "label": int(sample["label"])}
    if include_track:
        record["track"] = sample["track"]
    return record


def _counts(train_samples: list[dict], validation_samples: list[dict], test_samples: list[dict]) -> dict:
    return {
        "train_images": len(train_samples),
        "validation_images": len(validation_samples),
        "test_images": len(test_samples),
        "train_tracks": len({sample["track"] for sample in train_samples}),
        "validation_tracks": len({sample["track"] for sample in validation_samples}),
    }


def _per_class(
    train_samples: list[dict],
    validation_samples: list[dict],
    test_samples: list[dict],
    num_classes: int,
) -> list[dict]:
    rows = []
    for label in range(num_classes):
        train = [sample for sample in train_samples if sample["label"] == label]
        validation = [sample for sample in validation_samples if sample["label"] == label]
        test = [sample for sample in test_samples if sample["label"] == label]
        rows.append(
            {
                "label": label,
                "train_tracks": len({sample["track"] for sample in train}),
                "validation_tracks": len({sample["track"] for sample in validation}),
                "train_images": len(train),
                "validation_images": len(validation),
                "test_images": len(test),
            }
        )
    return rows


def _reject_duplicate_ids(*groups: list[dict]) -> None:
    seen = set()
    for group in groups:
        for sample in group:
            sample_id = sample["id"]
            if sample_id in seen:
                raise DatasetError(f"Image id {sample_id} appears in more than one split or twice in one split.")
            seen.add(sample_id)


def _reject_shared_tracks(train_samples: list[dict], validation_samples: list[dict]) -> None:
    train_tracks = {sample["track"] for sample in train_samples}
    validation_tracks = {sample["track"] for sample in validation_samples}
    shared = sorted(train_tracks & validation_tracks)
    if shared:
        preview = ", ".join(shared[:5])
        raise DatasetError(
            "A traffic-sign track appears in both training and validation "
            f"({preview}). Related frames must stay in one split."
        )


def _validate_sample(sample, split: str, num_classes: int) -> None:
    if not isinstance(sample, dict):
        raise DatasetError(f"A {split} sample is not an object.")
    sample_id = sample.get("id")
    if not isinstance(sample_id, str) or not sample_id:
        raise DatasetError(f"A {split} sample is missing its id.")
    path = Path(sample_id)
    if "\\" in sample_id or path.is_absolute() or ".." in path.parts:
        raise DatasetError(
            f"Sample id {sample_id!r} must be a dataset-relative path with forward slashes and without '..'."
        )
    label = sample.get("label")
    if isinstance(label, bool) or not isinstance(label, int) or label < 0 or label >= num_classes:
        raise DatasetError(f"Sample {sample_id} has a label outside 0..{num_classes - 1}.")
    if split in ("train", "validation"):
        track = sample.get("track")
        if not isinstance(track, str) or not track:
            raise DatasetError(f"Training sample {sample_id} is missing its track id.")
