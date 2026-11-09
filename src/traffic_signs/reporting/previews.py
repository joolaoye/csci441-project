"""PNG previews of evaluation conditions, independent of any trained model."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from traffic_signs.data.labels import class_name_for_model
from traffic_signs.data.splits import load_manifest
from traffic_signs.data.transforms import load_rgb, preprocess_array
from traffic_signs.evaluation.conditions import (
    apply_condition,
    condition_fingerprint,
    condition_label,
    condition_notes,
    load_conditions,
)
from traffic_signs.reporting import ReportError
from traffic_signs.reporting.plots import save_figure
from traffic_signs.utils.runs import file_sha256, read_json, resolve_path, utc_now, write_json

PREVIEW_COUNT = 6


def generate_condition_previews(
    *,
    manifest_path: Path,
    data_root: Path,
    output_root: Path,
    split: str,
    image_size: int,
    conditions_path: Path,
    condition_names: list[str] | None = None,
    project_dir: Path | None = None,
) -> dict:
    """Write or reuse one preview per condition. This does not run a model."""
    if split not in ("validation", "test"):
        raise ReportError("Condition previews use the validation or test split.")
    project_dir = Path(project_dir or Path.cwd())
    manifest_path = resolve_path(manifest_path, project_dir)
    manifest = load_manifest(manifest_path)
    manifest_hash = file_sha256(manifest_path)
    samples = select_preview_samples(manifest["samples"][split], PREVIEW_COUNT)
    if not samples:
        raise ReportError(f"The {split} split has no images to preview.")
    root = resolve_path(data_root, project_dir)
    prepared = []
    for sample in samples:
        path = root / sample["id"]
        if not path.is_file():
            raise ReportError(f"Preview image is missing: {path}")
        prepared.append((sample, preprocess_array(load_rgb(path), image_size)))
    conditions = load_conditions(resolve_path(conditions_path, project_dir))
    names = list(conditions) if condition_names is None else condition_names
    missing = [name for name in names if name not in conditions]
    if missing:
        available = ", ".join(sorted(conditions))
        raise ReportError(f"Unknown condition(s): {', '.join(missing)}. Available: {available}.")
    written = []
    reused = []
    for name in names:
        condition = conditions[name]
        fingerprint = condition_fingerprint(
            condition,
            split=split,
            image_size=int(image_size),
            split_manifest_sha256=manifest_hash,
            sample_ids=[sample["id"] for sample, _image in prepared],
        )
        directory = Path(output_root) / split / fingerprint
        png_path = directory / "preview.png"
        metadata_path = directory / "metadata.json"
        if _reusable(metadata_path, png_path, fingerprint):
            reused.append(png_path)
            continue
        if metadata_path.is_file():
            raise ReportError(
                f"{directory} already holds a different preview. Refusing to overwrite it."
            )
        panels = preview_panels(prepared, condition, image_size)
        directory.mkdir(parents=True, exist_ok=True)
        _plot_preview(
            panels,
            manifest,
            condition,
            image_size,
            png_path,
        )
        metadata = {
            "fingerprint": fingerprint,
            "split": split,
            "split_manifest": str(manifest_path),
            "split_manifest_sha256": manifest_hash,
            "image_size": int(image_size),
            "condition": condition,
            "label": condition_label(condition, image_size),
            "notes": condition_notes(condition),
            "sample_ids": [sample["id"] for sample, _image in prepared],
            "created_at": utc_now(),
        }
        write_json(metadata_path, metadata)
        written.append(png_path)
    return {"written": written, "reused": reused, "output_root": Path(output_root)}


def select_preview_samples(samples: list[dict], count: int = PREVIEW_COUNT) -> list[dict]:
    """Pick a stable set of samples, preferring distinct class ids."""
    ordered = sorted(samples, key=lambda sample: sample["id"])
    chosen = []
    seen = set()
    for sample in ordered:
        label = int(sample["label"])
        if label in seen:
            continue
        chosen.append(sample)
        seen.add(label)
        if len(chosen) == count:
            return chosen
    chosen_ids = {sample["id"] for sample in chosen}
    for sample in ordered:
        if sample["id"] in chosen_ids:
            continue
        chosen.append(sample)
        if len(chosen) == count:
            break
    return chosen


def preview_panels(
    prepared: list[tuple[dict, np.ndarray]],
    condition: dict,
    image_size: int,
) -> list[tuple[dict, np.ndarray, np.ndarray]]:
    """Return ``(sample, preprocessed, transformed)`` using the evaluation transform."""
    panels = []
    for sample, image in prepared:
        transformed = apply_condition(image, condition, image_size)
        panels.append((sample, image, transformed))
    return panels


def _reusable(metadata_path: Path, png_path: Path, fingerprint: str) -> bool:
    if not metadata_path.is_file() or not png_path.is_file():
        return False
    metadata = read_json(metadata_path)
    return metadata.get("fingerprint") == fingerprint


def _plot_preview(panels, manifest, condition, image_size, path: Path) -> None:
    import textwrap

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = len(panels)
    figure, axes = plt.subplots(rows, 2, figsize=(7.2, 1.85 * rows + 0.9), squeeze=False)
    label = condition_label(condition, image_size)
    axes[0, 0].set_title("Preprocessed", fontsize=10)
    axes[0, 1].set_title("Transformed", fontsize=10)
    for row, (sample, original, transformed) in enumerate(panels):
        class_label = "\n".join(textwrap.wrap(_class_label(int(sample["label"]), manifest), 28))
        for column, image in ((0, original), (1, transformed)):
            axes[row, column].imshow(np.clip(image, 0.0, 1.0))
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
        axes[row, 0].set_ylabel(class_label, rotation=0, ha="right", va="center", fontsize=7, labelpad=12)
    figure.suptitle(f"{label}\n{condition_notes(condition)}", fontsize=10)
    figure.tight_layout(rect=(0.16, 0.0, 1.0, 0.96))
    save_figure(figure, path)


def _class_label(class_id: int, manifest: dict) -> str:
    name = None
    names = manifest.get("class_names") or []
    if class_id < len(names) and names[class_id]:
        name = names[class_id]
    else:
        name = class_name_for_model(class_id, int(manifest["num_classes"]))
    if name:
        return f"{class_id} {name}"
    return str(class_id)
