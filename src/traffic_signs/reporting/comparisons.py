"""Comparison tables for completed runs that share an evaluation protocol."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np

from traffic_signs.reporting import ReportError
from traffic_signs.reporting.plots import TRAIN_COLOR, VAL_COLOR, save_figure
from traffic_signs.utils.config import ConfigError, load_config
from traffic_signs.utils.runs import read_json, utc_stamp

METRIC_FIELDS = (
    ("validation_accuracy", "validation_accuracy_std"),
    ("macro_f1", "macro_f1_std"),
    ("best_val_loss", "best_val_loss_std"),
)

CSV_FIELDS = [
    "run_id",
    "seed",
    "comparison_type",
    "configuration",
    "depth",
    "channels",
    "kernel_size",
    "dropout",
    "dense_units",
    "trainable_parameters",
    "rotation_degrees",
    "brightness",
    "zoom",
    "epochs",
    "batch_size",
    "learning_rate",
    "checkpoint_epoch",
    "duration_seconds",
    "validation_accuracy",
    "macro_f1",
    "best_val_loss",
    "split_manifest_sha256",
]

COMPARISON_TYPES = ("augmentation", "depth")
PROTOCOL_FIELDS = (
    ("epochs", "epoch budgets"),
    ("batch_size", "batch sizes"),
    ("learning_rate", "learning rates"),
    ("image_size", "image sizes"),
    ("num_classes", "class counts"),
)
ARCHITECTURE_FIELDS = (
    ("channels", "channel lists"),
    ("kernel_size", "kernel sizes"),
    ("dropout", "dropout rates"),
    ("dense_units", "dense-unit counts"),
    ("min_spatial", "minimum spatial sizes"),
)


def compare_runs(
    run_dirs: list[Path],
    comparison_type: str,
    output_root: Path,
    conditions: list[str] | None = None,
) -> dict:
    """Write the CSV, LaTeX table, and comparison figure for one research question.

    ``conditions`` adds a robustness table for either research question. The
    evaluation files must already exist. This command does not train or
    evaluate a model. Checkpoint selection stays on clean validation accuracy.
    """
    if comparison_type not in COMPARISON_TYPES:
        raise ReportError("comparison type must be 'augmentation' or 'depth'.")
    summaries = [load_run_summary(Path(run_dir)) for run_dir in run_dirs]
    check_compatible(summaries, comparison_type)
    if conditions:
        return _compare_conditions(summaries, conditions, output_root, comparison_type)
    rows, warnings = comparison_rows(summaries, comparison_type)
    comparison_id = _comparison_id(summaries)
    output_dir = Path(output_root) / comparison_type / comparison_id
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "comparison.csv"
    latex_path = output_dir / "comparison.tex"
    _write_csv(csv_path, summaries, comparison_type)
    _write_latex(latex_path, rows, comparison_type, comparison_id)
    figure_paths = plot_comparison(rows, comparison_type, output_dir / "comparison")
    return {
        "csv": csv_path,
        "latex": latex_path,
        "png": figure_paths[0],
        "rows": rows,
        "warnings": warnings,
        "comparison_id": comparison_id,
        "output_dir": output_dir,
    }


def load_run_summary(run_dir: Path) -> dict:
    """Read the configuration, metadata, and validation metrics needed for a table."""
    run_dir = Path(run_dir)
    config_path = run_dir / "config.yaml"
    metadata_path = run_dir / "metadata.json"
    metrics_path = run_dir / "metrics.json"
    if not config_path.is_file() or not metadata_path.is_file() or not metrics_path.is_file():
        raise ReportError(
            f"{run_dir} is missing config.yaml, metadata.json, or metrics.json. "
            "Compare completed training runs."
        )
    try:
        config = load_config(config_path)
    except ConfigError as exc:
        detail = str(exc)
        if "num_classes" in detail:
            raise ReportError(
                f"{run_dir.name} is not a 43-class GTSRB run ({detail}). "
                "Exploratory runs from the earlier 52-class dataset cannot be compared "
                "with GTSRB experiments."
            ) from exc
        raise ReportError(f"{run_dir.name} has an unreadable configuration: {detail}") from exc
    metadata = read_json(metadata_path)
    metrics = read_json(metrics_path)
    if metadata.get("status") != "completed":
        raise ReportError(
            f"Run {metadata.get('run_id', run_dir.name)} has status "
            f"{metadata.get('status')!r}. Only completed runs can be compared."
        )
    if metadata.get("checkpoint_epoch") is None or metadata.get("best_val_loss") is None:
        raise ReportError(
            f"Run {metadata.get('run_id', run_dir.name)} has no selected checkpoint. "
            "Only completed runs with a saved validation checkpoint can be compared."
        )
    channels = [int(channel) for channel in config["model"]["channels"]]
    return {
        "run_id": str(metadata.get("run_id") or run_dir.name),
        "experiment": str(config["experiment"]),
        "seed": int(config["seed"]),
        "channels": channels,
        "depth": len(channels),
        "kernel_size": int(config["model"]["kernel_size"]),
        "dropout": float(config["model"]["dropout"]),
        "dense_units": int(config["model"]["dense_units"]),
        "min_spatial": int(config["model"]["min_spatial"]),
        "rotation_degrees": float(config["augmentation"]["rotation_degrees"]),
        "brightness": float(config["augmentation"]["brightness"]),
        "zoom": float(config["augmentation"]["zoom"]),
        "epochs": int(config["training"]["epochs"]),
        "batch_size": int(config["training"]["batch_size"]),
        "learning_rate": float(config["training"]["learning_rate"]),
        "image_size": int(config["data"]["image_size"]),
        "num_classes": int(config["data"]["num_classes"]),
        "trainable_parameters": metadata.get("trainable_parameters"),
        "checkpoint_epoch": metadata.get("checkpoint_epoch"),
        "best_val_loss": _optional_float(metadata.get("best_val_loss")),
        "duration_seconds": _optional_float(metadata.get("duration_seconds")),
        "validation_accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "split": metrics.get("split"),
        "split_manifest": str(config["data"]["split_manifest"]),
        "split_manifest_sha256": metadata.get("split_manifest_sha256"),
        "run_dir": run_dir,
    }


def check_compatible(summaries: list[dict], comparison_type: str) -> None:
    """Reject runs that do not answer the selected research question."""
    if len(summaries) < 2:
        raise ReportError("A comparison needs at least two completed runs.")
    problems = []
    hashes = {summary["split_manifest_sha256"] for summary in summaries}
    manifests = {summary["split_manifest"] for summary in summaries}
    if None in hashes or len(hashes) != 1 or len(manifests) != 1:
        problems.append(
            "These runs do not share one split manifest "
            f"({_preview(manifests)}; hashes {_preview(hashes)}). "
            "Do not mix runs from different datasets or split manifests."
        )
    for field, label in PROTOCOL_FIELDS:
        values = {summary[field] for summary in summaries}
        if len(values) != 1:
            problems.append(f"These runs use different {label}: {_preview(values)}.")
    if any(summary["split"] != "validation" for summary in summaries):
        problems.append("Comparison requires validation metrics from the selected checkpoint.")
    if comparison_type == "augmentation":
        for field, label in ARCHITECTURE_FIELDS:
            values = {_hashable(summary[field]) for summary in summaries}
            if len(values) != 1:
                problems.append(
                    f"RQ1 keeps the architecture fixed. These runs use different {label}: {_preview(values)}."
                )
        if not any(_is_baseline(summary) for summary in summaries):
            problems.append("RQ1 needs a baseline run with rotation, brightness, and zoom all set to 0.")
        if all(_is_baseline(summary) for summary in summaries):
            problems.append("RQ1 needs at least one run with augmentation enabled.")
    else:
        for field, label in (
            ("rotation_degrees", "rotation settings"),
            ("brightness", "brightness settings"),
            ("zoom", "zoom settings"),
        ):
            values = {summary[field] for summary in summaries}
            if len(values) != 1:
                problems.append(
                    f"RQ2 keeps augmentation fixed. These runs use different {label}: {_preview(values)}."
                )
        if len({summary["depth"] for summary in summaries}) < 2:
            problems.append("RQ2 needs at least two convolutional depths.")
    if problems:
        raise ReportError(" ".join(problems))


def comparison_rows(summaries: list[dict], comparison_type: str) -> tuple[list[dict], list[str]]:
    """One row per configuration. Mean and sample standard deviation require two or more seeds."""
    groups: dict[tuple, list[dict]] = {}
    for summary in summaries:
        groups.setdefault(_group_key(summary, comparison_type), []).append(summary)
    rows = []
    warnings = []
    for group in groups.values():
        group.sort(key=lambda summary: summary["run_id"])
        seeds = [summary["seed"] for summary in group]
        if len(seeds) != len(set(seeds)):
            warnings.append(
                f"{group[0]['experiment']} repeats seed(s) {sorted(set(seeds))}. "
                "Standard deviation is omitted for repeated seeds."
            )
            varied = False
        else:
            varied = len(group) >= 2
        first = group[0]
        row = {
            "experiment": first["experiment"],
            "run_ids": ";".join(summary["run_id"] for summary in group),
            "seeds": ";".join(str(seed) for seed in seeds),
            "n_runs": len(group),
            "depth": first["depth"],
            "channels": ",".join(str(channel) for channel in first["channels"]),
            "trainable_parameters": _shared_value(group, "trainable_parameters"),
            "rotation_degrees": first["rotation_degrees"],
            "brightness": first["brightness"],
            "zoom": first["zoom"],
            "epochs": first["epochs"],
            "batch_size": first["batch_size"],
            "learning_rate": first["learning_rate"],
            "checkpoint_epochs": ";".join(str(summary["checkpoint_epoch"]) for summary in group),
            "duration_seconds": _mean([summary["duration_seconds"] for summary in group]),
        }
        for source, std_field in METRIC_FIELDS:
            values = [summary[source] for summary in group]
            row[source] = float(np.mean(values))
            row[std_field] = _sample_std(values) if varied else None
        row["label"] = _configuration_label(first, comparison_type)
        row["comparison_type"] = comparison_type
        rows.append(row)
    if comparison_type == "augmentation":
        rows.sort(key=lambda row: (not _row_is_baseline(row), row["label"]))
    else:
        rows.sort(key=lambda row: (row["depth"], row["channels"]))
    return rows, warnings


def _compare_conditions(
    summaries: list[dict],
    conditions: list[str],
    output_root: Path,
    comparison_type: str,
) -> dict:
    """Compare saved evaluation conditions for depth or training augmentation."""
    requested = list(dict.fromkeys(conditions))
    if not requested:
        raise ReportError("Pass at least one evaluation condition.")
    selected = [_select_condition_evaluations(summary, requested) for summary in summaries]
    _check_condition_compatibility(selected, requested)
    records = []
    for summary, evaluations in zip(summaries, selected, strict=True):
        clean = evaluations.get("clean")
        for name in requested:
            evaluation = evaluations[name]
            accuracy = float(evaluation["accuracy"])
            drop = None if clean is None or name == "clean" else float(clean["accuracy"]) - accuracy
            records.append(
                {
                    **summary,
                    "configuration": _condition_configuration(summary, comparison_type),
                    "condition": name,
                    "condition_label": evaluation["condition"]["label"],
                    "condition_type": evaluation["condition"]["type"],
                    "condition_parameters": _parameter_text(evaluation["condition"].get("parameters") or {}),
                    "evaluation_id": evaluation.get("evaluation_id"),
                    "evaluation_created_at": evaluation.get("created_at"),
                    "evaluation_split": evaluation.get("split"),
                    "sample_ids_sha256": evaluation.get("sample_ids_sha256"),
                    "condition_accuracy": accuracy,
                    "condition_macro_f1": float(evaluation["macro_f1"]),
                    "condition_loss": float(evaluation["loss"]),
                    "accuracy_drop_from_clean": drop,
                }
            )
    grouped, warnings = _condition_rows(records, requested, comparison_type)
    identity = _comparison_id(summaries) + "_" + _condition_digest(selected, requested)
    output_dir = Path(output_root) / comparison_type / identity
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "comparison.csv"
    latex_path = output_dir / "comparison.tex"
    _write_condition_csv(csv_path, records, comparison_type)
    _write_condition_latex(latex_path, grouped, requested, identity, comparison_type)
    figure_paths = plot_condition_comparison(
        grouped, requested, output_dir / "comparison", comparison_type
    )
    return {
        "csv": csv_path,
        "latex": latex_path,
        "png": figure_paths[0],
        "rows": grouped,
        "warnings": warnings,
        "comparison_id": identity,
        "output_dir": output_dir,
    }


def _select_condition_evaluations(summary: dict, conditions: list[str]) -> dict[str, dict]:
    run_dir = Path(summary["run_dir"])
    directory = run_dir / "evaluations"
    files = sorted(directory.glob("*.json")) if directory.is_dir() else []
    chosen = {}
    for name in conditions:
        matches = []
        for path in files:
            payload = read_json(path)
            condition = payload.get("condition") or {}
            if payload.get("split") == "validation" and condition.get("name") == name:
                payload = dict(payload)
                payload["_path"] = path
                matches.append(payload)
        if not matches:
            raise ReportError(
                f"{summary['run_id']} has no validation evaluation for condition {name!r}. "
                "Run scripts/evaluate.py for that condition before comparing."
            )
        signatures = {_condition_signature(payload) for payload in matches}
        if len(matches) > 1 and len(signatures) > 1:
            rendered = "; ".join(sorted(signatures))
            raise ReportError(
                f"{summary['run_id']} has more than one {name!r} evaluation with different "
                f"parameters ({rendered}). Select one by removing the extra evaluation file. "
                "A condition name is not chosen automatically."
            )
        if len(matches) > 1:
            rendered = ", ".join(path.name for path in (item["_path"] for item in matches))
            raise ReportError(
                f"{summary['run_id']} has multiple {name!r} evaluations ({rendered}). "
                "Remove the extra file so the comparison has one result."
            )
        chosen[name] = matches[0]
    return chosen


def _check_condition_compatibility(selected: list[dict[str, dict]], conditions: list[str]) -> None:
    problems = []
    for name in conditions:
        evaluations = [item[name] for item in selected]
        manifests = {item.get("split_manifest_sha256") for item in evaluations}
        samples = {item.get("sample_ids_sha256") for item in evaluations}
        parameters = {_condition_signature(item) for item in evaluations}
        image_sizes = {item.get("image_size") for item in evaluations}
        if len(manifests) != 1 or None in manifests:
            problems.append(f"{name} evaluations do not share one split manifest.")
        if len(samples) != 1 or None in samples:
            problems.append(f"{name} evaluations do not use the same image ids.")
        if len(parameters) != 1:
            problems.append(f"{name} evaluations use different parameters: {', '.join(sorted(parameters))}.")
        if len(image_sizes) != 1:
            problems.append(f"{name} evaluations use different model image sizes.")
    if problems:
        raise ReportError(" ".join(problems))


def _condition_signature(evaluation: dict) -> str:
    condition = evaluation.get("condition") or {}
    return _parameter_text(condition.get("parameters") or {}) or condition.get("type") or "unspecified"


def _parameter_text(parameters: dict) -> str:
    if not parameters:
        return "none"
    return ", ".join(f"{key}={parameters[key]}" for key in sorted(parameters))


def _condition_rows(
    records: list[dict],
    conditions: list[str],
    comparison_type: str,
) -> tuple[list[dict], list[str]]:
    groups: dict[tuple, list[str]] = {}
    for record in records:
        key = _condition_group_key(record, comparison_type)
        groups.setdefault(key, [])
        if record["run_id"] not in groups[key]:
            groups[key].append(record["run_id"])
    rows = []
    warnings = []
    for key, run_ids in groups.items():
        members = [record for record in records if _condition_group_key(record, comparison_type) == key]
        seeds = [next(record["seed"] for record in members if record["run_id"] == run_id) for run_id in run_ids]
        varied = len(seeds) == len(set(seeds)) and len(run_ids) >= 2
        if len(seeds) != len(set(seeds)):
            name = _condition_configuration(members[0], comparison_type)
            warnings.append(
                f"{name} repeats seed(s) {sorted(set(seeds))}. "
                "Standard deviation is omitted for repeated seeds."
            )
        first = next(record for record in members if record["run_id"] == run_ids[0])
        channel_text = ", ".join(str(channel) for channel in _channel_key(first["channels"]))
        row = {
            "depth": first["depth"],
            "channels": channel_text,
            "rotation_degrees": first["rotation_degrees"],
            "brightness": first["brightness"],
            "zoom": first["zoom"],
            "label": _condition_configuration(first, comparison_type),
            "trainable_parameters": first["trainable_parameters"],
            "checkpoint_epoch": first["checkpoint_epoch"],
            "accuracies": {},
            "accuracy_stds": {},
            "labels": {},
        }
        for name in conditions:
            values = [float(record["condition_accuracy"]) for record in members if record["condition"] == name]
            row["accuracies"][name] = float(np.mean(values))
            row["accuracy_stds"][name] = _sample_std(values) if varied else None
            row["labels"][name] = next(record["condition_label"] for record in members if record["condition"] == name)
        rows.append(row)
    if comparison_type == "depth":
        counts: dict[int, int] = {}
        for row in rows:
            counts[row["depth"]] = counts.get(row["depth"], 0) + 1
        for row in rows:
            row["label"] = str(row["depth"]) if counts[row["depth"]] == 1 else f"{row['depth']} ({row['channels']})"
        rows.sort(key=lambda row: (row["depth"], row["channels"]))
    else:
        rows.sort(key=_strategy_sort)
    return rows, warnings


def _condition_group_key(record: dict, comparison_type: str) -> tuple:
    if comparison_type == "augmentation":
        return (record["rotation_degrees"], record["brightness"], record["zoom"])
    return (record["depth"], _channel_key(record["channels"]))


def _condition_configuration(record: dict, comparison_type: str) -> str:
    if comparison_type == "augmentation":
        return _strategy_label(record)
    return str(record["depth"])


def _channel_key(channels) -> tuple:
    if isinstance(channels, str):
        return tuple(part.strip() for part in channels.split(",") if part.strip())
    return tuple(channels)


def _write_condition_csv(path: Path, records: list[dict], comparison_type: str) -> None:
    fields = [
        "run_id",
        "seed",
        "configuration",
        "depth",
        "channels",
        "trainable_parameters",
        "checkpoint_epoch",
        "best_val_loss",
        "epochs",
        "batch_size",
        "learning_rate",
        "image_size",
        "rotation_degrees",
        "brightness",
        "zoom",
        "condition",
        "condition_label",
        "condition_type",
        "condition_parameters",
        "evaluation_id",
        "evaluation_created_at",
        "evaluation_split",
        "sample_ids_sha256",
        "split_manifest_sha256",
        "condition_accuracy",
        "condition_macro_f1",
        "condition_loss",
        "accuracy_drop_from_clean",
    ]
    ordered = sorted(
        records,
        key=lambda record: (
            _strategy_sort(record) if comparison_type == "augmentation" else (record["depth"], str(record["channels"])),
            record["condition"],
            record["run_id"],
        ),
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in ordered:
            row = {field: record.get(field) for field in fields}
            row["channels"] = record["channels"] if isinstance(record["channels"], str) else ",".join(
                str(channel) for channel in record["channels"]
            )
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def _write_condition_latex(
    path: Path,
    rows: list[dict],
    conditions: list[str],
    comparison_id: str,
    comparison_type: str,
) -> None:
    label = f"tab:{comparison_type}-conditions-{comparison_id.replace('_', '-')}"
    notes = [rows[0]["labels"][name] for name in conditions] if rows else []
    detail = ". ".join(notes)
    if comparison_type == "augmentation":
        caption = (
            "Validation accuracy at the selected checkpoint for each training "
            "augmentation on clean and transformed inputs. "
            + detail
            + "."
        )
        header = "Training augmentation & " + " & ".join(
            _latex_text(_short_condition_header(name)) for name in conditions
        )
        alignment = "l" + "r" * len(conditions)
    else:
        caption = (
            "Validation accuracy at the selected checkpoint for convolutional "
            "networks of different depth. "
            + detail
            + "."
        )
        header = "CNN depth & " + " & ".join(
            _latex_text(_short_condition_header(name)) for name in conditions
        )
        alignment = "l" + "r" * len(conditions)
    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        f"\\caption{{{_latex_text(caption)}}}",
        f"\\label{{{label}}}",
        "\\vspace{0.6em}",
        "\\small",
        "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}" + alignment + "@{}}",
        "\\toprule",
        header + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        cells = [_latex(row["label"])]
        for name in conditions:
            cells.append(_latex_metric(row["accuracies"][name], row["accuracy_stds"][name]))
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular*}", "\\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_condition_comparison(
    rows: list[dict],
    conditions: list[str],
    path: Path,
    comparison_type: str = "depth",
) -> list[Path]:
    """Grouped bars of validation accuracy for each configuration and condition."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [row["label"] for row in rows]
    positions = np.arange(len(rows))
    width = min(0.8 / max(len(conditions), 1), 0.18)
    colors = ("#1f4e79", "#c47b2b", "#2f6f4e", "#7a4e8a", "#8c3a3a", "#3d6d8c")
    figure, axis = plt.subplots(
        figsize=(max(7.2, 1.35 * len(rows) + 0.45 * len(conditions)), 4.8)
    )
    for index, name in enumerate(conditions):
        offset = (index - (len(conditions) - 1) / 2) * width
        values = [row["accuracies"][name] for row in rows]
        errors = [0 if row["accuracy_stds"][name] is None else row["accuracy_stds"][name] for row in rows]
        axis.bar(
            positions + offset,
            values,
            width,
            yerr=errors,
            capsize=3,
            color=colors[index % len(colors)],
            label=rows[0]["labels"][name] if rows else name,
        )
    axis.set_xticks(
        positions,
        labels,
        rotation=15 if len(rows) > 3 else 0,
        ha="right" if len(rows) > 3 else "center",
    )
    axis.set_ylim(0, 1)
    axis.set_ylabel("Validation accuracy")
    if comparison_type == "augmentation":
        axis.set_xlabel("Training augmentation")
        axis.set_title("RQ1 augmentation comparison\nClean and transformed validation accuracy")
    else:
        axis.set_xlabel("CNN depth")
        axis.set_title("RQ2 depth comparison\nClean and transformed validation accuracy")
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    return save_figure(figure, path)


def _condition_digest(selected: list[dict[str, dict]], conditions: list[str]) -> str:
    parts = []
    for name in conditions:
        parts.append(name)
        parts.append(_condition_signature(selected[0][name]))
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:6]
    return digest


def _group_key(summary: dict, comparison_type: str) -> tuple:
    if comparison_type == "augmentation":
        return (summary["rotation_degrees"], summary["brightness"], summary["zoom"])
    return (summary["depth"], tuple(summary["channels"]))


def _mean(values: list[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    if not present:
        return None
    return float(np.mean(present))


def _sample_std(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=1))


def _shared_value(group: list[dict], field: str):
    values = {summary[field] for summary in group}
    if len(values) == 1:
        return next(iter(values))
    return ";".join(str(summary[field]) for summary in group)


def plot_comparison(rows: list[dict], comparison_type: str, path: Path) -> list[Path]:
    """Bar chart of selected-checkpoint validation accuracy and macro-F1."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [row["label"] for row in rows]
    accuracy = [row["validation_accuracy"] for row in rows]
    macro_f1 = [row["macro_f1"] for row in rows]
    accuracy_err = [0 if row["validation_accuracy_std"] is None else row["validation_accuracy_std"] for row in rows]
    f1_err = [0 if row["macro_f1_std"] is None else row["macro_f1_std"] for row in rows]
    positions = np.arange(len(rows))
    width = 0.36
    figure, axis = plt.subplots(figsize=(max(6.0, 1.4 * len(rows) + 2.0), 4.4))
    axis.bar(
        positions - width / 2,
        accuracy,
        width,
        yerr=accuracy_err,
        capsize=3,
        color=TRAIN_COLOR,
        label="Validation accuracy",
    )
    axis.bar(
        positions + width / 2,
        macro_f1,
        width,
        yerr=f1_err,
        capsize=3,
        color=VAL_COLOR,
        label="Macro-F1",
    )
    axis.set_xticks(positions, labels, rotation=20 if len(rows) > 3 else 0, ha="right" if len(rows) > 3 else "center")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score")
    axis.set_xlabel("Augmentation configuration" if comparison_type == "augmentation" else "CNN depth")
    title = "RQ1 augmentation comparison" if comparison_type == "augmentation" else "RQ2 depth comparison"
    axis.set_title(f"{title}\nSelected checkpoint")
    axis.legend(frameon=False)
    figure.tight_layout()
    return save_figure(figure, path)


def _write_csv(path: Path, summaries: list[dict], comparison_type: str) -> None:
    ordered = sorted(summaries, key=lambda summary: (_configuration_label(summary, comparison_type), summary["run_id"]))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for summary in ordered:
            record = {field: summary.get(field) for field in CSV_FIELDS}
            record["comparison_type"] = comparison_type
            record["configuration"] = _configuration_label(summary, comparison_type)
            record["channels"] = ",".join(str(channel) for channel in summary["channels"])
            record["checkpoint_epoch"] = summary["checkpoint_epoch"]
            writer.writerow({field: _csv_value(record.get(field)) for field in CSV_FIELDS})


def _write_latex(path: Path, rows: list[dict], comparison_type: str, comparison_id: str) -> None:
    label = f"tab:{comparison_type}-comparison-{comparison_id.replace('_', '-')}"
    if comparison_type == "augmentation":
        caption = (
            "Clean validation metrics at the checkpoint with the highest "
            "clean validation accuracy, for the baseline and augmentation configurations."
        )
        header = (
            "Augmentation configuration & Validation accuracy & Macro-F1 "
            "& Validation loss"
        )
        alignment = "lrrr"
    else:
        caption = (
            "Clean validation metrics at the checkpoint with the highest "
            "clean validation accuracy, for convolutional networks of different depth."
        )
        header = (
            "CNN depth & Trainable parameters & Validation accuracy "
            "& Macro-F1 & Selected checkpoint epoch"
        )
        alignment = "lrrrr"
    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\vspace{0.6em}",
        "\\small",
        f"\\begin{{tabular}}{{{alignment}}}",
        "\\toprule",
        header + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        if comparison_type == "augmentation":
            leading = [_latex(row["label"])]
            trailing = [_latex_metric(row["best_val_loss"], row["best_val_loss_std"], digits=3)]
        else:
            leading = [_latex(row["label"]), _latex(_format_parameters(row["trainable_parameters"]))]
            trailing = []
        cells = leading + [
            _latex_metric(row["validation_accuracy"], row["validation_accuracy_std"]),
            _latex_metric(row["macro_f1"], row["macro_f1_std"]),
            *trailing,
        ]
        if comparison_type != "augmentation":
            cells.append(_latex(_epoch_text(row)))
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _comparison_id(summaries: list[dict]) -> str:
    joined = ",".join(sorted(summary["run_id"] for summary in summaries))
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()[:6]
    return f"{utc_stamp()}_{digest}"


def _is_baseline(summary: dict) -> bool:
    return summary["rotation_degrees"] == 0 and summary["brightness"] == 0 and summary["zoom"] == 0


def _row_is_baseline(row: dict) -> bool:
    return row["rotation_degrees"] == 0 and row["brightness"] == 0 and row["zoom"] == 0


def _configuration_label(summary: dict, comparison_type: str) -> str:
    if comparison_type == "depth":
        channels = ", ".join(str(channel) for channel in summary["channels"])
        return f"{summary['depth']} layers ({channels})"
    return _augmentation_text(summary)


def _epoch_text(row: dict) -> str:
    epochs = str(row["checkpoint_epochs"]).split(";")
    unique = list(dict.fromkeys(epochs))
    if len(unique) == 1:
        return unique[0]
    return ", ".join(unique)


def _hashable(value):
    if isinstance(value, list):
        return tuple(value)
    return value


def _strategy_label(row: dict) -> str:
    """Publication name for a training-augmentation setting."""
    rotation = float(row["rotation_degrees"]) != 0
    zoom = float(row["zoom"]) != 0
    brightness = float(row["brightness"]) != 0
    if not rotation and not zoom and not brightness:
        return "None (baseline)"
    names = {
        (True, False, False): "Rotation only",
        (False, True, False): "Zoom only",
        (False, False, True): "Brightness only",
        (True, True, True): "Rotation + Zoom + Brightness",
    }
    return names.get((rotation, zoom, brightness), " + ".join(
        part
        for part, enabled in (
            ("Rotation", rotation),
            ("Zoom", zoom),
            ("Brightness", brightness),
        )
        if enabled
    ))


def _strategy_sort(row: dict) -> tuple:
    rotation = float(row["rotation_degrees"]) != 0
    zoom = float(row["zoom"]) != 0
    brightness = float(row["brightness"]) != 0
    order = {
        (False, False, False): 0,
        (True, False, False): 1,
        (False, True, False): 2,
        (False, False, True): 3,
        (True, True, True): 4,
    }
    return (order.get((rotation, zoom, brightness), 5), _strategy_label(row))


def _augmentation_text(row: dict) -> str:
    if row["rotation_degrees"] == 0 and row["brightness"] == 0 and row["zoom"] == 0:
        return "none"
    return f"rot {row['rotation_degrees']:g}, bri {row['brightness']:g}, zoom {row['zoom']:g}"


def _latex_metric(mean: float, std: float | None, digits: int = 3) -> str:
    text = f"{mean:.{digits}f}"
    if std is None:
        return text
    return f"{text} $\\pm$ {std:.{digits}f}"


def _format_parameters(value) -> str:
    if isinstance(value, int):
        return f"{value:,}"
    return "" if value is None else str(value)


def _csv_value(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return value


def _short_condition_header(name: str) -> str:
    labels = {
        "clean": "Clean",
        "rotation": "Rotation",
        "zoom": "Zoom",
        "brightness": "Brightness",
        "lowres": "Low-res",
    }
    return labels.get(name, name.replace("-", " ").capitalize())


def _latex_text(value) -> str:
    return _latex(value).replace("×", r"$\times$").replace("°", r"$^{\circ}$")


def _latex(value) -> str:
    text = str(value)
    replacements = (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
    )
    for character, replacement in replacements:
        text = text.replace(character, replacement)
    return text


def _optional_float(value):
    if value is None:
        return None
    return float(value)


def _preview(values) -> str:
    rendered = ", ".join(str(value) for value in sorted(values, key=lambda item: (item is None, str(item))))
    return rendered or "none"
