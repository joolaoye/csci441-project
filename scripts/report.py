#!/usr/bin/env python3
"""Build figures and comparison tables from completed experiment runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create figures and tables from saved training artifacts. "
            "This does not train a model or download data."
        ),
        formatter_class=HelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser(
        "run",
        help="Write figures for one completed run.",
        description=(
            "Read history.csv, the validation or test evaluation, and the split "
            "manifest for one completed run. Figures are written under that run "
            "unless --output is set."
        ),
        epilog=(
            "example:\n"
            "  uv run python scripts/report.py run \\\n"
            "    --run artifacts/<experiment> \\\n"
            "    --data-root /path/to/gtsrb\n"
        ),
        formatter_class=HelpFormatter,
    )
    run.add_argument("--run", required=True, help="Directory of one completed training run.")
    run.add_argument(
        "--split",
        choices=("validation", "test"),
        default="validation",
        help="Evaluation split to plot. Test requires an existing evaluation file.",
    )
    run.add_argument(
        "--data-root",
        help="Extracted GTSRB folder. Needed when the run uses augmentation.",
    )
    run.add_argument("--output", help="Figure directory. Defaults to <run>/figures.")

    dataset = commands.add_parser(
        "dataset",
        help="Write the class distribution for the split manifest.",
        description=(
            "Read the committed split manifest and write a class-distribution "
            "figure plus a summary table. This does not use a training run."
        ),
        epilog=(
            "example:\n"
            "  uv run python scripts/report.py dataset \\\n"
            "    --manifest configs/splits/gtsrb_v1.json\n"
        ),
        formatter_class=HelpFormatter,
    )
    dataset.add_argument(
        "--manifest",
        default="configs/splits/gtsrb_v1.json",
        help="Split manifest. Defaults to the committed GTSRB manifest.",
    )
    dataset.add_argument(
        "--output",
        help="Output directory. Defaults to artifacts/dataset.",
    )

    conditions = commands.add_parser(
        "conditions",
        help="Preview evaluation conditions on real images.",
        description=(
            "Write PNG previews of the evaluation conditions. The previews use "
            "the same transforms as evaluation and do not run a model. An existing "
            "preview is reused when the split, image size, samples, and parameters match."
        ),
        epilog=(
            "example:\n"
            "  uv run python scripts/report.py conditions \\\n"
            "    --split validation \\\n"
            "    --data-root /path/to/gtsrb\n"
        ),
        formatter_class=HelpFormatter,
    )
    conditions.add_argument(
        "--split",
        choices=("validation", "test"),
        default="validation",
        help="Split to sample. Defaults to validation.",
    )
    conditions.add_argument(
        "--data-root",
        required=True,
        help="Extracted GTSRB folder. The manifest stores relative image paths.",
    )
    conditions.add_argument(
        "--manifest",
        default="configs/splits/gtsrb_v1.json",
        help="Split manifest. Defaults to the committed GTSRB manifest.",
    )
    conditions.add_argument(
        "--conditions-config",
        default=argparse.SUPPRESS,
        help=(
            "Evaluation conditions file. Omit this to use evaluation.yaml, "
            "or evaluation.yaml.example when that copy does not exist."
        ),
    )
    conditions.add_argument(
        "--condition",
        help="Preview one condition. Defaults to every condition in the YAML file.",
    )
    conditions.add_argument(
        "--image-size",
        type=int,
        default=48,
        help="Model input size used for preprocessing and low-resolution restoration.",
    )
    conditions.add_argument(
        "--output",
        help="Output directory. Defaults to artifacts/evaluation_previews.",
    )

    compare = commands.add_parser(
        "compare",
        help="Compare completed runs for augmentation or depth.",
        description=(
            "RQ1 compares a baseline with no augmentation against one or more "
            "augmentation settings. RQ2 compares convolutional depths. Both use "
            "validation metrics from the selected checkpoint."
        ),
        epilog=(
            "examples:\n"
            "  uv run python scripts/report.py compare \\\n"
            "    --type augmentation \\\n"
            "    --runs artifacts/<baseline-run> artifacts/<augmented-run>\n\n"
            "  uv run python scripts/report.py compare \\\n"
            "    --type depth \\\n"
            "    --runs artifacts/<depth2-run> artifacts/<depth3-run> artifacts/<depth4-run>\n\n"
            "  uv run python scripts/report.py compare \\\n"
            "    --type depth \\\n"
            "    --runs artifacts/<depth2-run> artifacts/<depth3-run> artifacts/<depth4-run> \\\n"
            "    --conditions clean rotation zoom brightness lowres\n\n"
            "  uv run python scripts/report.py compare \\\n"
            "    --type augmentation \\\n"
            "    --runs artifacts/<baseline-run> artifacts/<rotation-run> \\\n"
            "    --conditions clean rotation zoom brightness lowres\n"
        ),
        formatter_class=HelpFormatter,
    )
    compare.add_argument(
        "--type",
        required=True,
        choices=("augmentation", "depth"),
        help="augmentation compares RQ1. depth compares RQ2.",
    )
    compare.add_argument("--runs", nargs="+", required=True, help="Completed run directories.")
    compare.add_argument(
        "--conditions",
        nargs="+",
        help=(
            "Saved evaluation conditions to compare, such as "
            "clean rotation zoom brightness lowres. Omit this for the clean summary table."
        ),
    )
    compare.add_argument(
        "--output",
        help="Parent directory. Defaults to artifacts/comparisons.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "run":
            return _report_run(args)
        if args.command == "dataset":
            return _report_dataset(args)
        if args.command == "conditions":
            return _report_conditions(args)
        return _compare_runs(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _report_run(args: argparse.Namespace) -> int:
    from traffic_signs.reporting.plots import generate_run_report

    run_dir = _resolve(args.run)
    output_dir = _resolve(args.output) if args.output else None
    result = generate_run_report(
        run_dir,
        output_dir=output_dir,
        split=args.split,
        data_root=args.data_root,
        project_dir=ROOT,
    )
    print(f"figures: {result['output_dir']}")
    for path in result["written"]:
        print(path)
    for warning in result["warnings"]:
        print(f"warning: {warning}", file=sys.stderr)
    return 0


def _report_dataset(args: argparse.Namespace) -> int:
    from traffic_signs.reporting.plots import generate_dataset_report

    output_dir = _resolve(args.output) if args.output else ROOT / "artifacts" / "dataset"
    result = generate_dataset_report(_resolve(args.manifest), output_dir)
    print(f"dataset: {result['output_dir']}")
    for path in result["written"]:
        print(path)
    return 0


def _report_conditions(args: argparse.Namespace) -> int:
    from traffic_signs.evaluation.conditions import default_conditions_path
    from traffic_signs.reporting.previews import generate_condition_previews

    output_root = _resolve(args.output) if args.output else ROOT / "artifacts" / "evaluation_previews"
    requested = getattr(args, "conditions_config", None)
    conditions_path = _resolve(requested) if requested else default_conditions_path(ROOT)
    result = generate_condition_previews(
        manifest_path=_resolve(args.manifest),
        data_root=_resolve(args.data_root),
        output_root=output_root,
        split=args.split,
        image_size=args.image_size,
        conditions_path=conditions_path,
        condition_names=[args.condition] if args.condition else None,
        project_dir=ROOT,
    )
    print(f"previews: {result['output_root']}")
    for path in result["written"]:
        print(f"wrote {path}")
    for path in result["reused"]:
        print(f"reused {path}")
    return 0


def _compare_runs(args: argparse.Namespace) -> int:
    from traffic_signs.reporting.comparisons import compare_runs

    output_root = _resolve(args.output) if args.output else ROOT / "artifacts" / "comparisons"
    result = compare_runs(
        [_resolve(path) for path in args.runs],
        args.type,
        output_root,
        conditions=args.conditions,
    )
    print(f"comparison: {result['output_dir']}")
    print(f"csv: {result['csv']}")
    print(f"latex: {result['latex']}")
    print(f"png: {result['png']}")
    for warning in result["warnings"]:
        print(f"warning: {warning}", file=sys.stderr)
    return 0


def _resolve(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
