#!/usr/bin/env python3
"""Evaluate a saved Keras checkpoint without training it."""

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
            "Load a model.keras checkpoint and score it on the validation or "
            "test split. This does not modify the training metrics.json."
        ),
        epilog=(
            "examples:\n"
            "  uv run python scripts/evaluate.py \\\n"
            "    --checkpoint artifacts/<experiment>/model.keras \\\n"
            "    --split validation \\\n"
            "    --conditions clean rotation zoom brightness lowres \\\n"
            "    --data-root /path/to/gtsrb\n\n"
            "  uv run python scripts/evaluate.py \\\n"
            "    --checkpoint artifacts/<experiment>/model.keras \\\n"
            "    --split validation \\\n"
            "    --condition rotation \\\n"
            "    --data-root /path/to/gtsrb\n\n"
            "--data-root is required. Do not rely on the path stored by whoever trained the run.\n"
            "--conditions scores every named condition in one run and loads the checkpoint once.\n"
            "--condition scores one condition and defaults to clean. Parameters come from\n"
            "evaluation.yaml when that file exists, otherwise from evaluation.yaml.example.\n"
            "Each condition writes a new file and does not train the model.\n\n"
            "Use the validation split while comparing experiments. "
            "Reserve --split test for the final report."
        ),
        formatter_class=HelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, help="Path to a model.keras file from training.")
    parser.add_argument(
        "--split",
        required=True,
        choices=("validation", "test"),
        help="Manifest split to score. 'test' is the official GTSRB test partition.",
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="Extracted GTSRB folder for this evaluation. Required.",
    )
    parser.add_argument(
        "--condition",
        help="One condition name from the evaluation YAML. Defaults to clean.",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        help=(
            "Score each of these conditions in one run. "
            "Example: clean rotation zoom brightness lowres."
        ),
    )
    parser.add_argument(
        "--conditions-config",
        default=argparse.SUPPRESS,
        help=(
            "Evaluation conditions file. Omit this to use evaluation.yaml, "
            "or evaluation.yaml.example when that copy does not exist."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        from traffic_signs.evaluation.conditions import default_conditions_path
        from traffic_signs.evaluation.evaluate import evaluate_conditions, requested_condition_names
    except Exception as exc:
        print(f"error: could not import TensorFlow evaluation code: {exc}", file=sys.stderr)
        return 1
    try:
        checkpoint = Path(args.checkpoint)
        if not checkpoint.is_absolute():
            checkpoint = ROOT / checkpoint
        requested = getattr(args, "conditions_config", None)
        if requested:
            conditions_path = Path(requested)
            if not conditions_path.is_absolute():
                conditions_path = ROOT / conditions_path
        else:
            conditions_path = default_conditions_path(ROOT)
        names = requested_condition_names(args.condition, args.conditions)
        results = evaluate_conditions(
            checkpoint,
            args.split,
            names,
            project_dir=ROOT,
            data_root=args.data_root,
            conditions_path=conditions_path,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for result in results:
        metrics = result["metrics"]
        print(f"run: {result['run_id']}")
        print(f"split: {args.split}")
        print(f"condition: {metrics['condition']['label']}")
        print(f"accuracy: {metrics['accuracy']:.4f}")
        print(f"macro_f1: {metrics['macro_f1']:.4f}")
        print(f"results: {result['results_path']}")
    print(f"log: {results[-1]['log_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
