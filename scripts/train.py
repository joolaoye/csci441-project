#!/usr/bin/env python3
"""Train a traffic-sign CNN from a YAML configuration."""

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
            "Train one CNN run from a YAML configuration. The artifact directory "
            "is artifacts/<experiment>. Training the same name again replaces it."
        ),
        epilog=(
            "example:\n"
            "  uv run python scripts/train.py \\\n"
            "    --config training.yaml \\\n"
            "    --data-root /path/to/gtsrb\n\n"
            "Copy training.yaml.example to training.yaml and set experiment to the folder name, "
            "such as depth2 or rotation10. --data-root is required and is "
            "recorded in the run configuration."
        ),
        formatter_class=HelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Path to an experiment YAML file.")
    parser.add_argument(
        "--data-root",
        required=True,
        help="Extracted GTSRB folder for this run. Required. Recorded in the run configuration.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        from traffic_signs.training.trainer import train
        from traffic_signs.utils.config import load_config, override_data_root
    except Exception as exc:
        print(f"error: could not import TensorFlow training code: {exc}", file=sys.stderr)
        return 1
    try:
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = ROOT / config_path
        config = load_config(config_path)
        override_data_root(config, args.data_root)
        run_dir = train(config, project_dir=ROOT)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"run: {run_dir}")
    print(f"checkpoint: {run_dir / 'model.keras'}")
    print(f"metadata: {run_dir / 'metadata.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
