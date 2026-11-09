#!/usr/bin/env python3
"""Predict the class of one traffic-sign image."""

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
            "Load a model.keras checkpoint and predict one image. "
            "Preprocessing is taken from the training configuration stored beside the checkpoint."
        ),
        epilog=(
            "example:\n"
            "  uv run python scripts/predict.py \\\n"
            "    --checkpoint artifacts/<experiment>/model.keras \\\n"
            "    --image path/to/image.png"
        ),
        formatter_class=HelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, help="Path to a model.keras file from training.")
    parser.add_argument("--image", required=True, help="Path to one image file.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        from traffic_signs.inference.predict import predict_image
    except Exception as exc:
        print(f"error: could not import TensorFlow prediction code: {exc}", file=sys.stderr)
        return 1
    try:
        checkpoint = Path(args.checkpoint)
        image = Path(args.image)
        if not checkpoint.is_absolute():
            checkpoint = ROOT / checkpoint
        if not image.is_absolute():
            image = ROOT / image
        result = predict_image(checkpoint, image, project_dir=ROOT)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    name = result["class_name"] if result["class_name"] is not None else "unavailable"
    print(f"class_id: {result['class_id']}")
    print(f"class_name: {name}")
    print(f"confidence: {result['confidence']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
