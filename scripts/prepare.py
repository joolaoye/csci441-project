#!/usr/bin/env python3
"""Write a track-aware split manifest for a local GTSRB extract."""

from __future__ import annotations

import argparse
import logging
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
            "Read an extracted 43-class GTSRB dataset and write a reproducible "
            "track-aware split manifest. The split seed is independent of the "
            "training seed in the experiment YAML."
        ),
        epilog=(
            "example:\n"
            "  uv run python scripts/prepare.py \\\n"
            "    --path /path/to/gtsrb --seed 42\n\n"
            "--path is required. Each person passes the folder where they extracted GTSRB.\n\n"
            "Existing manifests are left unchanged unless --overwrite is set."
        ),
        formatter_class=HelpFormatter,
    )
    parser.add_argument(
        "--path",
        required=True,
        help="Extracted GTSRB folder (Kaggle Train.csv/Test.csv or the official archive). Required.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Split seed. This is not the training seed.")
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Fraction of tracks per class assigned to validation.",
    )
    parser.add_argument(
        "--manifest",
        default="configs/splits/gtsrb_v1.json",
        help="Where to write the split manifest.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing manifest. Without this flag an existing file is left untouched.",
    )
    return parser.parse_args(argv)


def _resolve_cli_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return ROOT / path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from traffic_signs.utils.runs import configure_logging, file_sha256, utc_stamp

    log_path = configure_logging("prepare", utc_stamp(), ROOT)
    logger = logging.getLogger("traffic_signs.prepare")
    try:
        from traffic_signs.data.splits import prepare_dataset

        result = prepare_dataset(
            _resolve_cli_path(args.path),
            _resolve_cli_path(args.manifest),
            seed=args.seed,
            validation_fraction=args.val_fraction,
            overwrite=args.overwrite,
        )
        digest = file_sha256(result["manifest"])
    except Exception as exc:
        logger.exception("Dataset preparation failed")
        print(f"error: {exc}", file=sys.stderr)
        print(f"log: {log_path}", file=sys.stderr)
        return 1

    counts = result["counts"]
    print(f"manifest: {result['manifest']}")
    print(f"sha256: {digest}")
    print(f"train images: {counts['train_images']} ({counts['train_tracks']} tracks)")
    print(f"validation images: {counts['validation_images']} ({counts['validation_tracks']} tracks)")
    print(f"test images: {counts['test_images']}")
    print(f"example id: {result['example_id']}")
    print(f"log: {log_path}")
    logger.info("manifest_sha256=%s", digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
