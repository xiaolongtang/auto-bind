#!/usr/bin/env python3
"""Train the complete two-phase metadata field matcher."""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401  # Ensures direct invocation works.

from metadata_matcher.config import MatcherConfig
from metadata_matcher.train import train_model


def build_parser() -> argparse.ArgumentParser:
    """Build the training CLI parser."""

    parser = argparse.ArgumentParser(
        description="Train a local CPU-only metadata field matcher."
    )
    parser.add_argument("--config", type=Path, required=True, help="JSON config file")
    parser.add_argument("--train", type=Path, default=Path("data/train.csv"))
    parser.add_argument(
        "--validation", type=Path, default=Path("data/validation.csv")
    )
    parser.add_argument("--test", type=Path, default=Path("data/test.csv"))
    parser.add_argument(
        "--artifacts-dir", type=Path, default=Path("artifacts")
    )
    parser.add_argument(
        "--strict-leakage-check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Fail when normalized fields occur across splits (default: enabled; "
            "use --no-strict-leakage-check only after review)"
        ),
    )
    return parser


def main() -> None:
    """Run training and print the artifact location once."""

    args = build_parser().parse_args()
    config = MatcherConfig.load(args.config)
    run_dir = train_model(
        train_path=args.train,
        validation_path=args.validation,
        test_path=args.test,
        config=config,
        artifacts_dir=args.artifacts_dir,
        strict_leakage_check=args.strict_leakage_check,
    )
    print(f"Model artifacts: {run_dir}")


if __name__ == "__main__":
    main()

