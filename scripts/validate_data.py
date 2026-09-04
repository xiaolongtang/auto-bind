#!/usr/bin/env python3
"""Validate input schemas, labels, duplicates, and split leakage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from metadata_matcher.train import validate_data_files
from metadata_matcher.utils import configure_logging, write_json


def build_parser() -> argparse.ArgumentParser:
    """Build the data validation CLI parser."""

    parser = argparse.ArgumentParser(description="Validate fixed A/B/label CSV splits.")
    parser.add_argument("--train", type=Path, default=Path("data/train.csv"))
    parser.add_argument(
        "--validation", type=Path, default=Path("data/validation.csv")
    )
    parser.add_argument("--test", type=Path, default=Path("data/test.csv"))
    parser.add_argument(
        "--strict-leakage-check",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Exit non-zero if any normalized field overlaps between splits",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    return parser


def main() -> None:
    """Validate data and emit a readable JSON report."""

    args = build_parser().parse_args()
    logger = configure_logging()
    _, _, _, report = validate_data_files(
        args.train,
        args.validation,
        args.test,
        strict_leakage_check=args.strict_leakage_check,
        logger=logger,
    )
    if args.output:
        write_json(args.output, report)
        print(f"Validation report: {args.output}")
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
