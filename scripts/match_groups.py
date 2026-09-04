#!/usr/bin/env python3
"""Retrieve and rerank target fields for every source field."""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from metadata_matcher.predict import match_groups_csv


def build_parser() -> argparse.ArgumentParser:
    """Build the A-group to B-group matching parser."""

    parser = argparse.ArgumentParser(
        description="Match an A field CSV against a B field CSV locally on CPU."
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True, help="CSV with A column")
    parser.add_argument("--target", type=Path, required=True, help="CSV with B column")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Review/no-match threshold (defaults to saved config)",
    )
    parser.add_argument(
        "--auto-accept-threshold",
        type=float,
        default=None,
        help="High-confidence threshold (defaults to saved config)",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--query-chunk-size", type=int, default=256)
    parser.add_argument("--candidate-chunk-size", type=int, default=None)
    parser.add_argument(
        "--save-candidates",
        action="store_true",
        help="Also write all retrieved/reranked candidates",
    )
    parser.add_argument(
        "--candidates-output",
        type=Path,
        help="Candidate CSV path (default: candidates.csv beside --output)",
    )
    return parser


def main() -> None:
    """Run group matching and report output paths."""

    args = build_parser().parse_args()
    if args.candidates_output and not args.save_candidates:
        raise SystemExit("--candidates-output requires --save-candidates")
    candidates_path = None
    if args.save_candidates:
        candidates_path = args.candidates_output or args.output.with_name("candidates.csv")
    match_groups_csv(
        args.model_dir,
        args.source,
        args.target,
        args.output,
        candidates_output_path=candidates_path,
        top_k=args.top_k,
        review_threshold=args.threshold,
        auto_accept_threshold=args.auto_accept_threshold,
        batch_size=args.batch_size,
        query_chunk_size=args.query_chunk_size,
        candidate_chunk_size=args.candidate_chunk_size,
    )
    print(f"Predictions: {args.output}")
    if candidates_path is not None:
        print(f"Candidates: {candidates_path}")


if __name__ == "__main__":
    main()
