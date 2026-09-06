"""Offline evaluation for saved metadata field matching runs."""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from .dataset import validate_normalized_pair_labels
from .evaluation_truth import group_truth_completeness
from .metrics import (
    classification_metrics,
    end_to_end_metrics,
    retrieval_metrics,
)
from .predict import (
    ModelBundle,
    _score_retrieved_candidates,
    encode_field_names,
    load_model_bundle,
    predict_pairs,
)
from .retrieval import cosine_top_k
from .preprocess import normalize_field_name


LOGGER = logging.getLogger("metadata_matcher")
MATCH_LABELS = frozenset({"DIRECT", "DERIVATION"})


def _load_evaluation_csv(path: str | Path) -> pd.DataFrame:
    """Load and validate a labeled ``A,B,label`` evaluation CSV."""

    csv_path = Path(path)
    try:
        frame = pd.read_csv(
            csv_path,
            encoding="utf-8-sig",
            dtype=str,
            keep_default_na=False,
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Evaluation CSV does not exist: {csv_path}") from exc
    return _validated_evaluation_frame(frame, context=str(csv_path))


def _validated_evaluation_frame(
    dataframe: pd.DataFrame, *, context: str = "evaluation dataframe",
    allow_empty: bool = False,
) -> pd.DataFrame:
    """Validate CLI and library inputs using the same annotation rules."""

    frame = dataframe
    required = ["A", "B", "label"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{context} is missing columns: {missing!r}")
    if frame.empty and not allow_empty:
        raise ValueError(
            f"{context} contains no rows. Formal evaluation requires a non-empty "
            "human-annotated split; inspect split_report.md for component limitations."
        )
    if frame[required].isna().any().any():
        raise ValueError(f"{context} contains missing A, B, or label values")
    frame = frame.copy()
    for column in ("A", "B"):
        frame[column] = frame[column].astype(str)
        empty = frame[column].map(normalize_field_name).eq("")
        if empty.any():
            rows = [position + 2 for position, invalid in enumerate(empty) if invalid]
            raise ValueError(
                f"{context} has {column} values that normalize to empty text "
                f"at CSV rows {rows[:10]}"
            )
    frame["label"] = frame["label"].astype(str).str.strip().str.upper()
    unknown = sorted(set(frame["label"]) - {"NO_MATCH", *MATCH_LABELS})
    if unknown:
        raise ValueError(f"Unsupported labels in {context}: {unknown!r}")
    validate_normalized_pair_labels(frame, context=context)
    group_truth_completeness(frame, context=context)
    return frame


def evaluate_pair_classification(
    bundle: ModelBundle,
    dataframe: pd.DataFrame,
    *,
    batch_size: int | None = None,
) -> dict[str, Any]:
    """Evaluate all human-provided pair labels in a dataframe."""

    dataframe = _validated_evaluation_frame(dataframe)
    predictions = predict_pairs(
        bundle,
        dataframe["A"].astype(str).tolist(),
        dataframe["B"].astype(str).tolist(),
        batch_size=batch_size,
    )
    true_ids = [bundle.label_map[label] for label in dataframe["label"]]
    predicted_ids = [
        bundle.label_map[label] for label in predictions["predicted_label"]
    ]
    report = classification_metrics(
        true_ids, predicted_ids, label_map=bundle.label_map
    )
    no_match_count = int((dataframe["label"] == "NO_MATCH").sum())
    report["no_match_examples"] = no_match_count
    report["no_match_metrics_available"] = no_match_count > 0
    if no_match_count == 0:
        report["warning"] = (
            "NO_MATCH metrics are unavailable or incomplete because this "
            "evaluation split contains no human-verified NO_MATCH rows."
        )
    return report


def _unique_in_order(values: Sequence[str]) -> list[str]:
    """Deduplicate strings while preserving their first observed order."""

    return list(dict.fromkeys(values))


def _positive_ground_truth(
    dataframe: pd.DataFrame,
) -> tuple[list[str], dict[str, dict[str, str]]]:
    """Build target-to-relationship truth maps and preserve all source order."""

    relationships: dict[str, dict[str, str]] = defaultdict(dict)
    source_order = _unique_in_order(
        dataframe["A"].astype(str).map(normalize_field_name).tolist()
    )
    positive_rows = dataframe[dataframe["label"].isin(MATCH_LABELS)]
    for row in positive_rows.itertuples(index=False):
        source = normalize_field_name(str(row.A))
        target = normalize_field_name(str(row.B))
        label = str(row.label)
        existing = relationships[source].get(target)
        if existing is not None and existing != label:
            raise ValueError(
                "Conflicting positive labels for evaluation pair "
                f"({source!r}, {target!r}): {existing!r} and {label!r}"
            )
        relationships[source][target] = label
    return source_order, dict(relationships)


def evaluate_retrieval_and_end_to_end(
    bundle: ModelBundle,
    dataframe: pd.DataFrame,
    *,
    top_k: int | None = None,
    retrieval_ks: Sequence[int] = (1, 5, 10, 20),
    retrieval_depth: int | None = None,
    thresholds: Sequence[float] = (0.80, 0.90, 0.95, 0.99),
    batch_size: int | None = None,
    query_chunk_size: int = 256,
    candidate_chunk_size: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate positive-source retrieval and the complete reranking pipeline.

    The B candidate group contains every unique target observed in the evaluation
    CSV, including targets from NO_MATCH rows. Names are deduplicated after the
    same normalization used by the encoder. Group decision metrics require an
    explicit ``group_truth_complete=true`` annotation; incomplete sources are
    reported separately and still contribute known-positive retrieval metrics.
    """

    # Retain the low-level diagnostic for empty groups; CSV and full-evaluation
    # entrypoints reject empty splits before loading or running a model.
    dataframe = _validated_evaluation_frame(dataframe, allow_empty=True)
    completeness = group_truth_completeness(dataframe)
    normalized_ks = sorted(set(int(k) for k in retrieval_ks))
    if not normalized_ks or any(k <= 0 for k in normalized_ks):
        raise ValueError("retrieval_ks must contain positive integers")
    resolved_top_k = bundle.config.top_k if top_k is None else int(top_k)
    if resolved_top_k <= 0:
        raise ValueError("top_k must be positive")
    minimum_depth = max(max(normalized_ks), resolved_top_k)
    if retrieval_depth is None:
        retrieval_depth = minimum_depth
    elif (
        isinstance(retrieval_depth, bool)
        or not isinstance(retrieval_depth, int)
        or retrieval_depth < minimum_depth
    ):
        raise ValueError(
            f"retrieval_depth must be an integer >= {minimum_depth} "
            "(the largest Recall@K cutoff and reranking top_k)"
        )

    sources, relationship_truth = _positive_ground_truth(dataframe)
    positive_sources = [source for source in sources if relationship_truth.get(source)]
    targets = _unique_in_order(
        dataframe["B"].astype(str).map(normalize_field_name).tolist()
    )
    if not sources or not targets:
        retrieval_report = retrieval_metrics(
            [[] for _ in positive_sources],
            [set(relationship_truth[source]) for source in positive_sources],
            ks=normalized_ks,
            mrr_cutoff=retrieval_depth,
        )
        retrieval_report.update(
            candidate_targets=len(targets),
            retrieval_depth_requested=retrieval_depth,
            retrieval_depth_effective=0,
            ranking_is_complete=True,
        )
        end_to_end_report = end_to_end_metrics(
            [],
            [],
            [],
            [],
            [],
            thresholds=thresholds,
            review_threshold=bundle.config.review_threshold,
            auto_accept_threshold=bundle.config.auto_accept_threshold,
        )
        end_to_end_report["warning"] = (
            "End-to-end matching metrics are unavailable because the evaluation "
            "split is empty."
        )
        return retrieval_report, end_to_end_report

    source_embeddings = encode_field_names(sources, bundle, batch_size=batch_size)
    target_embeddings = encode_field_names(targets, bundle, batch_size=batch_size)
    scores, indices = cosine_top_k(
        source_embeddings,
        target_embeddings,
        top_k=retrieval_depth,
        query_chunk_size=query_chunk_size,
        candidate_chunk_size=(
            bundle.config.retrieval_chunk_size
            if candidate_chunk_size is None
            else candidate_chunk_size
        ),
    )
    ranked_targets = [
        [targets[int(target_index)] for target_index in source_indices]
        for source_indices in indices.tolist()
    ]
    source_to_position = {source: index for index, source in enumerate(sources)}
    retrieval_report = retrieval_metrics(
        [ranked_targets[source_to_position[source]] for source in positive_sources],
        [set(relationship_truth[source]) for source in positive_sources],
        ks=normalized_ks,
        mrr_cutoff=retrieval_depth,
    )
    retrieval_report.update(
        candidate_targets=len(targets),
        retrieval_depth_requested=retrieval_depth,
        retrieval_depth_effective=int(indices.shape[1]),
        ranking_is_complete=int(indices.shape[1]) == len(targets),
        field_identity="normalized_name",
    )

    e2e_depth = min(resolved_top_k, indices.shape[1])
    e2e_indices = indices[:, :e2e_depth]
    candidate_probabilities = _score_retrieved_candidates(
        bundle,
        source_embeddings,
        target_embeddings,
        e2e_indices,
        batch_size=batch_size,
        query_chunk_size=query_chunk_size,
    )
    direct_id = bundle.label_map["DIRECT"]
    derivation_id = bundle.label_map["DERIVATION"]
    match_probabilities = (
        candidate_probabilities[:, :, direct_id]
        + candidate_probabilities[:, :, derivation_id]
    )
    winner_positions = torch.argmax(match_probabilities, dim=1)

    predicted_targets: list[str] = []
    predicted_labels: list[str] = []
    confidences: list[float] = []
    accepted_labels: list[set[str]] = []
    for source_index, source in enumerate(sources):
        winner_position = int(winner_positions[source_index].item())
        target_index = int(e2e_indices[source_index, winner_position].item())
        predicted_target = targets[target_index]
        probabilities = candidate_probabilities[source_index, winner_position]
        predicted_label = (
            "DIRECT"
            if probabilities[direct_id] > probabilities[derivation_id]
            else "DERIVATION"
        )
        predicted_targets.append(predicted_target)
        predicted_labels.append(predicted_label)
        confidences.append(float(match_probabilities[source_index, winner_position]))
        # When the target is right, require its target-specific relationship.
        # For a wrong target, the independent relationship metric compares with
        # all valid relationships, while overall correctness remains false.
        source_truth = relationship_truth.get(source, {})
        target_specific = source_truth.get(predicted_target)
        accepted_labels.append(
            {target_specific}
            if target_specific is not None
            else set(source_truth.values())
        )

    end_to_end_report = end_to_end_metrics(
        predicted_targets,
        predicted_labels,
        confidences,
        [set(relationship_truth.get(source, {})) for source in sources],
        accepted_labels,
        ground_truth_complete=[completeness[source] for source in sources],
        thresholds=thresholds,
        review_threshold=bundle.config.review_threshold,
        auto_accept_threshold=bundle.config.auto_accept_threshold,
    )
    end_to_end_report["retrieval_top_k_for_reranking"] = e2e_depth
    end_to_end_report["candidate_targets"] = len(targets)
    end_to_end_report["field_identity"] = "normalized_name"
    # Useful diagnostic: retrieval cosine of the MLP-selected candidate.  It is
    # intentionally not treated as a confidence score.
    end_to_end_report["mean_selected_retrieval_similarity"] = float(
        torch.stack(
            [scores[row, int(winner_positions[row])] for row in range(len(sources))]
        ).mean()
    )
    return retrieval_report, end_to_end_report


def evaluate_dataframe(
    model: ModelBundle | str | Path,
    dataframe: pd.DataFrame,
    *,
    top_k: int | None = None,
    retrieval_depth: int | None = None,
    thresholds: Sequence[float] = (0.80, 0.90, 0.95, 0.99),
    batch_size: int | None = None,
    query_chunk_size: int = 256,
    candidate_chunk_size: int | None = None,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Evaluate a validated dataframe and return a JSON-ready report."""

    frame = _validated_evaluation_frame(dataframe)
    bundle = model if isinstance(model, ModelBundle) else load_model_bundle(model, device=device)

    pair_report = evaluate_pair_classification(
        bundle, frame, batch_size=batch_size
    )
    retrieval_report, end_to_end_report = evaluate_retrieval_and_end_to_end(
        bundle,
        frame,
        top_k=top_k,
        retrieval_depth=retrieval_depth,
        thresholds=thresholds,
        batch_size=batch_size,
        query_chunk_size=query_chunk_size,
        candidate_chunk_size=candidate_chunk_size,
    )
    return {
        "data_summary": {
            "pair_rows": len(frame),
            "positive_pair_rows": int(frame["label"].isin(MATCH_LABELS).sum()),
            "no_match_pair_rows": int((frame["label"] == "NO_MATCH").sum()),
            "unique_source_fields": int(frame["A"].nunique()),
            "unique_target_fields": int(frame["B"].nunique()),
        },
        "pair_classification": pair_report,
        "retrieval": retrieval_report,
        "end_to_end": end_to_end_report,
    }


def evaluate_model(
    model_dir: str | Path,
    test_csv: str | Path | None = None,
    *,
    data_dir: str | Path = "data_split",
    split: str = "test",
    output_path: str | Path | None = None,
    top_k: int | None = None,
    retrieval_depth: int | None = None,
    thresholds: Sequence[float] = (0.80, 0.90, 0.95, 0.99),
    batch_size: int | None = None,
    query_chunk_size: int = 256,
    candidate_chunk_size: int | None = None,
) -> dict[str, Any]:
    """Evaluate a prepared test/validation split or an explicitly supplied CSV."""

    if split not in {"validation", "test"}:
        raise ValueError("Evaluation split must be 'validation' or 'test'")
    csv_path = Path(test_csv) if test_csv is not None else Path(data_dir) / f"{split}.csv"
    frame = _load_evaluation_csv(csv_path)
    report = evaluate_dataframe(
        model_dir,
        frame,
        top_k=top_k,
        retrieval_depth=retrieval_depth,
        thresholds=thresholds,
        batch_size=batch_size,
        query_chunk_size=query_chunk_size,
        candidate_chunk_size=candidate_chunk_size,
    )
    report["data_summary"]["input_file"] = str(csv_path.resolve())
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the saved-run evaluation CLI parser."""

    parser = argparse.ArgumentParser(
        description="Evaluate pair classification, retrieval, and group matching"
    )
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data_split"),
        help="Directory generated by prepare_dataset.py (default: data_split)",
    )
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument(
        "--test", type=Path, help="Explicit evaluation CSV; overrides --data-dir/--split"
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument(
        "--retrieval-depth", type=int, default=None,
        help="MRR cutoff; defaults to max(20, top_k), must be >= max(20, top_k)",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--query-chunk-size", type=int, default=256)
    parser.add_argument("--candidate-chunk-size", type=int, default=None)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.80, 0.90, 0.95, 0.99],
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Evaluate from command-line arguments and print the JSON report."""

    args = build_arg_parser().parse_args(argv)
    report = evaluate_model(
        args.model_dir,
        args.test,
        data_dir=args.data_dir,
        split=args.split,
        output_path=args.output,
        top_k=args.top_k,
        retrieval_depth=args.retrieval_depth,
        thresholds=args.thresholds,
        batch_size=args.batch_size,
        query_chunk_size=args.query_chunk_size,
        candidate_chunk_size=args.candidate_chunk_size,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through CLI wrapper.
    raise SystemExit(main())


__all__ = [
    "evaluate_dataframe",
    "evaluate_model",
    "evaluate_pair_classification",
    "evaluate_retrieval_and_end_to_end",
    "main",
]
