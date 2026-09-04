"""Metrics for pair classification, retrieval, and end-to-end matching."""

from __future__ import annotations

from collections.abc import Collection, Hashable, Mapping, Sequence
from typing import Any

import numpy as np
import torch


DEFAULT_LABEL_MAP: dict[str, int] = {
    "NO_MATCH": 0,
    "DIRECT": 1,
    "DERIVATION": 2,
}


def _to_numpy(values: Sequence[Any] | np.ndarray | torch.Tensor) -> np.ndarray:
    """Convert a one- or two-dimensional prediction input to NumPy."""

    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def classification_metrics(
    y_true: Sequence[int] | np.ndarray | torch.Tensor,
    y_pred: Sequence[int] | np.ndarray | torch.Tensor,
    *,
    label_map: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Compute accuracy, per-class metrics, macro F1, and a confusion matrix.

    ``y_pred`` may contain integer class IDs or a ``[rows, classes]`` score/logit
    matrix.  All configured classes are reported even when their support is zero,
    which keeps JSON artifacts stable and makes missing ``NO_MATCH`` data visible.
    Zero-denominator precision/recall/F1 values are defined as ``0.0``.
    """

    resolved_label_map = dict(label_map or DEFAULT_LABEL_MAP)
    if not resolved_label_map:
        raise ValueError("label_map must not be empty")
    if len(set(resolved_label_map.values())) != len(resolved_label_map):
        raise ValueError("label_map IDs must be unique")

    true_array = _to_numpy(y_true)
    pred_array = _to_numpy(y_pred)
    if true_array.ndim != 1:
        true_array = true_array.reshape(-1)
    if pred_array.ndim == 2:
        pred_array = pred_array.argmax(axis=1)
    elif pred_array.ndim != 1:
        pred_array = pred_array.reshape(-1)
    if true_array.shape[0] != pred_array.shape[0]:
        raise ValueError(
            "y_true and y_pred must contain the same number of examples: "
            f"{true_array.shape[0]} != {pred_array.shape[0]}"
        )

    ordered_labels = sorted(resolved_label_map.items(), key=lambda item: item[1])
    valid_ids = {class_id for _, class_id in ordered_labels}
    observed = set(true_array.tolist()) | set(pred_array.tolist())
    unknown = observed - valid_ids
    if unknown:
        raise ValueError(f"Unknown class IDs in predictions/targets: {sorted(unknown)!r}")

    matrix = np.zeros((len(ordered_labels), len(ordered_labels)), dtype=np.int64)
    id_to_position = {
        class_id: position for position, (_, class_id) in enumerate(ordered_labels)
    }
    for truth, prediction in zip(true_array.tolist(), pred_array.tolist()):
        matrix[id_to_position[int(truth)], id_to_position[int(prediction)]] += 1

    sample_count = int(true_array.shape[0])
    accuracy = (
        float(np.trace(matrix) / sample_count) if sample_count > 0 else 0.0
    )
    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    for position, (name, _) in enumerate(ordered_labels):
        true_positive = int(matrix[position, position])
        false_positive = int(matrix[:, position].sum() - true_positive)
        false_negative = int(matrix[position, :].sum() - true_positive)
        support = int(matrix[position, :].sum())
        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative
        precision = (
            true_positive / precision_denominator
            if precision_denominator > 0
            else 0.0
        )
        recall = (
            true_positive / recall_denominator if recall_denominator > 0 else 0.0
        )
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall > 0.0
            else 0.0
        )
        values: dict[str, float | int] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": support,
        }
        per_class[name] = values
        f1_values.append(float(f1))

    result: dict[str, Any] = {
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1_values)) if f1_values else 0.0,
        "labels": [name for name, _ in ordered_labels],
        "confusion_matrix": matrix.tolist(),
        "per_class": per_class,
        "sample_count": sample_count,
    }
    # Top-level class keys are convenient in reports and preserve the exact
    # terminology requested by the CLI contract.
    result.update(per_class)
    return result


def _as_relevant_set(value: Any) -> set[Hashable]:
    """Interpret a scalar target as one truth and a collection as many truths."""

    if isinstance(value, (str, bytes)) or not isinstance(value, Collection):
        return {value}
    return set(value)


def retrieval_metrics(
    rankings: Mapping[Hashable, Sequence[Hashable]]
    | Sequence[Sequence[Hashable]],
    ground_truth: Mapping[Hashable, Collection[Hashable] | Hashable]
    | Sequence[Collection[Hashable] | Hashable],
    *,
    ks: Sequence[int] = (1, 5, 10, 20),
) -> dict[str, Any]:
    """Compute per-source Recall@K and mean reciprocal rank (MRR).

    Each source contributes once.  When a source has multiple valid targets, a
    hit occurs when *any* valid target is retrieved, matching the business rule in
    the project specification.  Sources without a valid target are excluded from
    retrieval evaluation and reported via ``skipped_without_truth``.

    Args:
        rankings: Ranked target IDs, either keyed by source or position aligned.
        ground_truth: One target or a collection of targets for each source.
        ks: Positive recall cutoffs.
    """

    normalized_ks = sorted(set(int(k) for k in ks))
    if not normalized_ks or any(k <= 0 for k in normalized_ks):
        raise ValueError("ks must contain at least one positive integer")

    if isinstance(ground_truth, Mapping):
        if not isinstance(rankings, Mapping):
            raise TypeError("rankings must be a mapping when ground_truth is a mapping")
        examples = [
            (list(rankings.get(source, ())), _as_relevant_set(targets))
            for source, targets in ground_truth.items()
        ]
    else:
        if isinstance(rankings, Mapping):
            raise TypeError("rankings must be a sequence when ground_truth is a sequence")
        if len(rankings) != len(ground_truth):
            raise ValueError(
                "rankings and ground_truth must have the same number of sources"
            )
        examples = [
            (list(ranked), _as_relevant_set(targets))
            for ranked, targets in zip(rankings, ground_truth)
        ]

    evaluated = 0
    skipped = 0
    hits = {k: 0 for k in normalized_ks}
    reciprocal_rank_sum = 0.0
    for ranked_targets, relevant_targets in examples:
        if not relevant_targets:
            skipped += 1
            continue
        evaluated += 1
        first_relevant_rank: int | None = None
        for rank, target in enumerate(ranked_targets, start=1):
            if target in relevant_targets:
                first_relevant_rank = rank
                break
        if first_relevant_rank is not None:
            reciprocal_rank_sum += 1.0 / first_relevant_rank
            for k in normalized_ks:
                if first_relevant_rank <= k:
                    hits[k] += 1

    recalls = {
        k: (hits[k] / evaluated if evaluated > 0 else 0.0) for k in normalized_ks
    }
    result: dict[str, Any] = {
        "mrr": reciprocal_rank_sum / evaluated if evaluated > 0 else 0.0,
        "evaluated_sources": evaluated,
        "skipped_without_truth": skipped,
        "recall_at_k": {str(k): recalls[k] for k in normalized_ks},
    }
    # Machine-friendly and report-friendly spellings are both included.
    for k in normalized_ks:
        result[f"recall_at_{k}"] = recalls[k]
        result[f"Recall@{k}"] = recalls[k]
    result["MRR"] = result["mrr"]
    return result


def threshold_precision_coverage(
    correctness: Sequence[bool] | np.ndarray | torch.Tensor,
    confidences: Sequence[float] | np.ndarray | torch.Tensor,
    *,
    thresholds: Sequence[float] = (0.80, 0.90, 0.95, 0.99),
) -> list[dict[str, float | int]]:
    """Measure accepted-prediction precision and coverage by confidence cutoff.

    Precision is defined as correct accepted predictions divided by all accepted
    predictions.  Coverage is accepted predictions divided by all evaluated
    predictions.  Precision is reported as ``0.0`` when no item is accepted, and
    ``accepted_count`` makes this edge case explicit.
    """

    correct_array = _to_numpy(correctness).astype(bool).reshape(-1)
    confidence_array = _to_numpy(confidences).astype(float).reshape(-1)
    if correct_array.shape[0] != confidence_array.shape[0]:
        raise ValueError("correctness and confidences must have equal length")
    if not np.all(np.isfinite(confidence_array)):
        raise ValueError("confidences must contain only finite values")

    normalized_thresholds = [float(threshold) for threshold in thresholds]
    if any(not 0.0 <= threshold <= 1.0 for threshold in normalized_thresholds):
        raise ValueError("thresholds must be in the inclusive range [0, 1]")

    total = int(correct_array.shape[0])
    table: list[dict[str, float | int]] = []
    for threshold in normalized_thresholds:
        accepted = confidence_array >= threshold
        accepted_count = int(accepted.sum())
        correct_count = int(np.logical_and(accepted, correct_array).sum())
        precision = correct_count / accepted_count if accepted_count > 0 else 0.0
        coverage = accepted_count / total if total > 0 else 0.0
        table.append(
            {
                "threshold": threshold,
                "precision": float(precision),
                "coverage": float(coverage),
                "accepted_count": accepted_count,
                "correct_count": correct_count,
            }
        )
    return table


def end_to_end_metrics(
    predicted_targets: Sequence[Hashable],
    predicted_labels: Sequence[str],
    match_probabilities: Sequence[float] | np.ndarray | torch.Tensor,
    true_targets: Sequence[Collection[Hashable] | Hashable],
    true_labels: Sequence[Collection[str] | str],
    *,
    thresholds: Sequence[float] = (0.80, 0.90, 0.95, 0.99),
    review_threshold: float | None = None,
    auto_accept_threshold: float = 0.95,
) -> dict[str, Any]:
    """Compute group-matching and confidence-gating metrics.

    ``true_targets`` and ``true_labels`` support multiple truths per source.  A
    target and relationship are counted independently, while ``overall`` requires
    both to be correct.  For target-specific relationship truth, callers should
    precompute each row's accepted relationship set for ``true_labels``.
    """

    total = len(predicted_targets)
    if not (
        len(predicted_labels)
        == len(true_targets)
        == len(true_labels)
        == len(match_probabilities)
        == total
    ):
        raise ValueError("All end-to-end metric inputs must have equal length")

    relevant_target_sets = [_as_relevant_set(truth) for truth in true_targets]
    positive_mask = np.asarray(
        [bool(targets) for targets in relevant_target_sets], dtype=bool
    )
    target_correct = np.asarray(
        [
            predicted in relevant
            for predicted, relevant in zip(predicted_targets, relevant_target_sets)
        ],
        dtype=bool,
    )
    label_correct = np.asarray(
        [
            predicted in _as_relevant_set(truth)
            for predicted, truth in zip(predicted_labels, true_labels)
        ],
        dtype=bool,
    )
    overall_correct = np.logical_and(target_correct, label_correct)
    denominator = total if total > 0 else 1
    positive_count = int(positive_mask.sum())
    positive_denominator = positive_count if positive_count > 0 else 1
    threshold_table = threshold_precision_coverage(
        overall_correct, match_probabilities, thresholds=thresholds
    )

    resolved_review_threshold = (
        float(review_threshold)
        if review_threshold is not None
        else 0.80
    )
    if not 0.0 <= resolved_review_threshold <= 1.0:
        raise ValueError("review_threshold must be in [0, 1]")
    resolved_auto_accept_threshold = float(auto_accept_threshold)
    if not 0.0 <= resolved_auto_accept_threshold <= 1.0:
        raise ValueError("auto_accept_threshold must be in [0, 1]")
    if resolved_auto_accept_threshold < resolved_review_threshold:
        raise ValueError("auto_accept_threshold must be >= review_threshold")
    auto_accept = threshold_precision_coverage(
        overall_correct,
        match_probabilities,
        thresholds=(resolved_auto_accept_threshold,),
    )[0]
    confidence_array = _to_numpy(match_probabilities).astype(float).reshape(-1)
    accepted_at_review = confidence_array >= resolved_review_threshold
    no_match_mask = np.logical_not(positive_mask)
    # At the operational review gate, a rejected positive is a miss while a
    # rejected source with no positive truth is a correct NO_MATCH decision.
    gated_overall_correct = np.where(
        positive_mask,
        np.logical_and(accepted_at_review, overall_correct),
        np.logical_not(accepted_at_review),
    )
    gated_relationship_correct = np.where(
        positive_mask,
        np.logical_and(accepted_at_review, label_correct),
        np.logical_not(accepted_at_review),
    )
    return {
        "top_1_correct_match_rate": float(
            np.logical_and(target_correct, positive_mask).sum()
            / positive_denominator
        ),
        "relationship_classification_accuracy": float(
            gated_relationship_correct.sum() / denominator
        ),
        "overall_end_to_end_accuracy": float(
            gated_overall_correct.sum() / denominator
        ),
        "raw_positive_relationship_classification_accuracy": float(
            np.logical_and(label_correct, positive_mask).sum()
            / positive_denominator
        ),
        "raw_positive_end_to_end_accuracy": float(
            np.logical_and(overall_correct, positive_mask).sum()
            / positive_denominator
        ),
        "review_threshold": resolved_review_threshold,
        "auto_accept_threshold": resolved_auto_accept_threshold,
        "auto_accept_precision": float(auto_accept["precision"]),
        "coverage": float(auto_accept["coverage"]),
        "threshold_precision_coverage": threshold_table,
        "evaluated_sources": total,
        "positive_sources": positive_count,
        "no_match_only_sources": int(no_match_mask.sum()),
    }


# Backwards-friendly singular name for callers that think of this as a report.
pair_classification_metrics = classification_metrics


__all__ = [
    "DEFAULT_LABEL_MAP",
    "classification_metrics",
    "end_to_end_metrics",
    "pair_classification_metrics",
    "retrieval_metrics",
    "threshold_precision_coverage",
]
