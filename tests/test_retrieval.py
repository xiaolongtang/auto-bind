"""Tests for exact retrieval, multi-truth metrics, and portable inference."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from metadata_matcher.config import MatcherConfig
from metadata_matcher.metrics import (
    classification_metrics,
    end_to_end_metrics,
    retrieval_metrics,
    threshold_precision_coverage,
)
from metadata_matcher.model import FieldEncoder, PairClassifier
from metadata_matcher.predict import load_model_bundle, predict_pair, predict_pairs
from metadata_matcher.retrieval import cosine_top_k
from metadata_matcher.vocab import CharVocabulary


def test_cosine_top_k_returns_expected_descending_order() -> None:
    queries = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    candidates = torch.tensor(
        [
            [0.0, 2.0],
            [3.0, 0.0],
            [1.0, 1.0],
        ]
    )

    scores, indices = cosine_top_k(
        queries,
        candidates,
        top_k=3,
        query_chunk_size=1,
        candidate_chunk_size=1,
    )

    assert indices.tolist() == [[1, 2, 0], [0, 2, 1]]
    assert torch.all(scores[:, :-1] >= scores[:, 1:])
    assert scores[0].tolist() == pytest.approx([1.0, 2**-0.5, 0.0])


def test_chunked_cosine_top_k_matches_full_matrix() -> None:
    generator = torch.Generator().manual_seed(123)
    queries = torch.randn(11, 7, generator=generator)
    candidates = torch.randn(23, 7, generator=generator)
    expected_scores, expected_indices = torch.topk(
        torch.nn.functional.normalize(queries, dim=1)
        @ torch.nn.functional.normalize(candidates, dim=1).T,
        k=8,
        dim=1,
    )

    actual_scores, actual_indices = cosine_top_k(
        queries,
        candidates,
        top_k=8,
        query_chunk_size=3,
        candidate_chunk_size=5,
    )

    torch.testing.assert_close(actual_scores, expected_scores)
    assert actual_indices.tolist() == expected_indices.tolist()


def test_cosine_top_k_clips_k_and_handles_empty_inputs() -> None:
    scores, indices = cosine_top_k(
        torch.tensor([[1.0, 0.0]]), torch.tensor([[1.0, 0.0]]), top_k=10
    )
    assert scores.shape == indices.shape == (1, 1)

    empty_scores, empty_indices = cosine_top_k(
        torch.empty((2, 4)), torch.empty((0, 4)), top_k=10
    )
    assert empty_scores.shape == empty_indices.shape == (2, 0)


def test_retrieval_metrics_count_each_source_once_with_multiple_truths() -> None:
    rankings = {
        "a": ["wrong", "target_2", "target_1"],
        "b": ["b_target", "wrong"],
    }
    truth = {"a": {"target_1", "target_2"}, "b": {"b_target"}}

    metrics = retrieval_metrics(rankings, truth, ks=(1, 2, 5, 10, 20))

    assert metrics["Recall@1"] == pytest.approx(0.5)
    assert metrics["Recall@2"] == pytest.approx(1.0)
    assert metrics["Recall@20"] == pytest.approx(1.0)
    assert metrics["MRR"] == pytest.approx(0.75)
    assert metrics["evaluated_sources"] == 2


def test_truncated_mrr_is_named_by_depth_and_ignores_later_hits() -> None:
    rankings = [[f"target_{rank}" for rank in range(1, 31)]]
    truth = [{"target_25"}]

    truncated = retrieval_metrics(rankings, truth, mrr_cutoff=20)
    assert truncated["MRR@20"] == truncated["mrr_at_20"] == 0.0
    assert truncated["mrr_cutoff"] == 20
    assert "MRR" not in truncated and "mrr" not in truncated
    assert truncated["Recall@20"] == 0.0

    deeper = retrieval_metrics(rankings, truth, ks=(1, 20, 25), mrr_cutoff=25)
    assert deeper["MRR@25"] == pytest.approx(1 / 25)
    assert deeper["Recall@25"] == 1.0
    assert retrieval_metrics(rankings, truth)["MRR"] == pytest.approx(1 / 25)


@pytest.mark.parametrize("cutoff", [0, -1, 1.5, True, "20"])
def test_retrieval_metrics_reject_invalid_mrr_cutoff(cutoff) -> None:
    with pytest.raises(ValueError, match="mrr_cutoff"):
        retrieval_metrics([["b"]], [{"b"}], mrr_cutoff=cutoff)


def test_recall_cutoffs_cannot_exceed_declared_retrieval_depth() -> None:
    with pytest.raises(ValueError, match="must not exceed"):
        retrieval_metrics([["b"]], [{"b"}], ks=(1, 20), mrr_cutoff=10)


def test_classification_metrics_include_all_three_classes() -> None:
    metrics = classification_metrics([0, 1, 2, 2], [0, 2, 2, 1])

    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["NO_MATCH"]["precision"] == pytest.approx(1.0)
    assert metrics["DIRECT"]["recall"] == pytest.approx(0.0)
    assert metrics["DERIVATION"]["recall"] == pytest.approx(0.5)
    assert metrics["confusion_matrix"] == [[1, 0, 0], [0, 0, 1], [0, 1, 1]]


def test_threshold_precision_coverage_and_no_match_gating() -> None:
    table = threshold_precision_coverage(
        [True, False, True, False],
        [0.99, 0.96, 0.85, 0.10],
        thresholds=(0.80, 0.95),
    )
    assert table[0]["precision"] == pytest.approx(2 / 3)
    assert table[0]["coverage"] == pytest.approx(3 / 4)
    assert table[1]["precision"] == pytest.approx(1 / 2)

    # Only explicitly complete annotation makes the second source's rejection
    # a correct group-level NO_MATCH at the operational review threshold.
    metrics = end_to_end_metrics(
        ["target", "candidate"],
        ["DIRECT", "DIRECT"],
        [0.99, 0.20],
        [{"target"}, set()],
        [{"DIRECT"}, set()],
        thresholds=(0.80, 0.95),
        review_threshold=0.80,
        ground_truth_complete=[True, True],
    )
    assert metrics["overall_end_to_end_accuracy"] == pytest.approx(1.0)
    assert metrics["relationship_classification_accuracy"] == pytest.approx(1.0)
    assert metrics["threshold_precision_coverage"][0]["precision"] == pytest.approx(1.0)
    assert metrics["threshold_precision_coverage"][0]["coverage"] == pytest.approx(0.5)
    assert metrics["auto_accept_threshold"] == pytest.approx(0.95)
    assert metrics["auto_accept_precision"] == pytest.approx(1.0)
    assert metrics["coverage"] == pytest.approx(0.5)
    assert metrics["no_match_only_sources"] == 1


def test_unannotated_positive_is_unknown_not_correct_no_match_rejection() -> None:
    metrics = end_to_end_metrics(
        ["unjudged_candidate"],
        ["DIRECT"],
        [0.1],
        [set()],
        [set()],
    )
    assert metrics["total_sources"] == 1
    assert metrics["evaluated_sources"] == 0
    assert metrics["excluded_incomplete_sources"] == 1
    assert metrics["unknown_no_match_sources"] == 1
    assert metrics["no_match_only_sources"] == 0
    assert metrics["overall_end_to_end_accuracy"] is None
    assert metrics["relationship_classification_accuracy"] is None
    assert metrics["auto_accept_precision"] is None
    assert metrics["coverage"] is None
    assert "incomplete ground truth" in metrics["warning"]
    for row in metrics["threshold_precision_coverage"]:
        assert row["precision"] is None
        assert row["coverage"] is None
        assert row["evaluated_sources"] == row["accepted_count"] == 0
        assert row["all_sources_coverage"] == 0.0


def test_partial_sources_are_excluded_from_all_group_metric_denominators() -> None:
    metrics = end_to_end_metrics(
        ["b", "unknown", "wrong", "unjudged"],
        ["DIRECT", "DIRECT", "DIRECT", "DIRECT"],
        [0.99, 0.99, 0.99, 0.1],
        [{"b"}, set(), {"known_positive"}, set()],
        [{"DIRECT"}, set(), {"DIRECT"}, set()],
        thresholds=(0.95,),
        ground_truth_complete=[True, False, False, True],
    )
    assert metrics["total_sources"] == 4
    assert metrics["evaluated_sources"] == 2
    assert metrics["excluded_incomplete_sources"] == 2
    assert metrics["known_positive_sources"] == 2
    assert metrics["positive_sources"] == 1
    assert metrics["incomplete_positive_sources"] == 1
    assert metrics["unknown_no_match_sources"] == 1
    assert metrics["no_match_only_sources"] == 1
    assert metrics["top_1_correct_match_rate"] == pytest.approx(0.5)
    assert metrics["raw_positive_end_to_end_accuracy"] == pytest.approx(0.5)
    assert metrics["overall_end_to_end_accuracy"] == 1.0
    assert metrics["relationship_classification_accuracy"] == 1.0
    assert metrics["auto_accept_precision"] == 1.0
    assert metrics["coverage"] == 0.5
    assert metrics["auto_accept_count"] == 1
    assert metrics["all_sources_auto_accept_coverage"] == 0.75
    assert metrics["excluded_incomplete_auto_accept_count"] == 2
    row = metrics["threshold_precision_coverage"][0]
    assert row["precision"] == 1.0
    assert row["coverage"] == 0.5
    assert row["accepted_count"] == row["correct_count"] == 1
    assert row["all_sources_accepted_count"] == 3
    assert row["all_sources_coverage"] == 0.75
    assert row["excluded_incomplete_accepted_count"] == 2


def test_complete_no_match_acceptance_is_a_group_error() -> None:
    metrics = end_to_end_metrics(
        ["candidate"], ["DIRECT"], [0.99], [set()], [set()],
        ground_truth_complete=[True],
    )
    assert metrics["overall_end_to_end_accuracy"] == 0.0
    assert metrics["relationship_classification_accuracy"] == 0.0
    assert metrics["auto_accept_precision"] == 0.0
    assert metrics["coverage"] == 1.0
    assert metrics["unknown_no_match_sources"] == 0
    assert metrics["no_match_only_sources"] == 1


@pytest.mark.parametrize("flags", [[], [True, False], [1], ["true"], [None]])
def test_group_completeness_requires_one_boolean_flag_per_source(flags) -> None:
    with pytest.raises(ValueError, match="ground_truth_complete"):
        end_to_end_metrics(
            ["b"], ["DIRECT"], [0.99], [{"b"}], [{"DIRECT"}],
            ground_truth_complete=flags,
        )


@pytest.mark.parametrize("confidences", [[float("nan")], [float("inf")], [[0.99]]])
def test_incomplete_truth_does_not_hide_invalid_confidence_input(confidences) -> None:
    with pytest.raises(ValueError, match="match_probabilities"):
        end_to_end_metrics(
            ["b"], ["DIRECT"], confidences, [{"b"}], [{"DIRECT"}],
        )


def test_empty_group_evaluation_has_no_accuracy_denominator() -> None:
    metrics = end_to_end_metrics([], [], [], [], [])
    assert metrics["total_sources"] == metrics["evaluated_sources"] == 0
    assert metrics["overall_end_to_end_accuracy"] is None
    assert metrics["auto_accept_precision"] is None
    assert metrics["coverage"] is None
    assert metrics["all_sources_auto_accept_coverage"] is None


def test_saved_models_reload_with_identical_pair_prediction(tmp_path) -> None:
    torch.manual_seed(7)
    config = MatcherConfig(
        max_length=12,
        char_embedding_dim=4,
        cnn_channels=3,
        cnn_kernel_sizes=(2, 3),
        field_embedding_dim=5,
        classifier_hidden_dim=7,
        classifier_second_hidden_dim=4,
        inference_batch_size=2,
        batch_size=2,
        torch_num_threads=1,
    )
    vocab = CharVocabulary.build(["sale_price", "SalePrice", "customer_no"])
    encoder = FieldEncoder(
        vocab_size=len(vocab),
        char_embedding_dim=config.char_embedding_dim,
        cnn_channels=config.cnn_channels,
        kernel_sizes=config.cnn_kernel_sizes,
        output_dim=config.field_embedding_dim,
        padding_idx=vocab.pad_id,
    )
    classifier = PairClassifier(
        embedding_dim=config.field_embedding_dim,
        hidden_dim=config.classifier_hidden_dim,
        second_hidden_dim=config.classifier_second_hidden_dim,
        dropout=config.dropout,
    )
    config.save(tmp_path / "config.json")
    vocab.save(tmp_path / "vocab.json")
    (tmp_path / "label_map.json").write_text(
        json.dumps({"NO_MATCH": 0, "DIRECT": 1, "DERIVATION": 2}),
        encoding="utf-8",
    )
    torch.save(encoder.state_dict(), tmp_path / "encoder.pt")
    torch.save(classifier.state_dict(), tmp_path / "classifier.pt")

    first_bundle = load_model_bundle(tmp_path)
    second_bundle = load_model_bundle(tmp_path)
    first = predict_pair(first_bundle, "sale_price", "SalePrice")
    second = predict_pair(second_bundle, "sale_price", "SalePrice")

    assert first["predicted_label"] == second["predicted_label"]
    np.testing.assert_allclose(
        [first["p_no_match"], first["p_direct"], first["p_derivation"]],
        [second["p_no_match"], second["p_direct"], second["p_derivation"]],
        rtol=0.0,
        atol=0.0,
    )
    with pytest.raises(ValueError, match="batch_size must be positive"):
        predict_pairs(first_bundle, ["sale_price"], ["SalePrice"], batch_size=0)

    (tmp_path / "label_map.json").write_text(
        json.dumps({"NO_MATCH": 1, "DIRECT": 0, "DERIVATION": 2}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="fixed mapping"):
        load_model_bundle(tmp_path)
    (tmp_path / "label_map.json").unlink()
    with pytest.raises(FileNotFoundError, match="Required model artifact"):
        load_model_bundle(tmp_path)
