"""Regression tests for annotation completeness and honest retrieval depth."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from metadata_matcher.config import MatcherConfig
from metadata_matcher.dataset import LABEL_MAP
from metadata_matcher.evaluate import (
    _load_evaluation_csv,
    build_arg_parser,
    evaluate_dataframe,
    evaluate_pair_classification,
    evaluate_retrieval_and_end_to_end,
)
from metadata_matcher.evaluation_truth import group_truth_completeness


@pytest.fixture
def bundle():
    return SimpleNamespace(
        config=MatcherConfig(top_k=2, torch_num_threads=1),
        label_map=LABEL_MAP,
    )


@pytest.fixture
def fixed_pipeline(monkeypatch):
    """Keep rankings and confidence fixed while changing only annotations."""

    def encode(fields, *args, **kwargs):
        return torch.ones((len(fields), 2))

    def rank(queries, targets, *, top_k, **kwargs):
        indices = torch.arange(min(top_k, len(targets))).repeat(len(queries), 1)
        return torch.ones_like(indices, dtype=torch.float32), indices

    def score(bundle, sources, targets, indices, **kwargs):
        result = torch.tensor([0.99, 0.009, 0.001]).repeat(*indices.shape, 1)
        # customer_no gets a low-confidence candidate; other selects client_id.
        result[0, 0] = torch.tensor([0.90, 0.09, 0.01])
        if len(sources) > 1:
            result[1, 1] = torch.tensor([0.01, 0.98, 0.01])
        return result

    monkeypatch.setattr("metadata_matcher.evaluate.encode_field_names", encode)
    monkeypatch.setattr("metadata_matcher.evaluate.cosine_top_k", rank)
    monkeypatch.setattr("metadata_matcher.evaluate._score_retrieved_candidates", score)


def _partial_truth():
    return pd.DataFrame(
        [("customer_no", "employee_id", "NO_MATCH"), ("other", "client_id", "DIRECT")],
        columns=["A", "B", "label"],
    )


def test_pair_negative_is_not_group_no_match(bundle, fixed_pipeline):
    retrieval, e2e = evaluate_retrieval_and_end_to_end(bundle, _partial_truth())
    assert e2e["no_match_only_sources"] == 0
    assert e2e["unknown_no_match_sources"] == 1
    assert e2e["evaluated_sources"] == 0
    assert e2e["excluded_incomplete_sources"] == 2
    assert e2e["overall_end_to_end_accuracy"] is None
    assert e2e["auto_accept_precision"] is None
    assert retrieval["evaluated_sources"] == 1
    assert retrieval["MRR@20"] == pytest.approx(0.5)
    assert "MRR" not in retrieval and "mrr" not in retrieval


def test_only_complete_sources_contribute_group_metrics(bundle, fixed_pipeline):
    frame = _partial_truth().assign(group_truth_complete=[False, True])
    _, e2e = evaluate_retrieval_and_end_to_end(bundle, frame)
    assert e2e["evaluated_sources"] == 1
    assert e2e["excluded_incomplete_sources"] == 1
    assert e2e["overall_end_to_end_accuracy"] == 1.0
    assert e2e["coverage"] == 1.0
    assert e2e["no_match_only_sources"] == 0


def test_explicitly_complete_negative_can_be_correctly_rejected(bundle, fixed_pipeline):
    frame = _partial_truth().assign(group_truth_complete=True)
    _, e2e = evaluate_retrieval_and_end_to_end(bundle, frame)
    assert e2e["evaluated_sources"] == 2
    assert e2e["no_match_only_sources"] == 1
    assert e2e["unknown_no_match_sources"] == 0
    assert e2e["overall_end_to_end_accuracy"] == 1.0
    assert e2e["coverage"] == 0.5


def test_completeness_csv_loading_and_normalized_dedup(tmp_path, bundle, fixed_pipeline):
    frame = pd.DataFrame(
        [("CustomerNo", "EmployeeId", "NO_MATCH"),
         ("customer_no", "employee-id", "NO_MATCH")],
        columns=["A", "B", "label"],
    ).assign(group_truth_complete=" TRUE ")
    path = tmp_path / "truth.csv"
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    loaded = _load_evaluation_csv(path)
    assert group_truth_completeness(loaded) == {"customer_no": True}
    retrieval, e2e = evaluate_retrieval_and_end_to_end(bundle, loaded)
    assert retrieval["candidate_targets"] == 1
    assert retrieval["retrieval_depth_effective"] == 1
    assert retrieval["ranking_is_complete"] is True
    assert e2e["total_sources"] == 1
    assert e2e["no_match_only_sources"] == 1


@pytest.mark.parametrize("value", ["", "yes", "unknown", "1", 1, None])
def test_invalid_completeness_is_rejected(value):
    with pytest.raises(ValueError, match="group_truth_complete.*CSV row 2"):
        group_truth_completeness(_partial_truth().assign(group_truth_complete=value))


def test_completeness_must_agree_for_normalized_source():
    frame = pd.DataFrame({
        "A": ["CustomerNo", "customer_no"],
        "group_truth_complete": [True, False],
    })
    with pytest.raises(ValueError, match="inconsistent.*CSV rows 2 and 3"):
        group_truth_completeness(frame)


@pytest.mark.parametrize("entrypoint", [evaluate_dataframe, evaluate_pair_classification])
def test_evaluation_rejects_normalized_label_conflict_before_loading_model(entrypoint):
    frame = pd.DataFrame({
        "A": ["sale_price", "SalePrice"],
        "B": ["SalePrice", "sale-price"],
        "label": ["DIRECT", "NO_MATCH"],
    })
    with pytest.raises(ValueError, match="conflicting labels after normalization"):
        entrypoint("/nonexistent/model", frame)


def test_mrr_depth_is_independent_of_reranking(bundle, monkeypatch):
    fields = [f"target_{i}" for i in range(1, 26)]
    frame = pd.DataFrame({
        "A": ["query"] * 25,
        "B": fields,
        "label": ["NO_MATCH"] * 24 + ["DIRECT"],
        "group_truth_complete": [True] * 25,
    })

    def encode(names, *args, **kwargs):
        if names == ["query"]:
            return torch.tensor([[1.0, 0.0]])
        cosine = torch.arange(25, 0, -1, dtype=torch.float32) / 26
        return torch.stack([cosine, torch.sqrt(1 - cosine.square())], dim=1)

    def score(bundle, sources, targets, indices, **kwargs):
        return torch.tensor([0.9, 0.09, 0.01]).repeat(*indices.shape, 1)

    monkeypatch.setattr("metadata_matcher.evaluate.encode_field_names", encode)
    monkeypatch.setattr("metadata_matcher.evaluate._score_retrieved_candidates", score)
    shallow, shallow_e2e = evaluate_retrieval_and_end_to_end(bundle, frame)
    deep, deep_e2e = evaluate_retrieval_and_end_to_end(bundle, frame, retrieval_depth=25)
    assert shallow["MRR@20"] == 0.0
    assert shallow["retrieval_depth_effective"] == 20
    assert shallow["ranking_is_complete"] is False
    assert deep["MRR@25"] == pytest.approx(1 / 25)
    assert deep["retrieval_depth_effective"] == 25
    assert deep["ranking_is_complete"] is True
    assert shallow_e2e["retrieval_top_k_for_reranking"] == 2
    assert deep_e2e["retrieval_top_k_for_reranking"] == 2


@pytest.mark.parametrize("depth", [0, 19, 20.5, True])
def test_invalid_retrieval_depth_fails_before_encoding(bundle, depth):
    with pytest.raises(ValueError, match="retrieval_depth must be an integer >= 20"):
        evaluate_retrieval_and_end_to_end(bundle, _partial_truth(), retrieval_depth=depth)


def test_empty_evaluation_has_no_complete_metrics(bundle):
    retrieval, e2e = evaluate_retrieval_and_end_to_end(
        bundle, pd.DataFrame(columns=["A", "B", "label"])
    )
    assert retrieval["mrr_cutoff"] == 20
    assert retrieval["retrieval_depth_effective"] == 0
    assert e2e["evaluated_sources"] == 0
    assert e2e["overall_end_to_end_accuracy"] is None


def test_cli_exposes_retrieval_depth():
    args = build_arg_parser().parse_args([
        "--model-dir", "run", "--test", "test.csv", "--retrieval-depth", "50",
    ])
    assert args.retrieval_depth == 50
