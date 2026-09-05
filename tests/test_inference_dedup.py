"""Normalized target identities must agree between evaluation and inference."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from metadata_matcher.config import MatcherConfig
from metadata_matcher.dataset import LABEL_MAP
from metadata_matcher.evaluate import evaluate_retrieval_and_end_to_end
from metadata_matcher.predict import match_field_groups
from metadata_matcher.preprocess import normalize_field_name


@pytest.fixture
def deterministic_pipeline(monkeypatch):
    """Wrong targets rank ahead of the true target, but the MLP rejects them."""

    bundle = SimpleNamespace(
        config=MatcherConfig(top_k=2, torch_num_threads=1), label_map=LABEL_MAP,
    )
    encoded_batches = []

    def encode(fields, *args, **kwargs):
        encoded_batches.append(list(fields))
        return torch.tensor([
            [0.9, (1 - 0.9**2) ** 0.5]
            if normalize_field_name(field) == "right_target" else [1.0, 0.0]
            for field in fields
        ])

    def score(bundle, sources, targets, indices, **kwargs):
        return torch.stack([
            torch.stack([
                torch.tensor([0.01, 0.98, 0.01])
                if targets[index, 1] > 0.1 else torch.tensor([0.9, 0.09, 0.01])
                for index in row
            ])
            for row in indices
        ])

    monkeypatch.setattr("metadata_matcher.predict._ensure_bundle", lambda *a, **kw: bundle)
    for module in ("predict", "evaluate"):
        monkeypatch.setattr(f"metadata_matcher.{module}.encode_field_names", encode)
        monkeypatch.setattr(f"metadata_matcher.{module}._score_retrieved_candidates", score)
    return bundle, encoded_batches


@pytest.mark.parametrize("top_k", [2, 20])
def test_target_aliases_do_not_consume_top_k_and_source_rows_are_preserved(
    deterministic_pipeline, top_k,
):
    bundle, encoded_batches = deterministic_pipeline
    source_fields = ["query", "Query", "query"]
    target_fields = ["WrongValue", "wrong_value", "RightTarget", "right-target"]

    predictions, candidates = match_field_groups(
        bundle, source_fields, target_fields, top_k=top_k, include_candidates=True,
    )

    # Without deduplication, the first two aliases exhaust top_k=2 and the real
    # match never reaches reranking. Both depths must now find the true target.
    assert predictions["A"].tolist() == source_fields
    assert predictions["best_B"].tolist() == ["RightTarget"] * 3
    assert predictions["predicted_label"].tolist() == ["DIRECT"] * 3
    assert predictions["status"].tolist() == ["AUTO_ACCEPT"] * 3
    assert predictions["rank"].tolist() == [2] * 3
    assert encoded_batches == [source_fields, ["WrongValue", "RightTarget"]]
    assert candidates is not None
    assert len(candidates) == 3 * 2
    assert candidates["candidate_B"].tolist() == ["WrongValue", "RightTarget"] * 3
    assert candidates["retrieval_rank"].tolist() == [1, 2] * 3

    # The same normalized candidate universe produces the same outcome offline.
    frame = pd.DataFrame({
        "A": ["query"] * 4,
        "B": target_fields,
        "label": ["NO_MATCH", "NO_MATCH", "DIRECT", "DIRECT"],
        "group_truth_complete": [True] * 4,
    })
    retrieval, e2e = evaluate_retrieval_and_end_to_end(bundle, frame, top_k=top_k)
    assert retrieval["candidate_targets"] == 2
    assert e2e["retrieval_top_k_for_reranking"] == 2
    assert e2e["overall_end_to_end_accuracy"] == 1.0


def test_single_normalized_target_retains_first_original_spelling(
    deterministic_pipeline,
):
    bundle, encoded_batches = deterministic_pipeline
    first_spelling = "ＲｉｇｈｔＴａｒｇｅｔ"
    predictions, candidates = match_field_groups(
        bundle, ["query"], [first_spelling, "right_target", "RightTarget"],
        top_k=10, include_candidates=True,
    )

    assert encoded_batches == [["query"], [first_spelling]]
    assert predictions["best_B"].tolist() == [first_spelling]
    assert predictions["rank"].tolist() == [1]
    assert candidates is not None
    assert candidates["candidate_B"].tolist() == [first_spelling]
    assert candidates["retrieval_rank"].tolist() == [1]
