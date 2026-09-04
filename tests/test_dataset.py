"""Tests for pair datasets and training-only negative generation."""

import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from metadata_matcher.dataset import LABEL_MAP, PairDataset, build_training_vocabulary
from metadata_matcher.negatives import (
    augment_training_data_if_needed,
    generate_synthetic_negatives,
)
from metadata_matcher.preprocess import normalize_field_name


def _training_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "A": ["sale_price", "customer_no", "trade_date"],
            "B": ["SalePrice", "client_id", "settlement_date"],
            "label": ["DIRECT", "DIRECT", "DERIVATION"],
        }
    )


def test_pair_dataset_shapes_and_label_mapping() -> None:
    dataframe = _training_frame()
    vocab = build_training_vocabulary(dataframe)
    dataset = PairDataset(dataframe, vocab, max_length=16)

    sample = dataset[2]
    assert set(sample) == {"a_ids", "b_ids", "label"}
    assert sample["a_ids"].shape == (16,)
    assert sample["b_ids"].shape == (16,)
    assert sample["a_ids"].dtype == torch.long
    assert sample["label"].shape == ()
    assert sample["label"].item() == LABEL_MAP["DERIVATION"]

    batch = next(iter(DataLoader(dataset, batch_size=2)))
    assert batch["a_ids"].shape == (2, 16)
    assert batch["b_ids"].shape == (2, 16)
    assert batch["label"].shape == (2,)


def test_pair_dataset_rejects_unknown_label() -> None:
    dataframe = _training_frame()
    dataframe.loc[0, "label"] = "MAYBE"
    vocab = build_training_vocabulary(dataframe)
    with pytest.raises(ValueError, match="unsupported labels"):
        PairDataset(dataframe, vocab)


def test_vocabulary_does_not_see_validation_characters() -> None:
    train = pd.DataFrame({"A": ["abc"], "B": ["abd"], "label": ["DIRECT"]})
    validation = pd.DataFrame(
        {"A": ["abc"], "B": ["ab€"], "label": ["DIRECT"]}
    )
    vocab = build_training_vocabulary(train)
    dataset = PairDataset(validation, vocab, max_length=3)

    assert dataset[0]["b_ids"][-1].item() == vocab.unk_id


def test_synthetic_negatives_exclude_all_known_positive_pairs() -> None:
    dataframe = _training_frame()
    generated = generate_synthetic_negatives(
        dataframe,
        random_negatives_per_positive=1,
        hard_negatives_per_positive=1,
        seed=7,
    )
    known_positive_pairs = {
        (normalize_field_name(row.A), normalize_field_name(row.B))
        for row in dataframe.itertuples()
    }

    assert len(generated) == 6
    assert set(generated["label"]) == {"NO_MATCH"}
    assert not any(
        (normalize_field_name(row.A), normalize_field_name(row.B))
        in known_positive_pairs
        for row in generated.itertuples()
    )


def test_synthetic_negative_pool_is_seeded_and_bounded(monkeypatch) -> None:
    rows = 150
    dataframe = pd.DataFrame(
        {
            "A": ["source_field"],
            "B": ["correct_target"],
            "label": ["DIRECT"],
        }
    )
    # NO_MATCH rows contribute legitimate target candidates but are never used
    # as positive anchors by the generator.
    distractors = pd.DataFrame(
        {
            "A": [f"other_{index}" for index in range(rows)],
            "B": [f"target_{index}" for index in range(rows)],
            "label": ["NO_MATCH"] * rows,
        }
    )
    dataframe = pd.concat([dataframe, distractors], ignore_index=True)

    calls = 0

    class CountingMatcher:
        def __init__(self, _junk, left: str, right: str) -> None:
            nonlocal calls
            calls += 1
            self.left = left
            self.right = right

        def ratio(self) -> float:
            return float(len(set(self.left).intersection(self.right)))

    monkeypatch.setattr("metadata_matcher.negatives.SequenceMatcher", CountingMatcher)
    first = generate_synthetic_negatives(
        dataframe,
        random_negatives_per_positive=0,
        hard_negatives_per_positive=1,
        hard_negative_pool_size=7,
        seed=19,
    )
    second = generate_synthetic_negatives(
        dataframe,
        random_negatives_per_positive=0,
        hard_negatives_per_positive=1,
        hard_negative_pool_size=7,
        seed=19,
    )

    pd.testing.assert_frame_equal(first, second)
    assert calls == 14


def test_augmentation_only_happens_without_human_no_match() -> None:
    positives = _training_frame()
    augmented, generated = augment_training_data_if_needed(positives, seed=5)
    assert generated is True
    assert len(augmented) > len(positives)

    human_negative = pd.concat(
        [
            positives,
            pd.DataFrame(
                {"A": ["sale_price"], "B": ["employee_id"], "label": ["NO_MATCH"]}
            ),
        ],
        ignore_index=True,
    )
    untouched, generated = augment_training_data_if_needed(human_negative)
    assert generated is False
    pd.testing.assert_frame_equal(untouched, human_negative)
