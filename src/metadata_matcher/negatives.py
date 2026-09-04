"""Training-only random and name-similar synthetic negative generation."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from difflib import SequenceMatcher

import pandas as pd

from .dataset import LABEL_MAP
from .preprocess import normalize_field_name


LOGGER = logging.getLogger(__name__)
POSITIVE_LABELS = frozenset({"DIRECT", "DERIVATION"})
SYNTHETIC_WARNING = (
    "Synthetic negatives should only be used for training and are not "
    "equivalent to human-verified NO_MATCH evaluation data."
)


@dataclass(frozen=True)
class SyntheticNegativeConfig:
    """Configuration for training-only negative sampling."""

    random_negatives_per_positive: int = 1
    hard_negatives_per_positive: int = 1
    hard_negative_pool_size: int = 100
    seed: int = 42

    def __post_init__(self) -> None:
        """Reject invalid negative counts early."""

        if self.random_negatives_per_positive < 0:
            raise ValueError("random_negatives_per_positive must be non-negative")
        if self.hard_negatives_per_positive < 0:
            raise ValueError("hard_negatives_per_positive must be non-negative")
        if self.hard_negative_pool_size < 1:
            raise ValueError("hard_negative_pool_size must be positive")


def contains_no_match(
    dataframe: pd.DataFrame, *, label_column: str = "label"
) -> bool:
    """Return whether a dataframe contains a human-provided NO_MATCH row."""

    if label_column not in dataframe.columns:
        raise ValueError(f"dataframe is missing label column {label_column!r}")
    labels = dataframe[label_column].astype(str).str.strip().str.upper()
    return bool(labels.eq("NO_MATCH").any())


def _validate_columns(
    dataframe: pd.DataFrame,
    source_column: str,
    target_column: str,
    label_column: str,
) -> None:
    required = {source_column, target_column, label_column}
    missing = sorted(required.difference(dataframe.columns))
    if missing:
        raise ValueError(f"training dataframe is missing required columns: {missing}")
    if dataframe[list(required)].isna().any().any():
        raise ValueError("training dataframe contains missing A, B, or label values")
    labels = dataframe[label_column].astype(str).str.strip().str.upper()
    unknown = sorted(set(labels).difference(LABEL_MAP))
    if unknown:
        raise ValueError(f"unsupported training labels: {unknown}")


def _repeat_sample(
    candidates: list[str], count: int, rng: random.Random
) -> list[str]:
    """Sample candidates deterministically, cycling only when count is large."""

    if not candidates or count == 0:
        return []
    selected: list[str] = []
    while len(selected) < count:
        shuffled = list(candidates)
        rng.shuffle(shuffled)
        selected.extend(shuffled)
    return selected[:count]


def _sample_eligible_targets(
    all_targets: tuple[str, ...],
    excluded: set[str],
    count: int,
    rng: random.Random,
) -> list[str]:
    """Sample unique eligible targets without scanning a large target set.

    Rejection sampling is efficient for normal mapping data where a source has
    one or a handful of positive targets. If exclusions cover at least half the
    target universe, a single scan avoids a pathological rejection loop.
    """

    if count <= 0 or not all_targets:
        return []
    # Every exclusion originates from the same target universe, so no full
    # membership scan is needed merely to determine the eligible population.
    eligible_count = len(all_targets) - len(excluded)
    if eligible_count <= 0:
        return []
    sample_size = min(count, eligible_count)

    if len(excluded) * 2 >= len(all_targets):
        eligible = [target for target in all_targets if target not in excluded]
        if len(eligible) <= sample_size:
            return eligible
        return rng.sample(eligible, sample_size)

    selected: list[str] = []
    selected_set: set[str] = set()
    while len(selected) < sample_size:
        candidate = all_targets[rng.randrange(len(all_targets))]
        if candidate in excluded or candidate in selected_set:
            continue
        selected.append(candidate)
        selected_set.add(candidate)
    return selected


def generate_synthetic_negatives(
    train_dataframe: pd.DataFrame,
    *,
    random_negatives_per_positive: int = 1,
    hard_negatives_per_positive: int = 1,
    hard_negative_pool_size: int = 100,
    seed: int = 42,
    source_column: str = "A",
    target_column: str = "B",
    label_column: str = "label",
) -> pd.DataFrame:
    """Generate synthetic ``NO_MATCH`` rows from training data only.

    For every positive training row, random negatives select another target
    with a seeded RNG. Hard negatives select targets whose normalized name has
    the highest :class:`difflib.SequenceMatcher` similarity within a bounded,
    seeded candidate pool. The pool prevents a full target scan per positive.
    Every known DIRECT/DERIVATION target for the same normalized source is
    excluded from both strategies.

    This function returns generated rows only. Use
    :func:`augment_training_data_if_needed` to conditionally append them when a
    training split has no human-verified ``NO_MATCH`` examples. Never call this
    function for validation or test splits.
    """

    config = SyntheticNegativeConfig(
        random_negatives_per_positive=random_negatives_per_positive,
        hard_negatives_per_positive=hard_negatives_per_positive,
        hard_negative_pool_size=hard_negative_pool_size,
        seed=seed,
    )
    _validate_columns(
        train_dataframe, source_column, target_column, label_column
    )
    LOGGER.warning(SYNTHETIC_WARNING)

    working = train_dataframe[[source_column, target_column, label_column]].copy()
    working[label_column] = (
        working[label_column].astype(str).str.strip().str.upper()
    )
    working["_norm_a"] = working[source_column].astype(str).map(normalize_field_name)
    working["_norm_b"] = working[target_column].astype(str).map(normalize_field_name)

    positive_rows = working[working[label_column].isin(POSITIVE_LABELS)]
    if positive_rows.empty:
        return pd.DataFrame(columns=[source_column, target_column, label_column])

    known_positive_targets: dict[str, set[str]] = {}
    for row in positive_rows.itertuples(index=False, name=None):
        # The selected dataframe's tuple order is A, B, label, norm_a, norm_b.
        norm_a = row[3]
        norm_b = row[4]
        known_positive_targets.setdefault(norm_a, set()).add(norm_b)

    # Preserve the first original spelling of each normalized target. This
    # prevents separator/case variants of the same B from becoming duplicates.
    target_by_normalized: dict[str, str] = {}
    for target in working[target_column].astype(str):
        target_by_normalized.setdefault(normalize_field_name(target), target)
    all_targets = tuple(target_by_normalized)

    synthetic_rows: list[dict[str, str]] = []
    rng = random.Random(config.seed)
    for row in positive_rows.itertuples(index=False, name=None):
        source = str(row[0])
        norm_a = str(row[3])
        excluded = known_positive_targets[norm_a]
        eligible_count = len(all_targets) - len(excluded)
        if eligible_count <= 0:
            LOGGER.warning(
                "Could not generate a synthetic negative for source %r: no "
                "eligible alternative targets exist.",
                source,
            )
            continue

        hard_pool = (
            _sample_eligible_targets(
                all_targets,
                excluded,
                config.hard_negative_pool_size,
                rng,
            )
            if config.hard_negatives_per_positive
            else []
        )
        hard_ranked = sorted(
            hard_pool,
            key=lambda candidate: (
                -SequenceMatcher(None, norm_a, candidate).ratio(),
                candidate,
            ),
        )
        hard_targets = [
            hard_ranked[index % len(hard_ranked)]
            for index in range(config.hard_negatives_per_positive)
        ]
        for normalized_target in hard_targets:
            synthetic_rows.append(
                {
                    source_column: source,
                    target_column: target_by_normalized[normalized_target],
                    label_column: "NO_MATCH",
                }
            )

        # Prefer targets not already used as hard negatives when alternatives
        # exist, while preserving the requested random count.
        random_excluded = excluded.union(hard_targets)
        random_pool = _sample_eligible_targets(
            all_targets,
            random_excluded,
            config.random_negatives_per_positive,
            rng,
        )
        if not random_pool:
            random_pool = _sample_eligible_targets(
                all_targets,
                excluded,
                config.random_negatives_per_positive,
                rng,
            )
        random_targets = _repeat_sample(
            random_pool, config.random_negatives_per_positive, rng
        )
        for normalized_target in random_targets:
            synthetic_rows.append(
                {
                    source_column: source,
                    target_column: target_by_normalized[normalized_target],
                    label_column: "NO_MATCH",
                }
            )

    return pd.DataFrame(
        synthetic_rows, columns=[source_column, target_column, label_column]
    )


def augment_training_data_if_needed(
    train_dataframe: pd.DataFrame,
    *,
    random_negatives_per_positive: int = 1,
    hard_negatives_per_positive: int = 1,
    hard_negative_pool_size: int = 100,
    seed: int = 42,
    source_column: str = "A",
    target_column: str = "B",
    label_column: str = "label",
) -> tuple[pd.DataFrame, bool]:
    """Append training-only negatives iff no ``NO_MATCH`` row is present.

    Returns:
        ``(dataframe, generated)`` where ``generated`` reports whether
        synthetic rows were appended. If human negatives exist, a defensive
        copy of the original dataframe and ``False`` are returned.
    """

    _validate_columns(
        train_dataframe, source_column, target_column, label_column
    )
    if contains_no_match(train_dataframe, label_column=label_column):
        return train_dataframe.copy(), False

    negatives = generate_synthetic_negatives(
        train_dataframe,
        random_negatives_per_positive=random_negatives_per_positive,
        hard_negatives_per_positive=hard_negatives_per_positive,
        hard_negative_pool_size=hard_negative_pool_size,
        seed=seed,
        source_column=source_column,
        target_column=target_column,
        label_column=label_column,
    )
    augmented = pd.concat([train_dataframe, negatives], ignore_index=True)
    return augmented, not negatives.empty
