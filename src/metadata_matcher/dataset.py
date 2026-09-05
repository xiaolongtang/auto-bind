"""CSV loading and PyTorch datasets for metadata field pairs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from numbers import Real
from pathlib import Path
from typing import TypedDict

import pandas as pd
import torch
from torch.utils.data import Dataset

from .preprocess import normalize_field_name
from .vocab import CharVocabulary


LABEL_MAP: dict[str, int] = {
    "NO_MATCH": 0,
    "DIRECT": 1,
    "DERIVATION": 2,
}
ID_TO_LABEL: dict[int, str] = {value: key for key, value in LABEL_MAP.items()}


class PairSample(TypedDict):
    """One fixed-length pair sample returned by :class:`PairDataset`."""

    a_ids: torch.Tensor
    b_ids: torch.Tensor
    label: torch.Tensor
    sample_weight: torch.Tensor


def validate_normalized_pair_labels(
    dataframe: pd.DataFrame,
    *,
    source_column: str = "A",
    target_column: str = "B",
    label_column: str = "label",
    context: str = "pair dataframe",
) -> None:
    """Reject contradictory labels for the same ordered normalized pair.

    Row numbers are one-based CSV record numbers, including the header. Case
    and separator variants with the same label remain valid duplicate records.
    """

    first_seen: dict[tuple[str, str], tuple[str, int, str, str]] = {}
    conflicts: list[str] = []
    values = dataframe[[source_column, target_column, label_column]]
    for row_number, (source, target, label) in enumerate(
        values.itertuples(index=False, name=None), start=2
    ):
        source, target = str(source), str(target)
        label = str(label).strip().upper()
        pair = (normalize_field_name(source), normalize_field_name(target))
        previous = first_seen.get(pair)
        if previous is None:
            first_seen[pair] = (label, row_number, source, target)
        elif previous[0] != label:
            conflicts.append(
                f"normalized pair {pair!r}: CSV row {previous[1]} "
                f"({previous[2]!r}, {previous[3]!r})={previous[0]} conflicts "
                f"with CSV row {row_number} ({source!r}, {target!r})={label}"
            )
            if len(conflicts) == 10:
                break
    if conflicts:
        raise ValueError(
            f"{context} has conflicting labels after normalization; "
            + "; ".join(conflicts)
            + ". Correct the original annotations before training or evaluation."
        )


def load_pair_csv(
    path: str | Path,
    *,
    require_label: bool = True,
    source_column: str = "A",
    target_column: str = "B",
    label_column: str = "label",
) -> pd.DataFrame:
    """Load a UTF-8 pair CSV and validate its required columns.

    ``utf-8-sig`` accepts ordinary UTF-8 as well as files containing a BOM.
    All columns are loaded as strings so field names such as ``001_code`` are
    not coerced by pandas.
    """

    dataframe = pd.read_csv(
        Path(path), encoding="utf-8-sig", dtype=str, keep_default_na=False
    )
    required = {source_column, target_column}
    if require_label:
        required.add(label_column)
    missing = sorted(required.difference(dataframe.columns))
    if missing:
        raise ValueError(f"pair CSV is missing required columns: {missing}")
    return dataframe


def build_training_vocabulary(
    train_dataframe: pd.DataFrame,
    *,
    source_column: str = "A",
    target_column: str = "B",
    min_frequency: int = 1,
) -> CharVocabulary:
    """Build a character vocabulary from a training dataframe only.

    The function's explicit ``train_dataframe`` name is intentional: callers
    must not concatenate validation or test data when building the vocabulary.
    """

    missing = {source_column, target_column}.difference(train_dataframe.columns)
    if missing:
        raise ValueError(f"training dataframe is missing columns: {sorted(missing)}")
    if train_dataframe[[source_column, target_column]].isna().any().any():
        raise ValueError("training field names must not be missing")
    fields = [
        *train_dataframe[source_column].astype(str).tolist(),
        *train_dataframe[target_column].astype(str).tolist(),
    ]
    return CharVocabulary.build(fields, min_frequency=min_frequency)


class PairDataset(Dataset[PairSample]):
    """Fixed-length encoded field pairs and integer relationship labels.

    Each item is a dictionary with ``a_ids`` and ``b_ids`` tensors of shape
    ``[max_length]`` and scalar ``label`` / ``sample_weight`` tensors. PyTorch's
    default collate produces ``[batch, max_length]`` and ``[batch]``. Only rows
    explicitly marked ``synthetic=True`` receive the configured negative weight.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        vocab: CharVocabulary,
        max_length: int = 64,
        *,
        label_map: Mapping[str, int] | None = None,
        source_column: str = "A",
        target_column: str = "B",
        label_column: str = "label",
        synthetic_negative_weight: float = 1.0,
    ) -> None:
        """Validate and retain pair data for on-demand tensor encoding."""

        if max_length < 1:
            raise ValueError("max_length must be at least 1")
        if (
            isinstance(synthetic_negative_weight, bool)
            or not isinstance(synthetic_negative_weight, Real)
            or not math.isfinite(synthetic_negative_weight)
            or not 0.0 < synthetic_negative_weight <= 1.0
        ):
            raise ValueError("synthetic_negative_weight must be finite and in (0, 1]")
        required = {source_column, target_column, label_column}
        missing = sorted(required.difference(dataframe.columns))
        if missing:
            raise ValueError(f"pair dataframe is missing required columns: {missing}")
        if dataframe[list(required)].isna().any().any():
            raise ValueError("pair dataframe contains missing A, B, or label values")

        effective_label_map = dict(label_map or LABEL_MAP)
        if set(effective_label_map.values()) != set(range(len(effective_label_map))):
            raise ValueError("label_map IDs must be contiguous from zero")

        a_values = dataframe[source_column].astype(str).tolist()
        b_values = dataframe[target_column].astype(str).tolist()
        normalized_a = [normalize_field_name(value) for value in a_values]
        normalized_b = [normalize_field_name(value) for value in b_values]
        empty_rows = [
            index
            for index, (a_value, b_value) in enumerate(zip(normalized_a, normalized_b))
            if not a_value or not b_value
        ]
        if empty_rows:
            preview = empty_rows[:5]
            raise ValueError(f"field names normalize to empty strings at rows {preview}")

        labels = [str(value).strip().upper() for value in dataframe[label_column]]
        unknown_labels = sorted(set(labels).difference(effective_label_map))
        if unknown_labels:
            raise ValueError(
                f"unsupported labels {unknown_labels}; expected "
                f"{sorted(effective_label_map)}"
            )
        validate_normalized_pair_labels(
            dataframe,
            source_column=source_column,
            target_column=target_column,
            label_column=label_column,
        )
        if "synthetic" in dataframe:
            flags = dataframe["synthetic"].astype(str).str.strip().str.lower()
            if not flags.isin({"true", "false"}).all():
                raise ValueError("synthetic markers must be true or false, without missing values")
            synthetic_flags = flags.eq("true").tolist()
        else:
            synthetic_flags = [False] * len(dataframe)
        if any(
            is_synthetic and label != "NO_MATCH"
            for is_synthetic, label in zip(synthetic_flags, labels)
        ):
            raise ValueError("only NO_MATCH rows may be marked synthetic")

        self.vocab = vocab
        self.max_length = max_length
        self.label_map = effective_label_map
        self.dataframe = dataframe.reset_index(drop=True).copy()
        self.a_fields = normalized_a
        self.b_fields = normalized_b
        self.labels = [effective_label_map[label] for label in labels]
        self.sample_weights = [
            float(synthetic_negative_weight) if synthetic else 1.0
            for synthetic in synthetic_flags
        ]

    def __len__(self) -> int:
        """Return the number of labeled pairs."""

        return len(self.labels)

    def __getitem__(self, index: int) -> PairSample:
        """Encode and return one labeled pair."""

        a_ids = self.vocab.encode(
            self.a_fields[index], self.max_length, normalize=False
        )
        b_ids = self.vocab.encode(
            self.b_fields[index], self.max_length, normalize=False
        )
        return {
            "a_ids": torch.tensor(a_ids, dtype=torch.long),
            "b_ids": torch.tensor(b_ids, dtype=torch.long),
            "label": torch.tensor(self.labels[index], dtype=torch.long),
            "sample_weight": torch.tensor(self.sample_weights[index], dtype=torch.float32),
        }


class FieldDataset(Dataset[torch.Tensor]):
    """Fixed-length character tensors for batched retrieval inference."""

    def __init__(
        self,
        fields: Sequence[str],
        vocab: CharVocabulary,
        max_length: int = 64,
    ) -> None:
        """Create an inference dataset while preserving field order."""

        if max_length < 1:
            raise ValueError("max_length must be at least 1")
        self.fields = [normalize_field_name(field) for field in fields]
        if any(not field for field in self.fields):
            raise ValueError("field names must not normalize to empty strings")
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self) -> int:
        """Return the number of field names."""

        return len(self.fields)

    def __getitem__(self, index: int) -> torch.Tensor:
        """Return one encoded field tensor of shape ``[max_length]``."""

        return torch.tensor(
            self.vocab.encode(
                self.fields[index], self.max_length, normalize=False
            ),
            dtype=torch.long,
        )
