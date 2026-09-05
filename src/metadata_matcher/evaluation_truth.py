"""Explicit source-level annotation completeness for group evaluation."""

from __future__ import annotations

import pandas as pd

from .preprocess import normalize_field_name


GROUP_TRUTH_COMPLETE_COLUMN = "group_truth_complete"


def group_truth_completeness(
    dataframe: pd.DataFrame, *, context: str = "evaluation dataframe"
) -> dict[str, bool]:
    """Return completeness by normalized A; absent annotations mean unknown.

    A true value asserts that all valid matches for this A in the entire B
    candidate group have been recorded. Only then does an empty positive set
    establish group-level NO_MATCH. Every row for the same normalized A must
    carry the same explicit true/false value when the column is supplied.
    """

    sources = dataframe["A"].astype(str).map(normalize_field_name).tolist()
    if GROUP_TRUTH_COMPLETE_COLUMN not in dataframe:
        return dict.fromkeys(sources, False)

    result: dict[str, bool] = {}
    first_rows: dict[str, int] = {}
    for row_number, (source, value) in enumerate(
        zip(sources, dataframe[GROUP_TRUTH_COMPLETE_COLUMN]), start=2
    ):
        marker = str(value).strip().lower()
        if marker not in {"true", "false"}:
            raise ValueError(
                f"{context}: {GROUP_TRUTH_COMPLETE_COLUMN} must be true or false "
                f"at CSV row {row_number}; got {value!r}"
            )
        complete = marker == "true"
        if source in result and result[source] != complete:
            raise ValueError(
                f"{context}: inconsistent {GROUP_TRUTH_COMPLETE_COLUMN} for "
                f"normalized A {source!r} at CSV rows {first_rows[source]} and "
                f"{row_number}; completeness must agree across all rows for an A"
            )
        result[source] = complete
        first_rows.setdefault(source, row_number)
    return result
