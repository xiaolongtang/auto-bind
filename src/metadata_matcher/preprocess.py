"""Conservative normalization for metadata field names."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable


_ACRONYM_BOUNDARY = re.compile(r"(?<=[A-Z])(?=[A-Z][a-z])")
_CAMEL_CASE_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_SEPARATORS = re.compile(r"[-.\s]+")
_REPEATED_UNDERSCORES = re.compile(r"_+")


def normalize_field_name(field_name: str) -> str:
    """Normalize a field name without inventing semantic equivalences.

    The transformation applies Unicode NFKC normalization, trims surrounding
    whitespace, splits common ASCII CamelCase boundaries, lowercases the text,
    converts hyphens, dots, and whitespace to underscores, collapses repeated
    underscores, and removes leading/trailing underscores.

    Words are deliberately not stemmed, expanded, or replaced.  For example,
    ``customer`` and ``client`` remain different strings and their relationship
    must be learned from training data.

    Args:
        field_name: Raw metadata field name.

    Returns:
        The normalized field name. An empty or separator-only input becomes an
        empty string.

    Raises:
        TypeError: If ``field_name`` is not a string.
    """

    if not isinstance(field_name, str):
        raise TypeError(
            f"field_name must be a string, got {type(field_name).__name__}"
        )

    normalized = unicodedata.normalize("NFKC", field_name).strip()
    normalized = _ACRONYM_BOUNDARY.sub("_", normalized)
    normalized = _CAMEL_CASE_BOUNDARY.sub("_", normalized)
    normalized = normalized.lower()
    normalized = _SEPARATORS.sub("_", normalized)
    normalized = _REPEATED_UNDERSCORES.sub("_", normalized)
    return normalized.strip("_")


def normalize_field_names(field_names: Iterable[str]) -> list[str]:
    """Normalize an iterable of field names while preserving input order."""

    return [normalize_field_name(field_name) for field_name in field_names]

