"""Tests for conservative metadata field-name normalization."""

import pytest

from metadata_matcher.preprocess import normalize_field_name


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("SalePrice", "sale_price"),
        ("sale-price", "sale_price"),
        ("sale.price", "sale_price"),
        ("SettlementDate", "settlement_date"),
        ("  sale   price  ", "sale_price"),
        ("__sale___price__", "sale_price"),
        ("HTTPResponseCode", "http_response_code"),
        ("ＳａｌｅＰｒｉｃｅ", "sale_price"),
        ("customer", "customer"),
        ("client", "client"),
    ],
)
def test_normalize_field_name(raw: str, expected: str) -> None:
    assert normalize_field_name(raw) == expected


def test_normalization_is_idempotent() -> None:
    normalized = normalize_field_name("  Settlement--Date. ")
    assert normalize_field_name(normalized) == normalized


def test_normalization_rejects_non_string() -> None:
    with pytest.raises(TypeError, match="must be a string"):
        normalize_field_name(None)  # type: ignore[arg-type]

