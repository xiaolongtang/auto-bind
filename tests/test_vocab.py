"""Tests for training-only character vocabulary behavior."""

import json

from metadata_matcher.vocab import PAD_ID, PAD_TOKEN, UNK_ID, UNK_TOKEN, CharVocabulary


def test_reserved_tokens_and_unknown_characters() -> None:
    vocab = CharVocabulary.build(["SalePrice", "customer_no"])

    assert vocab.id_to_token[:2] == (PAD_TOKEN, UNK_TOKEN)
    assert vocab.pad_id == PAD_ID == 0
    assert vocab.unk_id == UNK_ID == 1
    encoded = vocab.encode("sale€", max_length=8)
    assert encoded[4] == UNK_ID
    assert encoded[-1] == PAD_ID
    assert len(encoded) == 8


def test_vocab_is_deterministic_and_uses_normalized_train_text() -> None:
    first = CharVocabulary.build(["SalePrice", "sale-price"])
    second = CharVocabulary.build(["sale-price", "SalePrice"])

    assert first.id_to_token == second.id_to_token
    assert "_" in first
    assert "S" not in first
    assert "-" not in first


def test_validation_only_character_maps_to_unk() -> None:
    train_vocab = CharVocabulary.build(["abc"])
    assert train_vocab.encode("abz", max_length=3) == [
        train_vocab.stoi["a"],
        train_vocab.stoi["b"],
        UNK_ID,
    ]


def test_vocab_json_round_trip(tmp_path) -> None:
    vocab = CharVocabulary.build(["sale_price", "客户编号"])
    path = tmp_path / "vocab.json"

    vocab.save(path)
    loaded = CharVocabulary.load(path)

    assert loaded.id_to_token == vocab.id_to_token
    assert loaded.encode("客户", 8) == vocab.encode("客户", 8)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["tokens"][:2] == [PAD_TOKEN, UNK_TOKEN]

