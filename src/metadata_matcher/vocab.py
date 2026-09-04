"""Deterministic character vocabulary with explicit PAD and UNK tokens."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .preprocess import normalize_field_name


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_ID = 0
UNK_ID = 1


class CharVocabulary:
    """A serializable character vocabulary for local field-name encoding.

    Vocabulary construction is deterministic: reserved tokens come first and
    the remaining characters are sorted lexicographically.  To prevent data
    leakage, callers must pass training fields only to :meth:`build`.
    """

    def __init__(self, tokens: Sequence[str]) -> None:
        """Create a vocabulary from an ordered token sequence.

        Args:
            tokens: Tokens in ID order. The first two must be ``<PAD>`` and
                ``<UNK>``.

        Raises:
            ValueError: If reserved tokens are missing/misordered, tokens are
                duplicated, or a non-reserved token is not one character.
        """

        token_list = list(tokens)
        if len(token_list) < 2 or token_list[:2] != [PAD_TOKEN, UNK_TOKEN]:
            raise ValueError(
                f"tokens must start with {PAD_TOKEN!r}, {UNK_TOKEN!r}"
            )
        if len(set(token_list)) != len(token_list):
            raise ValueError("vocabulary tokens must be unique")
        invalid = [token for token in token_list[2:] if len(token) != 1]
        if invalid:
            raise ValueError("all non-reserved vocabulary tokens must be characters")

        self._id_to_token = tuple(token_list)
        self._token_to_id = {
            token: token_id for token_id, token in enumerate(self._id_to_token)
        }

    @classmethod
    def build(
        cls,
        fields: Iterable[str],
        *,
        min_frequency: int = 1,
        normalize: bool = True,
    ) -> "CharVocabulary":
        """Build a vocabulary from training field names only.

        Args:
            fields: Training field names. Validation and test fields must not
                be supplied because doing so leaks information across splits.
            min_frequency: Minimum training frequency for a character.
            normalize: Apply :func:`normalize_field_name` before counting.

        Returns:
            A deterministic character vocabulary.
        """

        if min_frequency < 1:
            raise ValueError("min_frequency must be at least 1")

        counts: Counter[str] = Counter()
        for field in fields:
            text = normalize_field_name(field) if normalize else field
            if not isinstance(text, str):
                raise TypeError("fields must contain strings")
            counts.update(text)

        characters = sorted(
            character
            for character, frequency in counts.items()
            if frequency >= min_frequency
            and character not in {PAD_TOKEN, UNK_TOKEN}
        )
        return cls([PAD_TOKEN, UNK_TOKEN, *characters])

    @classmethod
    def from_fields(
        cls,
        fields: Iterable[str],
        *,
        min_frequency: int = 1,
        normalize: bool = True,
    ) -> "CharVocabulary":
        """Alias for :meth:`build` for readability at call sites."""

        return cls.build(
            fields, min_frequency=min_frequency, normalize=normalize
        )

    @property
    def pad_id(self) -> int:
        """Return the fixed padding token ID."""

        return PAD_ID

    @property
    def unk_id(self) -> int:
        """Return the fixed unknown-character token ID."""

        return UNK_ID

    @property
    def size(self) -> int:
        """Return the vocabulary size."""

        return len(self)

    @property
    def token_to_id(self) -> dict[str, int]:
        """Return a copy of the token-to-ID mapping."""

        return dict(self._token_to_id)

    @property
    def id_to_token(self) -> tuple[str, ...]:
        """Return tokens in ID order."""

        return self._id_to_token

    @property
    def stoi(self) -> dict[str, int]:
        """Compatibility alias for the token-to-ID mapping."""

        return self.token_to_id

    @property
    def itos(self) -> tuple[str, ...]:
        """Compatibility alias for tokens in ID order."""

        return self.id_to_token

    def __len__(self) -> int:
        """Return the number of vocabulary tokens."""

        return len(self._id_to_token)

    def __contains__(self, character: object) -> bool:
        """Return whether a token exists in the vocabulary."""

        return character in self._token_to_id

    def encode(
        self,
        text: str,
        max_length: int,
        *,
        normalize: bool = True,
    ) -> list[int]:
        """Encode, truncate, and right-pad one field name.

        Args:
            text: Field name to encode.
            max_length: Exact output sequence length.
            normalize: Normalize the field before character lookup.

        Returns:
            A list of exactly ``max_length`` token IDs. Characters unseen in
            training map to ``UNK_ID``.
        """

        if max_length < 1:
            raise ValueError("max_length must be at least 1")
        encoded_text = normalize_field_name(text) if normalize else text
        if not isinstance(encoded_text, str):
            raise TypeError("text must be a string")

        token_ids = [
            self._token_to_id.get(character, UNK_ID)
            for character in encoded_text[:max_length]
        ]
        token_ids.extend([PAD_ID] * (max_length - len(token_ids)))
        return token_ids

    def decode(self, token_ids: Iterable[int], *, skip_pad: bool = True) -> str:
        """Decode token IDs to a diagnostic string.

        Unknown IDs and the explicit ``UNK_ID`` are represented by ``<UNK>``.
        This helper is intended for inspection, not lossless round trips.
        """

        output: list[str] = []
        for token_id in token_ids:
            if skip_pad and token_id == PAD_ID:
                continue
            if 0 <= token_id < len(self):
                output.append(self._id_to_token[token_id])
            else:
                output.append(UNK_TOKEN)
        return "".join(output)

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON representation of this vocabulary."""

        return {
            "version": 1,
            "pad_token": PAD_TOKEN,
            "unk_token": UNK_TOKEN,
            "tokens": list(self._id_to_token),
        }

    def save(self, path: str | Path) -> None:
        """Save the vocabulary as UTF-8 JSON."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "CharVocabulary":
        """Load and validate a vocabulary JSON file."""

        source = Path(path)
        payload = json.loads(source.read_text(encoding="utf-8-sig"))
        if not isinstance(payload, Mapping):
            raise ValueError("vocabulary JSON must contain an object")

        # ``tokens`` is the canonical format. ``token_to_id`` is accepted for
        # compatibility with early internal artifacts.
        if "tokens" in payload:
            raw_tokens = payload["tokens"]
            if not isinstance(raw_tokens, list) or not all(
                isinstance(token, str) for token in raw_tokens
            ):
                raise ValueError("vocabulary 'tokens' must be a list of strings")
            tokens = raw_tokens
        elif "token_to_id" in payload:
            mapping = payload["token_to_id"]
            if not isinstance(mapping, Mapping):
                raise ValueError("vocabulary 'token_to_id' must be an object")
            try:
                indexed = sorted(
                    ((int(token_id), str(token)) for token, token_id in mapping.items()),
                    key=lambda item: item[0],
                )
            except (TypeError, ValueError) as exc:
                raise ValueError("vocabulary IDs must be integers") from exc
            if [token_id for token_id, _ in indexed] != list(range(len(indexed))):
                raise ValueError("vocabulary IDs must be contiguous from zero")
            tokens = [token for _, token in indexed]
        else:
            raise ValueError("vocabulary JSON is missing 'tokens'")

        return cls(tokens)


# Concise aliases make the type convenient to import without creating a second
# implementation or a second token contract.
CharacterVocabulary = CharVocabulary
Vocabulary = CharVocabulary

