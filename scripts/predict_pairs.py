#!/usr/bin/env python3
"""Predict one pair or a UTF-8 CSV of A/B pairs."""

from __future__ import annotations

import _bootstrap  # noqa: F401

from metadata_matcher.predict import main


if __name__ == "__main__":
    raise SystemExit(main())

