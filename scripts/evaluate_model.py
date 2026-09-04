#!/usr/bin/env python3
"""Evaluate pair, retrieval, and end-to-end matching quality."""

from __future__ import annotations

import _bootstrap  # noqa: F401

from metadata_matcher.evaluate import main


if __name__ == "__main__":
    raise SystemExit(main())
