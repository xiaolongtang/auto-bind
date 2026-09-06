#!/usr/bin/env python3
"""Script-directory entry point for the offline preparation CLI."""

import _bootstrap  # noqa: F401

from metadata_matcher.prepare_dataset import main


if __name__ == "__main__":
    raise SystemExit(main())
