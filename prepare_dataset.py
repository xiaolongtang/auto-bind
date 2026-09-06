#!/usr/bin/env python3
"""Offline dataset preparation. Only the Python standard library is needed.

Example: python prepare_dataset.py --input data.csv --output-dir data_split --seed 42
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from metadata_matcher.prepare_dataset import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
