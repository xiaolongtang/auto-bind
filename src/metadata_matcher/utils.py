"""Shared local-only utilities for reproducibility, logging, and artifacts."""

from __future__ import annotations

import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


LOGGER_NAME = "metadata_matcher"


def set_reproducible_seed(seed: int, torch_num_threads: int | None = None) -> None:
    """Seed Python, NumPy, and PyTorch and request deterministic CPU behavior."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch_num_threads is not None:
        torch.set_num_threads(torch_num_threads)
    # warn_only protects portability when a future PyTorch CPU operation has no
    # deterministic implementation while still selecting deterministic kernels.
    torch.use_deterministic_algorithms(True, warn_only=True)


def configure_logging(log_path: str | Path | None = None) -> logging.Logger:
    """Configure one concise console logger and, optionally, a UTF-8 file logger."""

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_path is not None:
        output_path = Path(log_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(output_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def create_run_directory(artifacts_dir: str | Path) -> Path:
    """Create a unique ``run_YYYYMMDD_HHMMSS`` artifact directory."""

    root = Path(artifacts_dir)
    root.mkdir(parents=True, exist_ok=True)
    base = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    suffix = 0
    while True:
        name = base if suffix == 0 else f"{base}_{suffix:02d}"
        candidate = root / name
        try:
            candidate.mkdir(parents=False)
            return candidate
        except FileExistsError:
            # Another local training process may create the same timestamped
            # directory between our check and mkdir. Retrying with a suffix is
            # atomic and keeps both runs intact.
            suffix += 1


def _json_default(value: Any) -> Any:
    """Convert common NumPy/PyTorch scalar containers for JSON output."""

    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: str | Path, values: Any) -> None:
    """Write a JSON artifact with stable, readable formatting."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(values, handle, ensure_ascii=False, indent=2, default=_json_default)
        handle.write("\n")


def read_json(path: str | Path) -> Any:
    """Read JSON using UTF-8 with optional BOM support."""

    with Path(path).open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def save_state_dict(model: torch.nn.Module, path: str | Path) -> None:
    """Persist only tensor weights, never a pickled model object."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)


def load_state_dict(
    model: torch.nn.Module,
    path: str | Path,
    *,
    strict: bool = True,
) -> torch.nn.Module:
    """Load CPU weights into an already constructed model safely."""

    weight_path = Path(path)
    try:
        state: Mapping[str, torch.Tensor] = torch.load(
            weight_path, map_location="cpu", weights_only=True
        )
    except TypeError:  # PyTorch versions before ``weights_only`` was added.
        state = torch.load(weight_path, map_location="cpu")
    model.load_state_dict(state, strict=strict)
    return model


def cpu_device() -> torch.device:
    """Return the explicit default device used by this CPU-only MVP."""

    return torch.device("cpu")
