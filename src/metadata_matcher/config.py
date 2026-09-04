"""Configuration for the metadata field matcher.

The project intentionally uses a plain JSON-backed dataclass so that it works in
restricted environments without an additional configuration dependency.
"""

from __future__ import annotations

import json
import math
from numbers import Real
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class MatcherConfig:
    """All tunable training and inference parameters.

    Defaults are conservative CPU-oriented values.  A run stores the fully
    resolved configuration next to its weights, making inference independent of
    the original command line.
    """

    seed: int = 42
    max_length: int = 64
    char_embedding_dim: int = 32
    cnn_channels: int = 64
    cnn_kernel_sizes: tuple[int, ...] = (2, 3, 4, 5)
    field_embedding_dim: int = 128
    classifier_hidden_dim: int = 256
    classifier_second_hidden_dim: int = 64
    embedding_epochs: int = 15
    classifier_epochs: int = 15
    batch_size: int = 256
    inference_batch_size: int = 512
    learning_rate: float = 0.001
    classifier_learning_rate: float = 0.001
    weight_decay: float = 0.0001
    cosine_margin: float = 0.2
    dropout: float = 0.2
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 0.0
    fine_tune_encoder: bool = False
    random_negatives_per_positive: int = 1
    hard_negatives_per_positive: int = 1
    hard_negative_pool_size: int = 100
    top_k: int = 10
    review_threshold: float = 0.80
    auto_accept_threshold: float = 0.95
    retrieval_chunk_size: int = 4096
    num_workers: int = 0
    torch_num_threads: int = 8

    def __post_init__(self) -> None:
        """Validate values early so bad runs fail with actionable messages."""

        integer_values = {
            "seed": self.seed,
            "max_length": self.max_length,
            "char_embedding_dim": self.char_embedding_dim,
            "cnn_channels": self.cnn_channels,
            "field_embedding_dim": self.field_embedding_dim,
            "classifier_hidden_dim": self.classifier_hidden_dim,
            "classifier_second_hidden_dim": self.classifier_second_hidden_dim,
            "embedding_epochs": self.embedding_epochs,
            "classifier_epochs": self.classifier_epochs,
            "batch_size": self.batch_size,
            "inference_batch_size": self.inference_batch_size,
            "early_stopping_patience": self.early_stopping_patience,
            "random_negatives_per_positive": self.random_negatives_per_positive,
            "hard_negatives_per_positive": self.hard_negatives_per_positive,
            "hard_negative_pool_size": self.hard_negative_pool_size,
            "top_k": self.top_k,
            "retrieval_chunk_size": self.retrieval_chunk_size,
            "num_workers": self.num_workers,
            "torch_num_threads": self.torch_num_threads,
        }
        for name, value in integer_values.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if not isinstance(self.fine_tune_encoder, bool):
            raise TypeError("fine_tune_encoder must be a boolean")
        numeric_values = {
            "learning_rate": self.learning_rate,
            "classifier_learning_rate": self.classifier_learning_rate,
            "weight_decay": self.weight_decay,
            "cosine_margin": self.cosine_margin,
            "dropout": self.dropout,
            "early_stopping_min_delta": self.early_stopping_min_delta,
            "review_threshold": self.review_threshold,
            "auto_accept_threshold": self.auto_accept_threshold,
        }
        for name, value in numeric_values.items():
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if not isinstance(self.cnn_kernel_sizes, tuple):
            raise TypeError("cnn_kernel_sizes must be a tuple or a JSON array")
        if not self.cnn_kernel_sizes or any(
            isinstance(k, bool) or not isinstance(k, int) or k <= 0
            for k in self.cnn_kernel_sizes
        ):
            raise ValueError("cnn_kernel_sizes must contain positive integers")
        if self.max_length < max(self.cnn_kernel_sizes):
            raise ValueError("max_length must be >= the largest CNN kernel size")
        positive_ints = {
            "max_length": self.max_length,
            "char_embedding_dim": self.char_embedding_dim,
            "cnn_channels": self.cnn_channels,
            "field_embedding_dim": self.field_embedding_dim,
            "classifier_hidden_dim": self.classifier_hidden_dim,
            "classifier_second_hidden_dim": self.classifier_second_hidden_dim,
            "embedding_epochs": self.embedding_epochs,
            "classifier_epochs": self.classifier_epochs,
            "batch_size": self.batch_size,
            "inference_batch_size": self.inference_batch_size,
            "top_k": self.top_k,
            "retrieval_chunk_size": self.retrieval_chunk_size,
            "torch_num_threads": self.torch_num_threads,
            "hard_negative_pool_size": self.hard_negative_pool_size,
        }
        for name, value in positive_ints.items():
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero")
        if self.num_workers < 0:
            raise ValueError("num_workers must be >= 0")
        if self.early_stopping_patience < 0:
            raise ValueError("early_stopping_patience must be >= 0")
        if self.early_stopping_min_delta < 0:
            raise ValueError("early_stopping_min_delta must be >= 0")
        if self.random_negatives_per_positive < 0 or self.hard_negatives_per_positive < 0:
            raise ValueError("synthetic negative counts must be >= 0")
        for name, value in {
            "learning_rate": self.learning_rate,
            "classifier_learning_rate": self.classifier_learning_rate,
        }.items():
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be >= 0")
        if not -1.0 <= self.cosine_margin <= 1.0:
            raise ValueError("cosine_margin must be between -1 and 1")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not 0.0 <= self.review_threshold <= 1.0:
            raise ValueError("review_threshold must be in [0, 1]")
        if not 0.0 <= self.auto_accept_threshold <= 1.0:
            raise ValueError("auto_accept_threshold must be in [0, 1]")
        if self.auto_accept_threshold < self.review_threshold:
            raise ValueError("auto_accept_threshold must be >= review_threshold")

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "MatcherConfig":
        """Create a validated configuration from a mapping.

        Unknown keys are rejected because silently ignoring a misspelled training
        parameter is unsafe in a reproducible ML workflow.
        """

        valid_keys = {field.name for field in fields(cls)}
        unknown = sorted(set(values) - valid_keys)
        if unknown:
            raise ValueError(f"Unknown configuration keys: {', '.join(unknown)}")
        converted = dict(values)
        if "cnn_kernel_sizes" in converted:
            raw_kernels = converted["cnn_kernel_sizes"]
            if not isinstance(raw_kernels, (list, tuple)):
                raise TypeError("cnn_kernel_sizes must be a JSON array")
            converted["cnn_kernel_sizes"] = tuple(raw_kernels)
        return cls(**converted)

    @classmethod
    def load(cls, path: str | Path) -> "MatcherConfig":
        """Load configuration from a UTF-8 JSON file."""

        config_path = Path(path)
        try:
            with config_path.open("r", encoding="utf-8-sig") as handle:
                values = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in configuration {config_path}: {exc}") from exc
        if not isinstance(values, dict):
            raise ValueError("Configuration JSON must contain an object")
        return cls.from_dict(values)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        values = asdict(self)
        values["cnn_kernel_sizes"] = list(self.cnn_kernel_sizes)
        return values

    def save(self, path: str | Path) -> None:
        """Save the resolved configuration as human-readable JSON."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(self.to_dict(), handle, ensure_ascii=False, indent=2)
            handle.write("\n")
