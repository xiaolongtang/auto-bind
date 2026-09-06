"""Metadata matching with lazily loaded optional training dependencies.

Data preparation and normalization can run with the standard library alone.
"""

from importlib import import_module
from typing import Any

_EXPORTS = {
    "ID_TO_LABEL": "dataset",
    "LABEL_MAP": "dataset",
    "FieldDataset": "dataset",
    "PairDataset": "dataset",
    "build_training_vocabulary": "dataset",
    "load_pair_csv": "dataset",
    "CosineEmbeddingObjective": "losses",
    "cosine_embedding_loss": "losses",
    "cosine_targets_from_labels": "losses",
    "pair_classification_loss": "losses",
    "FieldEncoder": "model",
    "PairClassifier": "model",
    "SiameseFieldMatcher": "model",
    "SiameseModel": "model",
    "SYNTHETIC_WARNING": "negatives",
    "SyntheticNegativeConfig": "negatives",
    "augment_training_data_if_needed": "negatives",
    "contains_no_match": "negatives",
    "generate_synthetic_negatives": "negatives",
    "normalize_field_name": "preprocess",
    "normalize_field_names": "preprocess",
    "PAD_ID": "vocab",
    "PAD_TOKEN": "vocab",
    "UNK_ID": "vocab",
    "UNK_TOKEN": "vocab",
    "CharacterVocabulary": "vocab",
    "CharVocabulary": "vocab",
    "Vocabulary": "vocab",
}
__all__ = sorted(_EXPORTS)
__version__ = "0.1.0"


def __getattr__(name: str) -> Any:
    """Load public ML objects only when they are actually requested."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{_EXPORTS[name]}", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
