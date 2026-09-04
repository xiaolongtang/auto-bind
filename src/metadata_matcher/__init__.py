"""Local-only metadata field matching with a shared character CNN."""

from .dataset import (
    ID_TO_LABEL,
    LABEL_MAP,
    FieldDataset,
    PairDataset,
    build_training_vocabulary,
    load_pair_csv,
)
from .losses import (
    CosineEmbeddingObjective,
    cosine_embedding_loss,
    cosine_targets_from_labels,
    pair_classification_loss,
)
from .model import FieldEncoder, PairClassifier, SiameseFieldMatcher, SiameseModel
from .negatives import (
    SYNTHETIC_WARNING,
    SyntheticNegativeConfig,
    augment_training_data_if_needed,
    contains_no_match,
    generate_synthetic_negatives,
)
from .preprocess import normalize_field_name, normalize_field_names
from .vocab import (
    PAD_ID,
    PAD_TOKEN,
    UNK_ID,
    UNK_TOKEN,
    CharacterVocabulary,
    CharVocabulary,
    Vocabulary,
)

__all__ = [
    "CharacterVocabulary",
    "CharVocabulary",
    "CosineEmbeddingObjective",
    "FieldDataset",
    "FieldEncoder",
    "ID_TO_LABEL",
    "LABEL_MAP",
    "PAD_ID",
    "PAD_TOKEN",
    "PairClassifier",
    "PairDataset",
    "SYNTHETIC_WARNING",
    "SiameseFieldMatcher",
    "SiameseModel",
    "SyntheticNegativeConfig",
    "UNK_ID",
    "UNK_TOKEN",
    "Vocabulary",
    "augment_training_data_if_needed",
    "build_training_vocabulary",
    "contains_no_match",
    "cosine_embedding_loss",
    "cosine_targets_from_labels",
    "generate_synthetic_negatives",
    "load_pair_csv",
    "normalize_field_name",
    "normalize_field_names",
    "pair_classification_loss",
]

__version__ = "0.1.0"
