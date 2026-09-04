"""Loss helpers for the encoder and pair-classifier training phases."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .dataset import LABEL_MAP


def cosine_targets_from_labels(
    labels: torch.Tensor, *, no_match_id: int = LABEL_MAP["NO_MATCH"]
) -> torch.Tensor:
    """Map relationship class IDs to CosineEmbeddingLoss targets.

    DIRECT and DERIVATION are positive matches (``+1``); NO_MATCH is a
    negative pair (``-1``).
    """

    if labels.ndim != 1:
        raise ValueError("labels must have shape [batch]")
    return torch.where(
        labels == no_match_id,
        torch.full_like(labels, -1, dtype=torch.float32),
        torch.full_like(labels, 1, dtype=torch.float32),
    )


def cosine_embedding_loss(
    embedding_a: torch.Tensor,
    embedding_b: torch.Tensor,
    labels: torch.Tensor,
    *,
    margin: float = 0.2,
    no_match_id: int = LABEL_MAP["NO_MATCH"],
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute the phase-one Siamese cosine embedding objective."""

    if embedding_a.shape != embedding_b.shape:
        raise ValueError("embedding_a and embedding_b must have equal shapes")
    if embedding_a.ndim != 2:
        raise ValueError("embeddings must have shape [batch, dimension]")
    if labels.ndim != 1 or labels.shape[0] != embedding_a.shape[0]:
        raise ValueError("labels must have shape [batch]")
    targets = cosine_targets_from_labels(labels, no_match_id=no_match_id).to(
        device=embedding_a.device, dtype=embedding_a.dtype
    )
    return F.cosine_embedding_loss(
        embedding_a,
        embedding_b,
        targets,
        margin=margin,
        reduction=reduction,
    )


def pair_classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the phase-two three-class cross-entropy loss."""

    return F.cross_entropy(logits, labels, weight=weight)


class CosineEmbeddingObjective(nn.Module):
    """Module wrapper around :func:`cosine_embedding_loss`."""

    def __init__(
        self,
        margin: float = 0.2,
        *,
        no_match_id: int = LABEL_MAP["NO_MATCH"],
        reduction: str = "mean",
    ) -> None:
        """Store a configurable cosine margin and reduction."""

        super().__init__()
        self.margin = margin
        self.no_match_id = no_match_id
        self.reduction = reduction

    def forward(
        self,
        embedding_a: torch.Tensor,
        embedding_b: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the configured cosine embedding objective."""

        return cosine_embedding_loss(
            embedding_a,
            embedding_b,
            labels,
            margin=self.margin,
            no_match_id=self.no_match_id,
            reduction=self.reduction,
        )

