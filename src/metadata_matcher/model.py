"""Shared character CNN encoder and pair relationship classifier."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class FieldEncoder(nn.Module):
    """Encode field names with a CPU-friendly multi-kernel character CNN."""

    def __init__(
        self,
        vocab_size: int,
        char_embedding_dim: int = 32,
        cnn_channels: int = 64,
        kernel_sizes: Sequence[int] = (2, 3, 4, 5),
        output_dim: int = 128,
        padding_idx: int = 0,
    ) -> None:
        """Initialize the character embedding, convolutions, and projection.

        Args:
            vocab_size: Number of character tokens, including PAD and UNK.
            char_embedding_dim: Dimension of each character embedding.
            cnn_channels: Output channels for every convolution kernel.
            kernel_sizes: Positive Conv1D window sizes.
            output_dim: Final L2-normalized field embedding dimension.
            padding_idx: Vocabulary ID used for right padding.
        """

        super().__init__()
        if vocab_size < 2:
            raise ValueError("vocab_size must include at least PAD and UNK")
        if char_embedding_dim < 1 or cnn_channels < 1 or output_dim < 1:
            raise ValueError("model dimensions must be positive")
        kernels = tuple(int(kernel_size) for kernel_size in kernel_sizes)
        if not kernels or any(kernel_size < 1 for kernel_size in kernels):
            raise ValueError("kernel_sizes must contain positive integers")
        if not 0 <= padding_idx < vocab_size:
            raise ValueError("padding_idx must be a valid vocabulary ID")

        self.output_dim = output_dim
        self.embedding_dim = output_dim
        self.kernel_sizes = kernels
        self.padding_idx = padding_idx
        self.char_embedding = nn.Embedding(
            vocab_size,
            char_embedding_dim,
            padding_idx=padding_idx,
        )
        self.convolutions = nn.ModuleList(
            nn.Conv1d(
                in_channels=char_embedding_dim,
                out_channels=cnn_channels,
                kernel_size=kernel_size,
            )
            for kernel_size in kernels
        )
        self.projection = nn.Linear(cnn_channels * len(kernels), output_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Return L2-normalized embeddings with shape ``[batch, output_dim]``.

        Very short sequences are right-padded internally to the largest kernel,
        so direct calls are safe even outside a fixed-length dataset. Training
        config should still set ``max_length >= max(kernel_sizes)`` so useful
        characters are not unnecessarily truncated.
        """

        if token_ids.ndim != 2:
            raise ValueError(
                "FieldEncoder expects token_ids with shape [batch, sequence]"
            )
        if token_ids.shape[1] == 0:
            raise ValueError("FieldEncoder sequence length must be at least one")
        if token_ids.dtype not in {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }:
            raise TypeError("FieldEncoder token_ids must have an integer dtype")

        embedded = self.char_embedding(token_ids).transpose(1, 2)
        required_padding = max(0, max(self.kernel_sizes) - embedded.shape[-1])
        if required_padding:
            embedded = F.pad(embedded, (0, required_padding))

        pooled_features = []
        for convolution in self.convolutions:
            activated = F.relu(convolution(embedded))
            pooled_features.append(torch.amax(activated, dim=-1))
        concatenated = torch.cat(pooled_features, dim=-1)
        projected = self.projection(concatenated)
        return F.normalize(projected, p=2, dim=-1)


class PairClassifier(nn.Module):
    """Classify a pair as NO_MATCH, DIRECT, or DERIVATION."""

    def __init__(
        self,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        second_hidden_dim: int = 64,
        dropout: float = 0.2,
        num_classes: int = 3,
    ) -> None:
        """Initialize the MLP over ``a, b, |a-b|, a*b`` pair features."""

        super().__init__()
        if min(embedding_dim, hidden_dim, second_hidden_dim, num_classes) < 1:
            raise ValueError("classifier dimensions must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.network = nn.Sequential(
            nn.Linear(embedding_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, second_hidden_dim),
            nn.ReLU(),
            nn.Linear(second_hidden_dim, num_classes),
        )

    @staticmethod
    def pair_features(
        embedding_a: torch.Tensor, embedding_b: torch.Tensor
    ) -> torch.Tensor:
        """Build symmetric-comparison features while retaining pair order."""

        if embedding_a.shape != embedding_b.shape:
            raise ValueError("embedding_a and embedding_b must have equal shapes")
        if embedding_a.ndim != 2:
            raise ValueError("pair embeddings must have shape [batch, dimension]")
        return torch.cat(
            (
                embedding_a,
                embedding_b,
                torch.abs(embedding_a - embedding_b),
                embedding_a * embedding_b,
            ),
            dim=-1,
        )

    def forward(
        self, embedding_a: torch.Tensor, embedding_b: torch.Tensor
    ) -> torch.Tensor:
        """Return unnormalized class logits with shape ``[batch, 3]``."""

        if embedding_a.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"expected embedding dimension {self.embedding_dim}, "
                f"got {embedding_a.shape[-1]}"
            )
        features = self.pair_features(embedding_a, embedding_b)
        return self.network(features)


class SiameseFieldMatcher(nn.Module):
    """Compose exactly one shared encoder with a pair classifier."""

    def __init__(
        self, encoder: FieldEncoder, classifier: PairClassifier
    ) -> None:
        """Create a matcher and validate encoder/classifier dimensions."""

        super().__init__()
        if encoder.output_dim != classifier.embedding_dim:
            raise ValueError(
                "encoder output_dim must equal classifier embedding_dim"
            )
        self.encoder = encoder
        self.classifier = classifier

    def encode(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Encode field IDs using the one shared encoder instance."""

        return self.encoder(token_ids)

    def forward(
        self, a_ids: torch.Tensor, b_ids: torch.Tensor
    ) -> torch.Tensor:
        """Encode both sides with shared weights and return class logits."""

        embedding_a = self.encoder(a_ids)
        embedding_b = self.encoder(b_ids)
        return self.classifier(embedding_a, embedding_b)


# Short alias retained for callers that prefer architecture-centric naming.
SiameseModel = SiameseFieldMatcher

