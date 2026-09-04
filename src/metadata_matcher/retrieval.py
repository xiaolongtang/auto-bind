"""CPU-friendly embedding generation and exact cosine retrieval.

The project deliberately uses exact matrix multiplication instead of FAISS.  The
helpers in this module bound peak memory by chunking both queries and candidates,
while still returning the exact global top-k results.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _positive_int(name: str, value: int) -> int:
    """Validate an integer chunk/batch parameter."""

    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _as_2d_float_tensor(name: str, value: Tensor) -> Tensor:
    """Validate an embedding matrix without silently changing its device."""

    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 2:
        raise ValueError(
            f"{name} must have shape [rows, embedding_dim], got {tuple(value.shape)}"
        )
    if value.is_complex():
        raise TypeError(f"{name} must contain real-valued embeddings")
    if not value.is_floating_point():
        value = value.float()
    return value


def cosine_top_k(
    query_embeddings: Tensor,
    candidate_embeddings: Tensor,
    top_k: int = 10,
    *,
    query_chunk_size: int = 256,
    candidate_chunk_size: int = 4096,
) -> tuple[Tensor, Tensor]:
    """Return exact cosine-similarity top-k candidates for every query.

    Both matrices are normalized inside the function.  Therefore callers may pass
    either raw or already L2-normalized embeddings.  Candidate chunks are merged
    incrementally, so the largest score matrix has approximately
    ``query_chunk_size * candidate_chunk_size`` elements rather than
    ``num_queries * num_candidates``.

    Args:
        query_embeddings: Float tensor shaped ``[num_queries, embedding_dim]``.
        candidate_embeddings: Float tensor shaped
            ``[num_candidates, embedding_dim]``.
        top_k: Maximum number of results per query.  It is clipped to the number
            of candidates.
        query_chunk_size: Number of query rows processed at once.
        candidate_chunk_size: Number of candidate rows scored at once.

    Returns:
        ``(scores, indices)`` tensors, each shaped
        ``[num_queries, min(top_k, num_candidates)]``.  Scores are sorted in
        descending order and indices address rows in ``candidate_embeddings``.

    Raises:
        ValueError: If dimensions or numeric parameters are invalid.
    """

    query_embeddings = _as_2d_float_tensor("query_embeddings", query_embeddings)
    candidate_embeddings = _as_2d_float_tensor(
        "candidate_embeddings", candidate_embeddings
    )
    _positive_int("top_k", top_k)
    _positive_int("query_chunk_size", query_chunk_size)
    _positive_int("candidate_chunk_size", candidate_chunk_size)

    if query_embeddings.shape[1] != candidate_embeddings.shape[1]:
        raise ValueError(
            "Embedding dimensions differ: "
            f"{query_embeddings.shape[1]} != {candidate_embeddings.shape[1]}"
        )
    if query_embeddings.device != candidate_embeddings.device:
        raise ValueError(
            "query_embeddings and candidate_embeddings must be on the same device"
        )

    num_queries = query_embeddings.shape[0]
    num_candidates = candidate_embeddings.shape[0]
    result_k = min(top_k, num_candidates)
    score_dtype = torch.promote_types(
        query_embeddings.dtype, candidate_embeddings.dtype
    )

    if num_queries == 0 or result_k == 0:
        return (
            torch.empty(
                (num_queries, result_k),
                dtype=score_dtype,
                device=query_embeddings.device,
            ),
            torch.empty(
                (num_queries, result_k),
                dtype=torch.long,
                device=query_embeddings.device,
            ),
        )

    # Float16 matrix multiplication on CPU is both less portable and less stable.
    # Upcast only for computation, preserving ordinary float32/float64 inputs.
    compute_dtype = score_dtype
    if query_embeddings.device.type == "cpu" and score_dtype in {
        torch.float16,
        torch.bfloat16,
    }:
        compute_dtype = torch.float32

    with torch.inference_mode():
        queries = F.normalize(query_embeddings.to(compute_dtype), p=2, dim=1)
        candidates = F.normalize(candidate_embeddings.to(compute_dtype), p=2, dim=1)
        all_scores: list[Tensor] = []
        all_indices: list[Tensor] = []

        for query_start in range(0, num_queries, query_chunk_size):
            query_end = min(query_start + query_chunk_size, num_queries)
            query_chunk = queries[query_start:query_end]
            best_scores: Tensor | None = None
            best_indices: Tensor | None = None

            for candidate_start in range(0, num_candidates, candidate_chunk_size):
                candidate_end = min(
                    candidate_start + candidate_chunk_size, num_candidates
                )
                candidate_chunk = candidates[candidate_start:candidate_end]
                chunk_scores = query_chunk @ candidate_chunk.transpose(0, 1)
                chunk_k = min(result_k, candidate_chunk.shape[0])
                local_scores, local_indices = torch.topk(
                    chunk_scores, k=chunk_k, dim=1, largest=True, sorted=True
                )
                local_indices = local_indices + candidate_start

                if best_scores is None:
                    best_scores = local_scores
                    best_indices = local_indices
                    continue

                merged_scores = torch.cat((best_scores, local_scores), dim=1)
                merged_indices = torch.cat((best_indices, local_indices), dim=1)
                merged_k = min(result_k, merged_scores.shape[1])
                best_scores, positions = torch.topk(
                    merged_scores, k=merged_k, dim=1, largest=True, sorted=True
                )
                best_indices = torch.gather(merged_indices, 1, positions)

            # num_candidates > 0 guarantees that at least one candidate chunk ran.
            assert best_scores is not None and best_indices is not None
            all_scores.append(best_scores)
            all_indices.append(best_indices)

    return torch.cat(all_scores, dim=0), torch.cat(all_indices, dim=0)


def encode_in_batches(
    token_ids: Tensor,
    encoder: nn.Module,
    *,
    batch_size: int = 256,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Encode padded token IDs in batches and return a CPU embedding matrix.

    Returning CPU tensors is intentional: group matching is CPU-first and keeping
    the cached target matrix on CPU prevents an accidental large accelerator
    allocation if callers opt into another device in the future.
    """

    if not isinstance(token_ids, Tensor) or token_ids.ndim != 2:
        shape = tuple(token_ids.shape) if isinstance(token_ids, Tensor) else None
        raise ValueError(f"token_ids must be a 2-D tensor, got shape {shape}")
    _positive_int("batch_size", batch_size)
    resolved_device = torch.device(device)

    if token_ids.shape[0] == 0:
        # Infer the configured output size when possible.  Consumers generally do
        # not need the second dimension for an empty source list, but retaining it
        # makes the result more useful and predictable.
        embedding_dim = int(
            getattr(
                encoder,
                "embedding_dim",
                getattr(encoder, "field_embedding_dim", 0),
            )
        )
        return torch.empty((0, embedding_dim), dtype=torch.float32)

    encoder = encoder.to(resolved_device)
    encoder.eval()
    batches: list[Tensor] = []
    with torch.inference_mode():
        for start in range(0, token_ids.shape[0], batch_size):
            batch = token_ids[start : start + batch_size].to(
                resolved_device, non_blocking=False
            )
            output = encoder(batch)
            if not isinstance(output, Tensor) or output.ndim != 2:
                raise ValueError(
                    "Field encoder must return a 2-D tensor shaped "
                    "[batch_size, embedding_dim]"
                )
            batches.append(F.normalize(output, p=2, dim=1).cpu())
    return torch.cat(batches, dim=0)


def encode_fields(
    fields: Sequence[str],
    encoder: nn.Module,
    encode_field: Callable[[str], Sequence[int] | Tensor],
    *,
    batch_size: int = 256,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Tokenize and encode field names while bounding inference batch memory.

    ``encode_field`` should return an already padded one-dimensional ID sequence.
    This small callback keeps retrieval independent of a particular vocabulary
    implementation and is also convenient for tests and downstream integrations.
    """

    rows: list[Tensor] = []
    expected_length: int | None = None
    for field in fields:
        encoded: Any = encode_field(str(field))
        row = (
            encoded.detach().cpu()
            if isinstance(encoded, Tensor)
            else torch.tensor(encoded)
        )
        row = row.to(dtype=torch.long).flatten()
        if expected_length is None:
            expected_length = int(row.numel())
        elif row.numel() != expected_length:
            raise ValueError(
                "encode_field must return equally padded sequences; "
                f"expected length {expected_length}, got {row.numel()}"
            )
        rows.append(row)

    if not rows:
        return encode_in_batches(
            torch.empty((0, 0), dtype=torch.long),
            encoder,
            batch_size=batch_size,
            device=device,
        )
    return encode_in_batches(
        torch.stack(rows, dim=0), encoder, batch_size=batch_size, device=device
    )


# Descriptive aliases retained for library users; both names have identical exact
# semantics (they are not approximate retrieval implementations).
chunked_cosine_top_k = cosine_top_k
top_k_cosine_similarity = cosine_top_k


__all__ = [
    "chunked_cosine_top_k",
    "cosine_top_k",
    "encode_fields",
    "encode_in_batches",
    "top_k_cosine_similarity",
]
