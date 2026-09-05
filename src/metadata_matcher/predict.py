"""Local CPU inference for individual pairs and A-group to B-group matching."""

from __future__ import annotations

import argparse
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn

from .config import MatcherConfig
from .metrics import DEFAULT_LABEL_MAP
from .preprocess import normalize_field_name
from .retrieval import cosine_top_k, encode_in_batches


PAIR_OUTPUT_COLUMNS = [
    "A",
    "B",
    "p_no_match",
    "p_direct",
    "p_derivation",
    "predicted_label",
    "match_probability",
]
GROUP_OUTPUT_COLUMNS = [
    "A",
    "best_B",
    "predicted_label",
    "match_probability",
    "p_direct",
    "p_derivation",
    "retrieval_similarity",
    "rank",
    "status",
]
CANDIDATE_OUTPUT_COLUMNS = [
    "A",
    "candidate_B",
    "retrieval_rank",
    "retrieval_similarity",
    "p_no_match",
    "p_direct",
    "p_derivation",
    "match_probability",
]


@dataclass
class ModelBundle:
    """All immutable artifacts required for local inference."""

    model_dir: Path
    config: MatcherConfig
    vocab: Any
    encoder: nn.Module
    classifier: nn.Module
    label_map: dict[str, int]
    device: torch.device

    @property
    def id_to_label(self) -> dict[int, str]:
        """Return the inverse label mapping saved with the training run."""

        return {class_id: name for name, class_id in self.label_map.items()}


def _read_json_object(path: Path) -> dict[str, Any]:
    """Read and validate one UTF-8/UTF-8-BOM JSON object."""

    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            value = json.load(handle)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Required model artifact is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _normalize_label_map(raw: Mapping[Any, Any]) -> dict[str, int]:
    """Accept either name-to-ID or ID-to-name JSON label maps."""

    if raw and all(isinstance(value, str) for value in raw.values()):
        normalized = {str(name): int(class_id) for class_id, name in raw.items()}
    else:
        normalized = {str(name): int(class_id) for name, class_id in raw.items()}
    missing = set(DEFAULT_LABEL_MAP) - set(normalized)
    if missing:
        raise ValueError(f"label_map.json is missing labels: {sorted(missing)!r}")
    expected_ids = set(range(len(normalized)))
    if set(normalized.values()) != expected_ids:
        raise ValueError(
            "label_map.json IDs must be unique and contiguous starting from zero"
        )
    return normalized


def _load_vocabulary(path: Path) -> Any:
    """Load the project's vocabulary while tolerating its documented class names."""

    from . import vocab as vocab_module

    vocabulary_class = None
    for class_name in ("FieldVocabulary", "CharVocabulary", "CharacterVocabulary"):
        vocabulary_class = getattr(vocab_module, class_name, None)
        if vocabulary_class is not None:
            break
    if vocabulary_class is None:
        raise ImportError(
            "metadata_matcher.vocab must define FieldVocabulary or CharVocabulary"
        )
    loader = getattr(vocabulary_class, "load", None)
    if not callable(loader):
        raise TypeError(f"{vocabulary_class.__name__} must provide classmethod load(path)")
    return loader(path)


def _vocabulary_size(vocab: Any) -> int:
    """Resolve the vocabulary size from common explicit APIs."""

    for attribute in ("size", "vocab_size"):
        value = getattr(vocab, attribute, None)
        if value is not None:
            return int(value() if callable(value) else value)
    try:
        return int(len(vocab))
    except TypeError:
        pass
    for attribute in ("char_to_id", "stoi", "token_to_id"):
        mapping = getattr(vocab, attribute, None)
        if mapping is not None:
            return len(mapping)
    raise TypeError("Could not determine vocabulary size")


def _padding_id(vocab: Any) -> int:
    """Resolve the PAD ID, defaulting to the project contract value zero."""

    for attribute in ("pad_id", "padding_idx", "PAD_ID"):
        value = getattr(vocab, attribute, None)
        if value is not None:
            return int(value() if callable(value) else value)
    return 0


def _construct_model(model_class: type[nn.Module], values: Mapping[str, Any]) -> nn.Module:
    """Pass only constructor keywords supported by a model implementation."""

    signature = inspect.signature(model_class)
    parameters = signature.parameters
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    ):
        supported = dict(values)
    else:
        supported = {name: value for name, value in values.items() if name in parameters}
    return model_class(**supported)


def _load_weights(model: nn.Module, path: Path) -> None:
    """Load a state dict on CPU, supporting raw and named checkpoint mappings."""

    if not path.is_file():
        raise FileNotFoundError(f"Required model artifact is missing: {path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # Compatibility with older PyTorch releases.
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"Expected a state dictionary in {path}")
    for key in ("state_dict", "model_state_dict", "encoder_state_dict", "classifier_state_dict"):
        nested = checkpoint.get(key)
        if isinstance(nested, Mapping):
            checkpoint = nested
            break
    state_dict = {
        (str(name)[7:] if str(name).startswith("module.") else str(name)): tensor
        for name, tensor in checkpoint.items()
    }
    model.load_state_dict(state_dict, strict=True)


def load_model_bundle(
    model_dir: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> ModelBundle:
    """Reconstruct models from portable artifacts and load their state dicts.

    No Python model object is unpickled and no network access is performed.
    ``device`` defaults explicitly to CPU; callers may opt into another PyTorch
    device without changing the saved artifacts.
    """

    from .model import FieldEncoder, PairClassifier

    directory = Path(model_dir).expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {directory}")
    config = MatcherConfig.load(directory / "config.json")
    torch.set_num_threads(config.torch_num_threads)
    vocab = _load_vocabulary(directory / "vocab.json")
    label_map_path = directory / "label_map.json"
    label_map = _normalize_label_map(_read_json_object(label_map_path))
    if label_map != DEFAULT_LABEL_MAP:
        raise ValueError(
            "label_map.json must use the fixed mapping "
            "NO_MATCH=0, DIRECT=1, DERIVATION=2"
        )

    encoder_values = {
        "vocab_size": _vocabulary_size(vocab),
        "char_embedding_dim": config.char_embedding_dim,
        "embedding_dim": config.char_embedding_dim,
        "cnn_channels": config.cnn_channels,
        "channels": config.cnn_channels,
        "kernel_sizes": config.cnn_kernel_sizes,
        "cnn_kernel_sizes": config.cnn_kernel_sizes,
        "field_embedding_dim": config.field_embedding_dim,
        "output_dim": config.field_embedding_dim,
        "padding_idx": _padding_id(vocab),
        "pad_id": _padding_id(vocab),
        "config": config,
    }
    classifier_values = {
        "embedding_dim": config.field_embedding_dim,
        "field_embedding_dim": config.field_embedding_dim,
        "hidden_dims": (
            config.classifier_hidden_dim,
            config.classifier_second_hidden_dim,
        ),
        "hidden_dim": config.classifier_hidden_dim,
        "second_hidden_dim": config.classifier_second_hidden_dim,
        "dropout": config.dropout,
        "num_classes": len(label_map),
        "config": config,
    }
    encoder = _construct_model(FieldEncoder, encoder_values)
    classifier = _construct_model(PairClassifier, classifier_values)
    _load_weights(encoder, directory / "encoder.pt")
    _load_weights(classifier, directory / "classifier.pt")

    resolved_device = torch.device(device)
    encoder.to(resolved_device).eval()
    classifier.to(resolved_device).eval()
    return ModelBundle(
        model_dir=directory,
        config=config,
        vocab=vocab,
        encoder=encoder,
        classifier=classifier,
        label_map=label_map,
        device=resolved_device,
    )


def _ensure_bundle(
    model: ModelBundle | str | Path,
    *,
    device: str | torch.device = "cpu",
) -> ModelBundle:
    """Allow public prediction APIs to accept a loaded bundle or directory."""

    return model if isinstance(model, ModelBundle) else load_model_bundle(model, device=device)


def _encode_with_vocab(vocab: Any, field: str, max_length: int) -> list[int]:
    """Normalize, tokenize, truncate, and pad one field name."""

    normalized = normalize_field_name(str(field))
    if not normalized:
        raise ValueError(f"Field name {field!r} normalizes to an empty string")
    encoder = getattr(vocab, "encode", None)
    if not callable(encoder):
        raise TypeError("Vocabulary must define encode(field, max_length)")
    signature = inspect.signature(encoder)
    if "max_length" in signature.parameters:
        encoded = encoder(normalized, max_length=max_length)
    elif "max_len" in signature.parameters:
        encoded = encoder(normalized, max_len=max_length)
    elif len(signature.parameters) >= 2:
        encoded = encoder(normalized, max_length)
    else:
        encoded = encoder(normalized)
    if isinstance(encoded, Tensor):
        ids = encoded.detach().cpu().to(dtype=torch.long).flatten().tolist()
    else:
        ids = [int(value) for value in encoded]
    pad_id = _padding_id(vocab)
    return (ids[:max_length] + [pad_id] * max(0, max_length - len(ids)))[:max_length]


def encode_field_names(
    fields: Sequence[str],
    bundle: ModelBundle,
    *,
    batch_size: int | None = None,
) -> Tensor:
    """Generate normalized field embeddings in bounded batches on CPU."""

    resolved_batch_size = (
        bundle.config.inference_batch_size if batch_size is None else int(batch_size)
    )
    if resolved_batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not fields:
        return encode_in_batches(
            torch.empty((0, bundle.config.max_length), dtype=torch.long),
            bundle.encoder,
            batch_size=resolved_batch_size,
            device=bundle.device,
        )

    # Token IDs are short-lived per batch.  This matters for enterprise datasets
    # with hundreds of thousands of pairs: a nested Python list for every row can
    # otherwise be much larger than the eventual float embedding matrix.
    outputs: list[Tensor] = []
    for start in range(0, len(fields), resolved_batch_size):
        batch_fields = fields[start : start + resolved_batch_size]
        token_ids = torch.tensor(
            [
                _encode_with_vocab(
                    bundle.vocab, field, bundle.config.max_length
                )
                for field in batch_fields
            ],
            dtype=torch.long,
        )
        outputs.append(
            encode_in_batches(
                token_ids,
                bundle.encoder,
                batch_size=resolved_batch_size,
                device=bundle.device,
            )
        )
    return torch.cat(outputs, dim=0)


def _classify_embeddings(
    bundle: ModelBundle,
    embeddings_a: Tensor,
    embeddings_b: Tensor,
    *,
    batch_size: int | None = None,
) -> Tensor:
    """Return CPU class probabilities for aligned embedding pairs."""

    if embeddings_a.ndim != 2 or embeddings_b.ndim != 2:
        raise ValueError("Pair embeddings must both be 2-D")
    if embeddings_a.shape != embeddings_b.shape:
        raise ValueError(
            "Pair embedding matrices must have identical shapes, got "
            f"{tuple(embeddings_a.shape)} and {tuple(embeddings_b.shape)}"
        )
    if embeddings_a.shape[0] == 0:
        return torch.empty((0, len(bundle.label_map)), dtype=torch.float32)

    resolved_batch_size = (
        bundle.config.inference_batch_size if batch_size is None else int(batch_size)
    )
    if resolved_batch_size <= 0:
        raise ValueError("batch_size must be positive")
    outputs: list[Tensor] = []
    bundle.classifier.eval()
    with torch.inference_mode():
        for start in range(0, embeddings_a.shape[0], resolved_batch_size):
            batch_a = embeddings_a[start : start + resolved_batch_size].to(bundle.device)
            batch_b = embeddings_b[start : start + resolved_batch_size].to(bundle.device)
            logits = bundle.classifier(batch_a, batch_b)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            if not isinstance(logits, Tensor) or logits.ndim != 2:
                raise ValueError("PairClassifier must return [batch_size, num_classes] logits")
            if logits.shape[1] != len(bundle.label_map):
                raise ValueError(
                    "PairClassifier output class count does not match label_map.json"
                )
            outputs.append(torch.softmax(logits, dim=1).cpu())
    return torch.cat(outputs, dim=0)


def predict_pairs(
    model: ModelBundle | str | Path,
    fields_a: Sequence[str],
    fields_b: Sequence[str],
    *,
    batch_size: int | None = None,
    device: str | torch.device = "cpu",
) -> pd.DataFrame:
    """Predict relationship probabilities for aligned A/B field pairs."""

    if len(fields_a) != len(fields_b):
        raise ValueError("fields_a and fields_b must have equal length")
    bundle = _ensure_bundle(model, device=device)
    raw_a = [str(value) for value in fields_a]
    raw_b = [str(value) for value in fields_b]
    if not raw_a:
        return pd.DataFrame(columns=PAIR_OUTPUT_COLUMNS)

    resolved_batch_size = (
        bundle.config.inference_batch_size if batch_size is None else int(batch_size)
    )
    if resolved_batch_size <= 0:
        raise ValueError("batch_size must be positive")
    id_to_label = bundle.id_to_label
    no_match_id = bundle.label_map["NO_MATCH"]
    direct_id = bundle.label_map["DIRECT"]
    derivation_id = bundle.label_map["DERIVATION"]

    rows = []
    for start in range(0, len(raw_a), resolved_batch_size):
        end = min(start + resolved_batch_size, len(raw_a))
        batch_a = raw_a[start:end]
        batch_b = raw_b[start:end]
        embeddings_a = encode_field_names(
            batch_a, bundle, batch_size=resolved_batch_size
        )
        embeddings_b = encode_field_names(
            batch_b, bundle, batch_size=resolved_batch_size
        )
        probabilities = _classify_embeddings(
            bundle,
            embeddings_a,
            embeddings_b,
            batch_size=resolved_batch_size,
        ).numpy()
        predicted_ids = probabilities.argmax(axis=1)
        for index, (field_a, field_b) in enumerate(zip(batch_a, batch_b)):
            rows.append(
                {
                    "A": field_a,
                    "B": field_b,
                    "p_no_match": float(probabilities[index, no_match_id]),
                    "p_direct": float(probabilities[index, direct_id]),
                    "p_derivation": float(probabilities[index, derivation_id]),
                    "predicted_label": id_to_label[int(predicted_ids[index])],
                    "match_probability": float(
                        probabilities[index, direct_id]
                        + probabilities[index, derivation_id]
                    ),
                }
            )
    return pd.DataFrame(rows, columns=PAIR_OUTPUT_COLUMNS)


def predict_pair(
    model: ModelBundle | str | Path,
    field_a: str,
    field_b: str,
    *,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Predict one pair and return a JSON-serializable record."""

    return predict_pairs(model, [field_a], [field_b], device=device).iloc[0].to_dict()


def _score_retrieved_candidates(
    bundle: ModelBundle,
    source_embeddings: Tensor,
    target_embeddings: Tensor,
    target_indices: Tensor,
    *,
    batch_size: int | None = None,
    query_chunk_size: int = 256,
) -> Tensor:
    """Classify only retrieved pairs without materializing every pair tensor."""

    if target_indices.ndim != 2:
        raise ValueError("target_indices must be shaped [num_sources, top_k]")
    num_sources, result_k = target_indices.shape
    probabilities = torch.empty(
        (num_sources, result_k, len(bundle.label_map)), dtype=torch.float32
    )
    if num_sources == 0 or result_k == 0:
        return probabilities
    if query_chunk_size <= 0:
        raise ValueError("query_chunk_size must be positive")

    for start in range(0, num_sources, query_chunk_size):
        end = min(start + query_chunk_size, num_sources)
        chunk_indices = target_indices[start:end]
        flattened_indices = chunk_indices.reshape(-1)
        repeated_sources = source_embeddings[start:end].repeat_interleave(
            result_k, dim=0
        )
        selected_targets = target_embeddings.index_select(0, flattened_indices)
        chunk_probabilities = _classify_embeddings(
            bundle,
            repeated_sources,
            selected_targets,
            batch_size=batch_size,
        )
        probabilities[start:end] = chunk_probabilities.reshape(
            end - start, result_k, len(bundle.label_map)
        )
    return probabilities


def match_field_groups(
    model: ModelBundle | str | Path,
    source_fields: Sequence[str],
    target_fields: Sequence[str],
    *,
    top_k: int | None = None,
    review_threshold: float | None = None,
    auto_accept_threshold: float | None = None,
    batch_size: int | None = None,
    query_chunk_size: int = 256,
    candidate_chunk_size: int | None = None,
    include_candidates: bool = False,
    device: str | torch.device = "cpu",
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Retrieve, rerank, and confidence-gate A-group to B-group matches.

    Target embeddings are computed once.  Exact cosine retrieval selects Top-K;
    only those candidates enter the MLP classifier.  Final candidates are ranked
    by ``P(DIRECT) + P(DERIVATION)`` as required by the MVP design.
    Target names are deduplicated after normalization, matching offline group
    evaluation; the first original spelling is retained for output. Source rows
    retain their original order and spelling, including repeated source names.

    Returns:
        A tuple ``(best_predictions, candidates_or_none)``.  Candidate rows retain
        retrieval rank and are produced only when ``include_candidates=True``.
    """

    bundle = _ensure_bundle(model, device=device)
    raw_sources = [str(value) for value in source_fields]
    target_by_normalized: dict[str, str] = {}
    for value in target_fields:
        raw_target = str(value)
        target_by_normalized.setdefault(normalize_field_name(raw_target), raw_target)
    raw_targets = list(target_by_normalized.values())
    resolved_top_k = bundle.config.top_k if top_k is None else int(top_k)
    resolved_review = (
        bundle.config.review_threshold
        if review_threshold is None
        else float(review_threshold)
    )
    resolved_auto_accept = (
        bundle.config.auto_accept_threshold
        if auto_accept_threshold is None
        else float(auto_accept_threshold)
    )
    if resolved_top_k <= 0:
        raise ValueError("top_k must be positive")
    if not 0.0 <= resolved_review <= 1.0:
        raise ValueError("review_threshold must be in [0, 1]")
    if not 0.0 <= resolved_auto_accept <= 1.0:
        raise ValueError("auto_accept_threshold must be in [0, 1]")
    if resolved_auto_accept < resolved_review:
        raise ValueError("auto_accept_threshold must be >= review_threshold")
    if query_chunk_size <= 0:
        raise ValueError("query_chunk_size must be positive")

    if not raw_sources:
        empty_predictions = pd.DataFrame(columns=GROUP_OUTPUT_COLUMNS)
        empty_candidates = (
            pd.DataFrame(columns=CANDIDATE_OUTPUT_COLUMNS)
            if include_candidates
            else None
        )
        return empty_predictions, empty_candidates
    if not raw_targets:
        no_match_rows = [
            {
                "A": source,
                "best_B": "",
                "predicted_label": "NO_MATCH",
                "match_probability": 0.0,
                "p_direct": 0.0,
                "p_derivation": 0.0,
                "retrieval_similarity": np.nan,
                "rank": 0,
                "status": "NO_MATCH",
            }
            for source in raw_sources
        ]
        empty_candidates = (
            pd.DataFrame(columns=CANDIDATE_OUTPUT_COLUMNS)
            if include_candidates
            else None
        )
        return (
            pd.DataFrame(no_match_rows, columns=GROUP_OUTPUT_COLUMNS),
            empty_candidates,
        )

    source_embeddings = encode_field_names(raw_sources, bundle, batch_size=batch_size)
    target_embeddings = encode_field_names(raw_targets, bundle, batch_size=batch_size)
    retrieval_scores, retrieval_indices = cosine_top_k(
        source_embeddings,
        target_embeddings,
        top_k=resolved_top_k,
        query_chunk_size=query_chunk_size,
        candidate_chunk_size=(
            bundle.config.retrieval_chunk_size
            if candidate_chunk_size is None
            else candidate_chunk_size
        ),
    )
    candidate_probabilities = _score_retrieved_candidates(
        bundle,
        source_embeddings,
        target_embeddings,
        retrieval_indices,
        batch_size=batch_size,
        query_chunk_size=query_chunk_size,
    )

    no_match_id = bundle.label_map["NO_MATCH"]
    direct_id = bundle.label_map["DIRECT"]
    derivation_id = bundle.label_map["DERIVATION"]
    best_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] | None = [] if include_candidates else None
    for source_index, source in enumerate(raw_sources):
        probabilities = candidate_probabilities[source_index]
        match_probabilities = (
            probabilities[:, direct_id] + probabilities[:, derivation_id]
        )
        # stable=True keeps retrieval order as the tie breaker.
        reranked = torch.argsort(
            match_probabilities, descending=True, stable=True
        )
        winner_position = int(reranked[0].item())
        target_index = int(retrieval_indices[source_index, winner_position].item())
        probability = float(match_probabilities[winner_position].item())
        p_direct = float(probabilities[winner_position, direct_id].item())
        p_derivation = float(probabilities[winner_position, derivation_id].item())
        relationship = "DIRECT" if p_direct > p_derivation else "DERIVATION"

        if probability < resolved_review:
            output_target = ""
            output_relationship = "NO_MATCH"
            status = "NO_MATCH"
        elif probability < resolved_auto_accept:
            output_target = raw_targets[target_index]
            output_relationship = relationship
            status = "REVIEW"
        else:
            output_target = raw_targets[target_index]
            output_relationship = relationship
            status = "AUTO_ACCEPT"
        best_rows.append(
            {
                "A": source,
                "best_B": output_target,
                "predicted_label": output_relationship,
                "match_probability": probability,
                "p_direct": p_direct,
                "p_derivation": p_derivation,
                "retrieval_similarity": float(
                    retrieval_scores[source_index, winner_position].item()
                ),
                "rank": winner_position + 1,
                "status": status,
            }
        )

        if candidate_rows is not None:
            for retrieval_position in range(retrieval_indices.shape[1]):
                candidate_target_index = int(
                    retrieval_indices[source_index, retrieval_position].item()
                )
                candidate_probability = candidate_probabilities[
                    source_index, retrieval_position
                ]
                candidate_rows.append(
                    {
                        "A": source,
                        "candidate_B": raw_targets[candidate_target_index],
                        "retrieval_rank": retrieval_position + 1,
                        "retrieval_similarity": float(
                            retrieval_scores[source_index, retrieval_position].item()
                        ),
                        "p_no_match": float(candidate_probability[no_match_id].item()),
                        "p_direct": float(candidate_probability[direct_id].item()),
                        "p_derivation": float(
                            candidate_probability[derivation_id].item()
                        ),
                        "match_probability": float(
                            candidate_probability[direct_id].item()
                            + candidate_probability[derivation_id].item()
                        ),
                    }
                )

    predictions_frame = pd.DataFrame(best_rows, columns=GROUP_OUTPUT_COLUMNS)
    candidates_frame = (
        pd.DataFrame(candidate_rows, columns=CANDIDATE_OUTPUT_COLUMNS)
        if candidate_rows is not None
        else None
    )
    return predictions_frame, candidates_frame


def match_groups(
    model: ModelBundle | str | Path,
    source_fields: Sequence[str],
    target_fields: Sequence[str],
    *,
    save_candidates: bool = False,
    **kwargs: Any,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience wrapper returning one frame, or two when candidates are asked for."""

    predictions, candidates = match_field_groups(
        model,
        source_fields,
        target_fields,
        include_candidates=save_candidates,
        **kwargs,
    )
    if save_candidates:
        assert candidates is not None
        return predictions, candidates
    return predictions


def _read_required_columns(path: str | Path, columns: Iterable[str]) -> pd.DataFrame:
    """Read a UTF-8 CSV and reject missing/blank inference fields."""

    csv_path = Path(path)
    try:
        frame = pd.read_csv(
            csv_path,
            encoding="utf-8-sig",
            dtype=str,
            keep_default_na=False,
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Input CSV does not exist: {csv_path}") from exc
    required = list(columns)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {missing!r}")
    if frame[required].isna().any().any() or any(
        frame[column].str.strip().eq("").any() for column in required
    ):
        raise ValueError(f"{csv_path} contains blank values in required columns")
    return frame


def predict_pairs_csv(
    model: ModelBundle | str | Path,
    input_path: str | Path,
    output_path: str | Path,
    *,
    batch_size: int | None = None,
    device: str | torch.device = "cpu",
) -> pd.DataFrame:
    """Predict an ``A,B`` CSV and write the specified output columns."""

    frame = _read_required_columns(input_path, ("A", "B"))
    predictions = predict_pairs(
        model,
        frame["A"].tolist(),
        frame["B"].tolist(),
        batch_size=batch_size,
        device=device,
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(destination, index=False, encoding="utf-8")
    return predictions


def match_groups_csv(
    model: ModelBundle | str | Path,
    source_path: str | Path,
    target_path: str | Path,
    output_path: str | Path,
    *,
    candidates_output_path: str | Path | None = None,
    **kwargs: Any,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Run group matching from ``A`` and ``B`` CSV files and persist results."""

    source_frame = _read_required_columns(source_path, ("A",))
    target_frame = _read_required_columns(target_path, ("B",))
    predictions, candidates = match_field_groups(
        model,
        source_frame["A"].tolist(),
        target_frame["B"].tolist(),
        include_candidates=candidates_output_path is not None,
        **kwargs,
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(destination, index=False, encoding="utf-8")
    if candidates_output_path is not None:
        assert candidates is not None
        candidate_destination = Path(candidates_output_path)
        candidate_destination.parent.mkdir(parents=True, exist_ok=True)
        candidates.to_csv(candidate_destination, index=False, encoding="utf-8")
    return predictions, candidates


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the pair-prediction CLI parser used by ``scripts/predict_pairs.py``."""

    parser = argparse.ArgumentParser(description="Predict metadata field pairs locally")
    parser.add_argument("--model-dir", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--a", help="Source field for single-pair prediction")
    mode.add_argument("--input", type=Path, help="Input UTF-8 CSV with A,B columns")
    parser.add_argument("--b", help="Target field for single-pair prediction")
    parser.add_argument("--output", type=Path, help="Output CSV for batch mode")
    parser.add_argument("--batch-size", type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run pair prediction from command-line arguments."""

    args = build_arg_parser().parse_args(argv)
    if args.a is not None:
        if args.b is None:
            raise SystemExit("--b is required when --a is used")
        result = predict_pair(args.model_dir, args.a, args.b)
        print(f"A: {result['A']}")
        print(f"B: {result['B']}")
        print()
        print(f"NO_MATCH: {result['p_no_match']:.6f}")
        print(f"DIRECT: {result['p_direct']:.6f}")
        print(f"DERIVATION: {result['p_derivation']:.6f}")
        print()
        print(f"Prediction: {result['predicted_label']}")
        print(f"Match Probability: {result['match_probability']:.6f}")
        return 0

    if args.b is not None:
        raise SystemExit("--b may only be used together with --a")
    if args.output is None:
        raise SystemExit("--output is required in CSV batch mode")
    predict_pairs_csv(
        args.model_dir,
        args.input,
        args.output,
        batch_size=args.batch_size,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI script.
    raise SystemExit(main())


__all__ = [
    "CANDIDATE_OUTPUT_COLUMNS",
    "GROUP_OUTPUT_COLUMNS",
    "ModelBundle",
    "PAIR_OUTPUT_COLUMNS",
    "encode_field_names",
    "load_model_bundle",
    "main",
    "match_field_groups",
    "match_groups",
    "match_groups_csv",
    "predict_pair",
    "predict_pairs",
    "predict_pairs_csv",
]
