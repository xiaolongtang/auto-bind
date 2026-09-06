"""Two-phase CPU training pipeline for the metadata field matcher."""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .config import MatcherConfig
from .dataset import LABEL_MAP, PairDataset, validate_normalized_pair_labels
from .evaluation_truth import group_truth_completeness
from .model import FieldEncoder, PairClassifier
from .negatives import generate_synthetic_negatives
from .preprocess import normalize_field_name
from .utils import (
    configure_logging,
    create_run_directory,
    save_state_dict,
    set_reproducible_seed,
    write_json,
)
from .vocab import CharVocabulary


EXPECTED_COLUMNS = ("A", "B", "label")
EXPECTED_LABEL_MAP = {"NO_MATCH": 0, "DIRECT": 1, "DERIVATION": 2}


class DataValidationError(ValueError):
    """Raised when a supplied CSV cannot safely be used by the pipeline."""


def read_labeled_csv(path: str | Path, split_name: str) -> pd.DataFrame:
    """Read and validate one pre-split UTF-8/BOM-compatible labeled CSV."""

    csv_path = Path(path)
    if not csv_path.is_file():
        raise DataValidationError(
            f"{split_name} CSV does not exist: {csv_path}. "
            "Run prepare_dataset.py --input data.csv --output-dir data_split "
            "first, or supply --data-dir / explicit split paths."
        )
    try:
        frame = pd.read_csv(
            csv_path,
            encoding="utf-8-sig",
            dtype=str,
            keep_default_na=False,
        )
    except (UnicodeDecodeError, pd.errors.ParserError) as exc:
        raise DataValidationError(f"Cannot read {split_name} CSV {csv_path}: {exc}") from exc

    missing = [column for column in EXPECTED_COLUMNS if column not in frame.columns]
    if missing:
        raise DataValidationError(
            f"{split_name} CSV is missing columns: {', '.join(missing)}"
        )
    columns = [*EXPECTED_COLUMNS]
    if "group_truth_complete" in frame:
        columns.append("group_truth_complete")
    frame = frame.loc[:, columns].copy()
    for column in ("A", "B"):
        frame[column] = frame[column].astype(str)
        invalid = frame[column].map(lambda value: not bool(value.strip()))
        if invalid.any():
            rows = (invalid[invalid].index + 2).tolist()[:10]
            raise DataValidationError(
                f"{split_name} has blank {column} values at CSV rows {rows}"
            )
        normalized_empty = frame[column].map(
            lambda value: not bool(normalize_field_name(value))
        )
        if normalized_empty.any():
            rows = (normalized_empty[normalized_empty].index + 2).tolist()[:10]
            raise DataValidationError(
                f"{split_name} has {column} values that normalize to empty text "
                f"at CSV rows {rows}"
            )
    frame["label"] = frame["label"].astype(str).str.strip().str.upper()
    invalid_labels = sorted(set(frame["label"]) - set(EXPECTED_LABEL_MAP))
    if invalid_labels:
        raise DataValidationError(
            f"{split_name} has unsupported labels {invalid_labels}; expected "
            f"{list(EXPECTED_LABEL_MAP)}"
        )
    if frame.empty:
        raise DataValidationError(
            f"{split_name} CSV contains no rows. Inspect split_report.md and "
            "provide more independent annotated field components; empty splits "
            "cannot be used for training or formal evaluation."
        )
    try:
        validate_normalized_pair_labels(frame, context=f"{split_name} CSV {csv_path}")
        group_truth_completeness(frame, context=f"{split_name} CSV {csv_path}")
    except ValueError as exc:
        raise DataValidationError(str(exc)) from exc
    return frame.reset_index(drop=True)


def _normalized_split_values(frame: pd.DataFrame) -> dict[str, set[Any]]:
    """Build normalized values used by leakage diagnostics."""

    normalized_a = {normalize_field_name(value) for value in frame["A"]}
    normalized_b = {normalize_field_name(value) for value in frame["B"]}
    normalized_pairs = {
        (normalize_field_name(a), normalize_field_name(b))
        for a, b in zip(frame["A"], frame["B"])
    }
    return {
        "a": normalized_a,
        "b": normalized_b,
        "all_fields": normalized_a | normalized_b,
        "pairs": normalized_pairs,
    }


def check_split_leakage(
    splits: Mapping[str, pd.DataFrame],
) -> dict[str, dict[str, Any]]:
    """Report exact normalized field/pair overlap between every split pair."""

    prepared = {
        name: _normalized_split_values(frame) for name, frame in splits.items()
    }
    names = list(splits)
    report: dict[str, dict[str, Any]] = {}
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            left = prepared[left_name]
            right = prepared[right_name]
            a_overlap = sorted(left["a"] & right["a"])
            b_overlap = sorted(left["b"] & right["b"])
            all_overlap = sorted(left["all_fields"] & right["all_fields"])
            pair_overlap = sorted(left["pairs"] & right["pairs"])
            key = f"{left_name}_vs_{right_name}"
            report[key] = {
                "normalized_a_overlap_count": len(a_overlap),
                "normalized_b_overlap_count": len(b_overlap),
                "normalized_any_field_overlap_count": len(all_overlap),
                "normalized_pair_overlap_count": len(pair_overlap),
                "sample_a_overlaps": a_overlap[:20],
                "sample_b_overlaps": b_overlap[:20],
                "sample_any_field_overlaps": all_overlap[:20],
                "sample_pair_overlaps": [list(pair) for pair in pair_overlap[:20]],
            }
    return report


def validate_data_files(
    train_path: str | Path | None = None,
    validation_path: str | Path | None = None,
    test_path: str | Path | None = None,
    *,
    data_dir: str | Path = "data_split",
    strict_leakage_check: bool = True,
    logger: logging.Logger | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load prepared splits, allowing explicit legacy paths to override the directory."""

    log = logger or logging.getLogger("metadata_matcher")
    split_paths = {
        name: Path(path) if path is not None else Path(data_dir) / f"{name}.csv"
        for name, path in (
            ("train", train_path),
            ("validation", validation_path),
            ("test", test_path),
        )
    }
    train = read_labeled_csv(split_paths["train"], "train")
    validation = read_labeled_csv(split_paths["validation"], "validation")
    test = read_labeled_csv(split_paths["test"], "test")
    splits = {"train": train, "validation": validation, "test": test}
    leakage = check_split_leakage(splits)

    leaking = {
        name: details
        for name, details in leakage.items()
        if details["normalized_any_field_overlap_count"] > 0
    }
    if leaking:
        for comparison, details in leaking.items():
            log.warning(
                "DATA LEAKAGE WARNING %s: %d normalized fields overlap "
                "(A-role=%d, B-role=%d, exact pairs=%d). Samples=%s",
                comparison,
                details["normalized_any_field_overlap_count"],
                details["normalized_a_overlap_count"],
                details["normalized_b_overlap_count"],
                details["normalized_pair_overlap_count"],
                details["sample_any_field_overlaps"][:5],
            )
        if strict_leakage_check:
            raise DataValidationError(
                "Strict leakage check failed: normalized field names occur in more "
                "than one split. Rebuild field-disjoint splits or explicitly pass "
                "--no-strict-leakage-check after reviewing the warning."
            )

    report: dict[str, Any] = {
        "input_files": {name: str(path.resolve()) for name, path in split_paths.items()},
        "rows": {name: len(frame) for name, frame in splits.items()},
        "label_counts": {
            name: {
                label: int((frame["label"] == label).sum())
                for label in EXPECTED_LABEL_MAP
            }
            for name, frame in splits.items()
        },
        "duplicate_row_counts": {
            name: int(frame.duplicated(subset=list(EXPECTED_COLUMNS)).sum())
            for name, frame in splits.items()
        },
        "strict_leakage_check": strict_leakage_check,
        "leakage": leakage,
    }
    for split_name in ("validation", "test"):
        if report["label_counts"][split_name]["NO_MATCH"] == 0:
            log.warning(
                "%s contains no NO_MATCH rows. NO_MATCH metrics are unavailable "
                "or incomplete; synthetic negatives are never added to evaluation.",
                split_name,
            )
    return train, validation, test, report


def _make_loader(
    frame: pd.DataFrame,
    vocab: CharVocabulary,
    config: MatcherConfig,
    *,
    shuffle: bool,
) -> DataLoader:
    dataset = PairDataset(
        frame,
        vocab,
        config.max_length,
        synthetic_negative_weight=config.synthetic_negative_weight,
    )
    generator = torch.Generator().manual_seed(config.seed)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        generator=generator if shuffle else None,
        pin_memory=False,
        persistent_workers=config.num_workers > 0,
    )


def _batch_tensors(batch: Mapping[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
    try:
        return batch["a_ids"], batch["b_ids"], batch["label"]
    except KeyError as exc:
        raise RuntimeError(
            "PairDataset batches must contain a_ids, b_ids, and label"
        ) from exc


def _classification_metrics(
    true_labels: Iterable[int], predicted_labels: Iterable[int]
) -> dict[str, Any]:
    """Compute deterministic three-class metrics without hidden averaging rules."""

    truth = list(int(value) for value in true_labels)
    predicted = list(int(value) for value in predicted_labels)
    if len(truth) != len(predicted):
        raise ValueError("true and predicted label lengths differ")
    confusion = [[0 for _ in EXPECTED_LABEL_MAP] for _ in EXPECTED_LABEL_MAP]
    for actual, guess in zip(truth, predicted):
        confusion[actual][guess] += 1

    per_class: dict[str, dict[str, float | int | bool]] = {}
    f1_values: list[float] = []
    for label_name, class_id in EXPECTED_LABEL_MAP.items():
        tp = confusion[class_id][class_id]
        fp = sum(confusion[row][class_id] for row in range(3)) - tp
        fn = sum(confusion[class_id]) - tp
        support = sum(confusion[class_id])
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        f1_values.append(f1)
        per_class[label_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "available": support > 0,
        }
    correct = sum(confusion[index][index] for index in range(3))
    result: dict[str, Any] = {
        "accuracy": correct / len(truth) if truth else 0.0,
        "macro_f1": sum(f1_values) / 3.0,
        "per_class": per_class,
        "confusion_matrix": confusion,
        "confusion_matrix_labels": list(EXPECTED_LABEL_MAP),
        "sample_count": len(truth),
    }
    if not per_class["NO_MATCH"]["available"]:
        result["warning"] = "NO_MATCH metrics are unavailable or incomplete."
    return result


def _run_encoder_epoch(
    encoder: FieldEncoder,
    loader: DataLoader,
    criterion: nn.CosineEmbeddingLoss,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, float]:
    training = optimizer is not None
    encoder.train(training)
    total_loss = 0.0
    total_examples = 0
    correct = 0
    for batch in loader:
        a_ids, b_ids, labels = _batch_tensors(batch)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            embedding_a = encoder(a_ids)
            embedding_b = encoder(b_ids)
            targets = torch.where(
                labels == EXPECTED_LABEL_MAP["NO_MATCH"],
                torch.full_like(labels, -1, dtype=torch.float32),
                torch.ones_like(labels, dtype=torch.float32),
            )
            per_sample_loss = criterion(embedding_a, embedding_b, targets)
            loss = (per_sample_loss * batch["sample_weight"]).mean()
            if training:
                loss.backward()
                optimizer.step()
        batch_size = int(labels.shape[0])
        total_loss += float(loss.detach()) * batch_size
        total_examples += batch_size
        similarities = F.cosine_similarity(embedding_a.detach(), embedding_b.detach())
        predicted_matches = similarities >= 0.0
        actual_matches = labels != EXPECTED_LABEL_MAP["NO_MATCH"]
        correct += int((predicted_matches == actual_matches).sum())
    return total_loss / max(total_examples, 1), correct / max(total_examples, 1)


def _run_classifier_epoch(
    encoder: FieldEncoder,
    classifier: PairClassifier,
    loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    optimizer: torch.optim.Optimizer | None,
    *,
    fine_tune_encoder: bool,
) -> tuple[float, dict[str, Any]]:
    training = optimizer is not None
    classifier.train(training)
    encoder.train(training and fine_tune_encoder)
    total_loss = 0.0
    total_examples = 0
    all_labels: list[int] = []
    all_predictions: list[int] = []

    for batch in loader:
        a_ids, b_ids, labels = _batch_tensors(batch)
        if training:
            optimizer.zero_grad(set_to_none=True)
        encoder_gradients = training and fine_tune_encoder
        with torch.set_grad_enabled(encoder_gradients):
            embedding_a = encoder(a_ids)
            embedding_b = encoder(b_ids)
        # Detaching is explicit in the frozen path and avoids retaining encoder
        # graphs while the classifier learns on CPU.
        if not encoder_gradients:
            embedding_a = embedding_a.detach()
            embedding_b = embedding_b.detach()
        with torch.set_grad_enabled(training):
            logits = classifier(embedding_a, embedding_b)
            per_sample_loss = criterion(logits, labels)
            loss = (per_sample_loss * batch["sample_weight"]).mean()
            if training:
                loss.backward()
                optimizer.step()
        predictions = logits.detach().argmax(dim=1)
        batch_size = int(labels.shape[0])
        total_loss += float(loss.detach()) * batch_size
        total_examples += batch_size
        all_labels.extend(labels.tolist())
        all_predictions.extend(predictions.tolist())

    metrics = _classification_metrics(all_labels, all_predictions)
    return total_loss / max(total_examples, 1), metrics


def _should_stop(epochs_without_improvement: int, patience: int) -> bool:
    return patience > 0 and epochs_without_improvement >= patience


def _build_encoder(vocab_size: int, config: MatcherConfig) -> FieldEncoder:
    return FieldEncoder(
        vocab_size=vocab_size,
        char_embedding_dim=config.char_embedding_dim,
        cnn_channels=config.cnn_channels,
        kernel_sizes=config.cnn_kernel_sizes,
        output_dim=config.field_embedding_dim,
        padding_idx=0,
    )


def _build_classifier(config: MatcherConfig) -> PairClassifier:
    return PairClassifier(
        embedding_dim=config.field_embedding_dim,
        hidden_dim=config.classifier_hidden_dim,
        second_hidden_dim=config.classifier_second_hidden_dim,
        dropout=config.dropout,
    )


def train_model(
    *,
    train_path: str | Path | None = None,
    validation_path: str | Path | None = None,
    test_path: str | Path | None = None,
    config: MatcherConfig,
    data_dir: str | Path = "data_split",
    artifacts_dir: str | Path = "artifacts",
    strict_leakage_check: bool = True,
) -> Path:
    """Train from prepared splits and return the newly created model directory.

    By default, read train.csv, validation.csv, and test.csv in ``data_split``.
    Explicit paths override the corresponding file inside ``data_dir``.
    """

    if dict(LABEL_MAP) != EXPECTED_LABEL_MAP:
        raise RuntimeError(
            f"Internal label mapping must be {EXPECTED_LABEL_MAP}, got {LABEL_MAP}"
        )
    set_reproducible_seed(config.seed, config.torch_num_threads)
    # Complete all fallible data preflight work before creating a run directory.
    # A failed strict-leakage check must not leave a partial ``run_*`` that can be
    # mistaken for the latest usable model.
    preflight_logger = configure_logging()
    train_frame, validation_frame, test_frame, data_report = validate_data_files(
        train_path,
        validation_path,
        test_path,
        data_dir=data_dir,
        strict_leakage_check=strict_leakage_check,
        logger=preflight_logger,
    )

    # Build strictly from original training text, before looking at validation or
    # test characters. Synthetic negatives only reuse training fields.
    vocab = CharVocabulary.build(
        list(train_frame["A"].astype(str)) + list(train_frame["B"].astype(str))
    )

    # Explicit provenance distinguishes verified annotations from generated
    # negatives, including after concatenation and DataLoader collation.
    train_frame = train_frame.assign(synthetic=False, negative_type="human")
    training_frame = train_frame
    generated_count = 0
    if not (train_frame["label"] == "NO_MATCH").any():
        generated_negatives = generate_synthetic_negatives(
            train_frame,
            random_negatives_per_positive=config.random_negatives_per_positive,
            hard_negatives_per_positive=config.hard_negatives_per_positive,
            hard_negative_pool_size=config.hard_negative_pool_size,
            seed=config.seed,
        )
        training_frame = pd.concat(
            [train_frame, generated_negatives], ignore_index=True
        )
        generated_count = len(generated_negatives)
        if generated_count == 0:
            raise DataValidationError(
                "train contains no NO_MATCH rows and no valid synthetic negative "
                "could be generated. Provide human-verified NO_MATCH rows or at "
                "least two distinct eligible target fields."
            )

    run_dir = create_run_directory(artifacts_dir)
    logger = configure_logging(run_dir / "training.log")
    config.save(run_dir / "config.json")
    write_json(run_dir / "label_map.json", EXPECTED_LABEL_MAP)
    write_json(run_dir / "data_validation.json", data_report)
    vocab.save(run_dir / "vocab.json")
    logger.info("Starting local CPU-only training; run directory: %s", run_dir)
    logger.info(
        "Loaded fixed splits: train=%d validation=%d test=%d (no random split)",
        len(train_frame),
        len(validation_frame),
        len(test_frame),
    )
    logger.info("Built train-only character vocabulary with %d tokens", len(vocab))
    for comparison, details in data_report["leakage"].items():
        if details["normalized_any_field_overlap_count"]:
            logger.warning(
                "DATA LEAKAGE WARNING %s: %d normalized fields overlap; "
                "training continued because strict checking was disabled.",
                comparison,
                details["normalized_any_field_overlap_count"],
            )
    for split_name in ("validation", "test"):
        if data_report["label_counts"][split_name]["NO_MATCH"] == 0:
            logger.warning(
                "%s contains no NO_MATCH rows. NO_MATCH metrics are unavailable "
                "or incomplete.",
                split_name,
            )
    if generated_count:
        logger.warning(
            "Train has no NO_MATCH rows; generated %d training-only negatives. "
            "Synthetic negatives should only be used for training and are not "
            "equivalent to human-verified NO_MATCH evaluation data.",
            generated_count,
        )

    train_loader = _make_loader(training_frame, vocab, config, shuffle=True)
    validation_loader = _make_loader(validation_frame, vocab, config, shuffle=False)

    encoder = _build_encoder(len(vocab), config)
    cosine_criterion = nn.CosineEmbeddingLoss(
        margin=config.cosine_margin, reduction="none"
    )
    encoder_optimizer = torch.optim.AdamW(
        encoder.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    history: dict[str, Any] = {
        "phase_1_encoder": [],
        "phase_2_classifier": [],
        "synthetic_training_negatives": len(training_frame) - len(train_frame),
        "synthetic_negative_weight": config.synthetic_negative_weight,
        "synthetic_negative_counts_by_type": (
            training_frame.loc[training_frame["synthetic"], "negative_type"]
            .value_counts().astype(int).to_dict()
        ),
    }

    best_encoder_state: dict[str, Tensor] | None = None
    best_loss = float("inf")
    stale_epochs = 0
    for epoch in range(1, config.embedding_epochs + 1):
        train_loss, train_binary_accuracy = _run_encoder_epoch(
            encoder, train_loader, cosine_criterion, encoder_optimizer
        )
        with torch.inference_mode():
            validation_loss, validation_binary_accuracy = _run_encoder_epoch(
                encoder, validation_loader, cosine_criterion, None
            )
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "train_match_accuracy_at_cosine_0": train_binary_accuracy,
            "validation_match_accuracy_at_cosine_0": validation_binary_accuracy,
            "learning_rate": encoder_optimizer.param_groups[0]["lr"],
        }
        history["phase_1_encoder"].append(record)
        logger.info(
            "Phase 1 | Epoch %d/%d | Train Loss %.6f | Validation Loss %.6f | "
            "Validation Match Accuracy %.4f | LR %.6g",
            epoch,
            config.embedding_epochs,
            train_loss,
            validation_loss,
            validation_binary_accuracy,
            encoder_optimizer.param_groups[0]["lr"],
        )
        if validation_loss < best_loss - config.early_stopping_min_delta:
            best_loss = validation_loss
            stale_epochs = 0
            best_encoder_state = copy.deepcopy(encoder.state_dict())
            save_state_dict(encoder, run_dir / "encoder.pt")
        else:
            stale_epochs += 1
            if _should_stop(stale_epochs, config.early_stopping_patience):
                logger.info("Phase 1 early stopping after %d stale epochs", stale_epochs)
                break
    if best_encoder_state is None:
        raise RuntimeError("Encoder training did not produce a checkpoint")
    encoder.load_state_dict(best_encoder_state)

    classifier = _build_classifier(config)
    for parameter in encoder.parameters():
        parameter.requires_grad = config.fine_tune_encoder
    classifier_parameters: list[nn.Parameter] = list(classifier.parameters())
    if config.fine_tune_encoder:
        classifier_parameters.extend(encoder.parameters())
    classifier_optimizer = torch.optim.AdamW(
        classifier_parameters,
        lr=config.classifier_learning_rate,
        weight_decay=config.weight_decay,
    )
    classifier_criterion = nn.CrossEntropyLoss(reduction="none")
    best_classifier_state: dict[str, Tensor] | None = None
    best_joint_encoder_state: dict[str, Tensor] | None = None
    best_loss = float("inf")
    stale_epochs = 0
    for epoch in range(1, config.classifier_epochs + 1):
        train_loss, train_metrics = _run_classifier_epoch(
            encoder,
            classifier,
            train_loader,
            classifier_criterion,
            classifier_optimizer,
            fine_tune_encoder=config.fine_tune_encoder,
        )
        with torch.inference_mode():
            validation_loss, validation_metrics = _run_classifier_epoch(
                encoder,
                classifier,
                validation_loader,
                classifier_criterion,
                None,
                fine_tune_encoder=False,
            )
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "validation_accuracy": validation_metrics["accuracy"],
            "validation_macro_f1": validation_metrics["macro_f1"],
            "learning_rate": classifier_optimizer.param_groups[0]["lr"],
        }
        history["phase_2_classifier"].append(record)
        logger.info(
            "Phase 2 | Epoch %d/%d | Train Loss %.6f | Validation Loss %.6f | "
            "Validation Accuracy %.4f | Macro F1 %.4f | LR %.6g",
            epoch,
            config.classifier_epochs,
            train_loss,
            validation_loss,
            validation_metrics["accuracy"],
            validation_metrics["macro_f1"],
            classifier_optimizer.param_groups[0]["lr"],
        )
        if validation_loss < best_loss - config.early_stopping_min_delta:
            best_loss = validation_loss
            stale_epochs = 0
            best_classifier_state = copy.deepcopy(classifier.state_dict())
            if config.fine_tune_encoder:
                best_joint_encoder_state = copy.deepcopy(encoder.state_dict())
            save_state_dict(classifier, run_dir / "classifier.pt")
            if config.fine_tune_encoder:
                save_state_dict(encoder, run_dir / "encoder.pt")
        else:
            stale_epochs += 1
            if _should_stop(stale_epochs, config.early_stopping_patience):
                logger.info("Phase 2 early stopping after %d stale epochs", stale_epochs)
                break
    if best_classifier_state is None:
        raise RuntimeError("Classifier training did not produce a checkpoint")
    classifier.load_state_dict(best_classifier_state)
    if best_joint_encoder_state is not None:
        encoder.load_state_dict(best_joint_encoder_state)
    else:
        encoder.load_state_dict(best_encoder_state)
    for parameter in encoder.parameters():
        parameter.requires_grad = True

    write_json(run_dir / "training_history.json", history)
    # Reload the just-written portable artifact bundle before final evaluation.
    # Besides producing the full pair/retrieval/end-to-end report, this verifies
    # that a trained model is usable independently of in-memory Python objects.
    from .evaluate import evaluate_dataframe
    from .predict import load_model_bundle

    saved_bundle = load_model_bundle(run_dir)
    thresholds = (0.80, 0.90, 0.95, 0.99)
    validation_metrics = evaluate_dataframe(
        saved_bundle,
        validation_frame,
        top_k=config.top_k,
        thresholds=thresholds,
        batch_size=config.inference_batch_size,
    )
    test_metrics = evaluate_dataframe(
        saved_bundle,
        test_frame,
        top_k=config.top_k,
        thresholds=thresholds,
        batch_size=config.inference_batch_size,
    )
    write_json(run_dir / "validation_metrics.json", validation_metrics)
    write_json(run_dir / "test_metrics.json", test_metrics)
    logger.info(
        "Training complete | validation accuracy %.4f | test accuracy %.4f",
        validation_metrics["pair_classification"]["accuracy"],
        test_metrics["pair_classification"]["accuracy"],
    )
    return run_dir


__all__ = [
    "DataValidationError",
    "check_split_leakage",
    "read_labeled_csv",
    "train_model",
    "validate_data_files",
]
