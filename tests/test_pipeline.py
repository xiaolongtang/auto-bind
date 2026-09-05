"""Small integration coverage for train/save/load/predict."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from metadata_matcher.config import MatcherConfig
from metadata_matcher.dataset import PairDataset, build_training_vocabulary
from metadata_matcher.model import FieldEncoder, PairClassifier
from metadata_matcher.predict import load_model_bundle, predict_pair
from metadata_matcher.train import (
    DataValidationError,
    _run_classifier_epoch,
    _run_encoder_epoch,
    read_labeled_csv,
    train_model,
)


def _write_split(
    path: Path, rows: list[tuple[str, str, str]], *, complete: bool = False
) -> None:
    frame = pd.DataFrame(rows, columns=["A", "B", "label"])
    if complete:
        frame["group_truth_complete"] = True
    frame.to_csv(path, index=False, encoding="utf-8")


@pytest.mark.parametrize("complete", [False, True])
@pytest.mark.parametrize("fine_tune", [False, True])
def test_tiny_pipeline_train_save_load_predict(
    tmp_path: Path, complete: bool, fine_tune: bool
) -> None:
    train_path = tmp_path / "train.csv"
    validation_path = tmp_path / "validation.csv"
    test_path = tmp_path / "test.csv"
    _write_split(
        train_path,
        [
            ("sale_price", "SalePrice", "DIRECT"),
            ("customer_no", "client_id", "DIRECT"),
            ("trade_date", "settlement_date", "DERIVATION"),
            ("invoice_total", "tax_due", "DERIVATION"),
        ],
    )
    _write_split(
        validation_path,
        [
            ("product_cost", "ProductCost", "DIRECT"),
            ("order_dt", "delivery_dt", "DERIVATION"),
            ("product_cost", "staff_code", "NO_MATCH"),
        ],
        complete=complete,
    )
    _write_split(
        test_path,
        [
            ("invoice_amt", "InvoiceAmount", "DIRECT"),
            ("ship_dt", "arrival_dt", "DERIVATION"),
            ("invoice_amt", "person_id", "NO_MATCH"),
        ],
        complete=complete,
    )
    config = MatcherConfig(
        seed=7,
        max_length=16,
        char_embedding_dim=4,
        cnn_channels=4,
        cnn_kernel_sizes=(2, 3),
        field_embedding_dim=8,
        classifier_hidden_dim=16,
        classifier_second_hidden_dim=8,
        embedding_epochs=1,
        classifier_epochs=1,
        batch_size=2,
        inference_batch_size=2,
        early_stopping_patience=1,
        top_k=2,
        retrieval_chunk_size=2,
        torch_num_threads=1,
        synthetic_negative_weight=0.25,
        fine_tune_encoder=fine_tune,
    )

    run_dir = train_model(
        train_path=train_path,
        validation_path=validation_path,
        test_path=test_path,
        config=config,
        artifacts_dir=tmp_path / "artifacts",
        strict_leakage_check=True,
    )

    required = {
        "encoder.pt",
        "classifier.pt",
        "vocab.json",
        "config.json",
        "label_map.json",
        "training_history.json",
        "validation_metrics.json",
        "test_metrics.json",
        "training.log",
    }
    assert required.issubset({path.name for path in run_dir.iterdir()})
    for split_name in ("validation", "test"):
        report = json.loads((run_dir / f"{split_name}_metrics.json").read_text("utf-8"))
        assert "MRR@20" in report["retrieval"]
        assert "MRR" not in report["retrieval"]
        assert report["end_to_end"]["total_sources"] == 2
        assert report["end_to_end"]["evaluated_sources"] == (2 if complete else 0)
        assert report["end_to_end"]["excluded_incomplete_sources"] == (0 if complete else 2)
        assert (report["end_to_end"]["overall_end_to_end_accuracy"] is None) is not complete
    history = json.loads((run_dir / "training_history.json").read_text("utf-8"))
    assert history["synthetic_training_negatives"] > 0
    assert history["synthetic_negative_weight"] == 0.25
    assert sum(history["synthetic_negative_counts_by_type"].values()) == (
        history["synthetic_training_negatives"]
    )
    assert MatcherConfig.load(run_dir / "config.json").synthetic_negative_weight == 0.25
    assert "Synthetic negatives should only be used for training" in (
        run_dir / "training.log"
    ).read_text("utf-8")

    first_bundle = load_model_bundle(run_dir)
    first = predict_pair(first_bundle, "sale_price", "SalePrice")
    second = predict_pair(load_model_bundle(run_dir), "sale_price", "SalePrice")
    assert first["predicted_label"] == second["predicted_label"]
    assert first["p_no_match"] == pytest.approx(second["p_no_match"], abs=1e-8)
    assert first["p_direct"] == pytest.approx(second["p_direct"], abs=1e-8)
    assert first["p_derivation"] == pytest.approx(
        second["p_derivation"], abs=1e-8
    )
    assert (
        first["p_no_match"] + first["p_direct"] + first["p_derivation"]
    ) == pytest.approx(1.0, abs=1e-6)


def test_config_rejects_negative_early_stopping_delta() -> None:
    with pytest.raises(ValueError, match="early_stopping_min_delta"):
        MatcherConfig(early_stopping_min_delta=-0.1)
    with pytest.raises(TypeError, match="fine_tune_encoder"):
        MatcherConfig.from_dict({"fine_tune_encoder": "false"})


@pytest.mark.parametrize("weight", [0.0, -0.1, 1.1, float("nan"), float("inf")])
def test_config_rejects_invalid_synthetic_negative_weight(weight: float) -> None:
    with pytest.raises(ValueError, match="synthetic_negative_weight"):
        MatcherConfig(synthetic_negative_weight=weight)


@pytest.mark.parametrize("weight", [True, "0.5"])
def test_config_rejects_non_numeric_synthetic_negative_weight(weight) -> None:
    with pytest.raises(TypeError, match="synthetic_negative_weight"):
        MatcherConfig.from_dict({"synthetic_negative_weight": weight})


def test_old_configs_keep_full_synthetic_weight() -> None:
    assert MatcherConfig.from_dict({}).synthetic_negative_weight == 1.0


def test_both_training_losses_weight_only_synthetic_rows() -> None:
    torch.manual_seed(7)
    torch.set_num_threads(1)
    dataframe = pd.DataFrame(
        {
            "A": ["first", "second", "third"],
            "B": ["one", "two", "three"],
            "label": ["DIRECT", "NO_MATCH", "NO_MATCH"],
            "synthetic": [False, True, False],
        }
    )
    vocab = build_training_vocabulary(dataframe)
    dataset = PairDataset(dataframe, vocab, max_length=8, synthetic_negative_weight=0.25)
    loader = DataLoader(dataset, batch_size=3)
    encoder = FieldEncoder(
        len(vocab), char_embedding_dim=2, cnn_channels=2, kernel_sizes=(2,), output_dim=4
    )
    classifier = PairClassifier(embedding_dim=4, hidden_dim=4, second_hidden_dim=2, dropout=0)
    batch = next(iter(loader))
    encoder.eval()
    classifier.eval()
    cosine_criterion = nn.CosineEmbeddingLoss(margin=0.2, reduction="none")
    classifier_criterion = nn.CrossEntropyLoss(reduction="none")
    with torch.no_grad():
        a_embeddings = encoder(batch["a_ids"])
        b_embeddings = encoder(batch["b_ids"])
        individual_cosine = cosine_criterion(
            a_embeddings, b_embeddings, torch.tensor([1.0, -1.0, -1.0])
        )
        individual_ce = classifier_criterion(
            classifier(a_embeddings, b_embeddings), batch["label"]
        )
    # Independent scalar expectations catch reducing a loss before applying
    # weights, or accidentally downweighting the human-verified NO_MATCH row.
    expected_cosine = (
        individual_cosine[0] + 0.25 * individual_cosine[1] + individual_cosine[2]
    ) / 3
    expected_ce = (individual_ce[0] + 0.25 * individual_ce[1] + individual_ce[2]) / 3
    encoder_loss, _ = _run_encoder_epoch(encoder, loader, cosine_criterion, None)
    classifier_loss, _ = _run_classifier_epoch(
        encoder, classifier, loader, classifier_criterion, None, fine_tune_encoder=False
    )
    assert encoder_loss == pytest.approx(expected_cosine.item())
    assert classifier_loss == pytest.approx(expected_ce.item())


@pytest.mark.parametrize("conflicting_split", ["train", "validation", "test"])
def test_conflicting_split_labels_fail_before_run_creation(
    tmp_path: Path, conflicting_split: str
) -> None:
    paths = {name: tmp_path / f"{name}.csv" for name in ("train", "validation", "test")}
    for name, path in paths.items():
        rows = [(f"{name}_source", f"{name}_target", "NO_MATCH")]
        if name == conflicting_split:
            rows = [
                ("sale_price", "SalePrice", "DIRECT"),
                ("SalePrice", "sale-price", "NO_MATCH"),
            ]
        _write_split(path, rows)
    artifacts_dir = tmp_path / "artifacts"
    with pytest.raises(DataValidationError, match="conflicting labels") as exc:
        train_model(
            train_path=paths["train"],
            validation_path=paths["validation"],
            test_path=paths["test"],
            config=MatcherConfig(torch_num_threads=1),
            artifacts_dir=artifacts_dir,
        )
    assert f"{conflicting_split} CSV" in str(exc.value)
    assert "CSV row 2" in str(exc.value)
    assert "CSV row 3" in str(exc.value)
    assert "DIRECT" in str(exc.value)
    assert "NO_MATCH" in str(exc.value)
    assert not artifacts_dir.exists()


def test_csv_preflight_preserves_complete_group_annotations(tmp_path: Path) -> None:
    path = tmp_path / "validation.csv"
    pd.DataFrame(
        {
            "A": ["first", "second"],
            "B": ["one", "two"],
            "label": ["DIRECT", "NO_MATCH"],
            "group_truth_complete": [True, False],
        }
    ).to_csv(path, index=False)
    frame = read_labeled_csv(path, "validation")
    assert frame["group_truth_complete"].str.lower().tolist() == ["true", "false"]


def test_csv_preflight_rejects_conflicting_group_completeness(tmp_path: Path) -> None:
    path = tmp_path / "validation.csv"
    pd.DataFrame(
        {
            "A": ["first_field", "FirstField"],
            "B": ["one", "two"],
            "label": ["DIRECT", "NO_MATCH"],
            "group_truth_complete": [True, False],
        }
    ).to_csv(path, index=False)
    with pytest.raises(DataValidationError, match="group_truth_complete"):
        read_labeled_csv(path, "validation")


def test_failed_preflight_does_not_leave_partial_run(tmp_path: Path) -> None:
    train_path = tmp_path / "train.csv"
    validation_path = tmp_path / "validation.csv"
    test_path = tmp_path / "test.csv"
    _write_split(train_path, [("same_field", "TrainTarget", "DIRECT")])
    _write_split(validation_path, [("same_field", "ValidTarget", "DIRECT")])
    _write_split(test_path, [("test_field", "TestTarget", "DIRECT")])
    artifacts_dir = tmp_path / "artifacts"

    with pytest.raises(DataValidationError, match="Strict leakage check failed"):
        train_model(
            train_path=train_path,
            validation_path=validation_path,
            test_path=test_path,
            config=MatcherConfig(
                max_length=8,
                cnn_kernel_sizes=(2, 3),
                embedding_epochs=1,
                classifier_epochs=1,
                torch_num_threads=1,
            ),
            artifacts_dir=artifacts_dir,
            strict_leakage_check=True,
        )
    assert not artifacts_dir.exists()


def test_impossible_synthetic_negatives_fail_before_run_creation(
    tmp_path: Path,
) -> None:
    train_path = tmp_path / "train.csv"
    validation_path = tmp_path / "validation.csv"
    test_path = tmp_path / "test.csv"
    _write_split(train_path, [("only_source", "OnlyTarget", "DIRECT")])
    _write_split(validation_path, [("valid_source", "ValidTarget", "NO_MATCH")])
    _write_split(test_path, [("test_source", "TestTarget", "NO_MATCH")])
    artifacts_dir = tmp_path / "artifacts"

    with pytest.raises(DataValidationError, match="no valid synthetic negative"):
        train_model(
            train_path=train_path,
            validation_path=validation_path,
            test_path=test_path,
            config=MatcherConfig(
                max_length=8,
                cnn_kernel_sizes=(2, 3),
                embedding_epochs=1,
                classifier_epochs=1,
                torch_num_threads=1,
            ),
            artifacts_dir=artifacts_dir,
            strict_leakage_check=True,
        )
    assert not artifacts_dir.exists()
