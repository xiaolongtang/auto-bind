"""Small integration coverage for train/save/load/predict."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from metadata_matcher.config import MatcherConfig
from metadata_matcher.predict import load_model_bundle, predict_pair
from metadata_matcher.train import DataValidationError, train_model


def _write_split(path: Path, rows: list[tuple[str, str, str]]) -> None:
    pd.DataFrame(rows, columns=["A", "B", "label"]).to_csv(
        path, index=False, encoding="utf-8"
    )


def test_tiny_pipeline_train_save_load_predict(tmp_path: Path) -> None:
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
    )
    _write_split(
        test_path,
        [
            ("invoice_amt", "InvoiceAmount", "DIRECT"),
            ("ship_dt", "arrival_dt", "DERIVATION"),
            ("invoice_amt", "person_id", "NO_MATCH"),
        ],
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
    history = json.loads((run_dir / "training_history.json").read_text("utf-8"))
    assert history["synthetic_training_negatives"] > 0
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
