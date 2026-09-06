"""Regression coverage for prepared split paths and the cleaning-to-model CLI."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from metadata_matcher.config import MatcherConfig
from metadata_matcher.evaluate import (
    _load_evaluation_csv,
    build_arg_parser,
    evaluate_model,
)
from metadata_matcher.train import DataValidationError, train_model, validate_data_files


ROOT = Path(__file__).resolve().parents[1]


def _write_splits(directory: Path) -> None:
    directory.mkdir(parents=True)
    for split in ("train", "validation", "test"):
        pd.DataFrame(
            [(f"{split}_source", f"{split}_target", "NO_MATCH")],
            columns=["A", "B", "label"],
        ).to_csv(directory / f"{split}.csv", index=False)


def test_default_directory_and_explicit_legacy_path_precedence(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_splits(tmp_path / "data_split")
    train, _, _, default_report = validate_data_files()
    assert train.iloc[0].A == "train_source"
    assert default_report["strict_leakage_check"] is True
    assert default_report["input_files"]["test"] == str(tmp_path / "data_split/test.csv")

    custom_dir = tmp_path / "prepared"
    _write_splits(custom_dir)
    override = tmp_path / "legacy_train.csv"
    pd.DataFrame(
        [("legacy_source", "legacy_target", "DIRECT")], columns=["A", "B", "label"]
    ).to_csv(override, index=False)
    train, _, _, report = validate_data_files(override, data_dir=custom_dir)
    assert train.iloc[0].A == "legacy_source"
    assert report["input_files"]["train"] == str(override)
    assert report["input_files"]["validation"] == str(custom_dir / "validation.csv")


def test_directory_preflight_rejects_normalized_cross_column_leakage(tmp_path):
    _write_splits(tmp_path / "prepared")
    pd.DataFrame(
        [("test_source", "Ｔｒａｉｎ Source", "NO_MATCH")],
        columns=["A", "B", "label"],
    ).to_csv(tmp_path / "prepared/test.csv", index=False)
    with pytest.raises(DataValidationError, match="Strict leakage check failed"):
        validate_data_files(data_dir=tmp_path / "prepared")


@pytest.mark.parametrize("empty_split", ["train", "validation", "test"])
def test_empty_prepared_split_fails_before_model_artifacts(tmp_path, empty_split):
    prepared = tmp_path / "prepared"
    _write_splits(prepared)
    (prepared / f"{empty_split}.csv").write_text("A,B,label\n", encoding="utf-8")
    artifacts = tmp_path / "artifacts"
    with pytest.raises(DataValidationError, match=f"{empty_split} CSV contains no rows"):
        train_model(
            data_dir=prepared,
            config=MatcherConfig(torch_num_threads=1),
            artifacts_dir=artifacts,
        )
    assert not artifacts.exists()


def test_empty_formal_evaluation_is_rejected(tmp_path):
    path = tmp_path / "test.csv"
    path.write_text("A,B,label\n", encoding="utf-8")
    with pytest.raises(ValueError, match="contains no rows"):
        _load_evaluation_csv(path)


def test_evaluation_directory_defaults_and_legacy_override(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_splits(tmp_path / "data_split")
    seen = []

    def fake_evaluate(model, dataframe, **kwargs):
        seen.append(dataframe.iloc[0].A)
        return {"data_summary": {"pair_rows": len(dataframe)}}

    monkeypatch.setattr("metadata_matcher.evaluate.evaluate_dataframe", fake_evaluate)
    default = evaluate_model("unused-model")
    assert seen[-1] == "test_source"
    assert default["data_summary"]["input_file"] == str(tmp_path / "data_split/test.csv")
    evaluate_model("unused-model", split="validation")
    assert seen[-1] == "validation_source"
    evaluate_model(
        "unused-model", tmp_path / "data_split/test.csv",
        data_dir=tmp_path / "missing", split="validation",
    )
    assert seen[-1] == "test_source"
    args = build_arg_parser().parse_args(["--model-dir", "unused-model"])
    assert args.data_dir == Path("data_split")
    assert args.split == "test"
    assert args.test is None


@pytest.mark.parametrize("human_no_match", [False, True])
def test_clean_train_and_evaluate_clis_use_prepared_data(tmp_path, human_no_match):
    """Use isolated annotated fixture data; never create formal project datasets."""

    source = tmp_path / "source.csv"
    labels = ["DIRECT", "DERIVATION", "NO_MATCH"] * 20 if human_no_match else (
        ["DIRECT", "DERIVATION"] * 30
    )
    rows = [
        (f"source_{index}", f"target_{index}", label)
        for index, label in enumerate(labels)
    ]
    pd.DataFrame(rows, columns=["A", "B", "label"]).to_csv(source, index=False)
    prepared = tmp_path / "prepared"
    config = MatcherConfig(
        seed=42, max_length=16, char_embedding_dim=2, cnn_channels=2,
        cnn_kernel_sizes=(2,), field_embedding_dim=4,
        classifier_hidden_dim=8, classifier_second_hidden_dim=4,
        embedding_epochs=1, classifier_epochs=1, batch_size=8,
        inference_batch_size=8, torch_num_threads=1,
    )
    config_path = tmp_path / "config.json"
    config.save(config_path)

    def run(*arguments):
        result = subprocess.run(
            [sys.executable, *map(str, arguments)], cwd=tmp_path,
            text=True, capture_output=True, timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        return result

    run(ROOT / "prepare_dataset.py", "--input", source, "--output-dir", prepared,
        "--seed", "42")
    run(ROOT / "scripts/validate_data.py", "--data-dir", prepared)
    run(ROOT / "scripts/train_model.py", "--config", config_path,
        "--data-dir", prepared, "--artifacts-dir", tmp_path / "artifacts")
    runs = list((tmp_path / "artifacts").glob("run_*"))
    assert len(runs) == 1
    data_report = json.loads((runs[0] / "data_validation.json").read_text("utf-8"))
    history = json.loads((runs[0] / "training_history.json").read_text("utf-8"))
    assert (history["synthetic_training_negatives"] == 0) is human_no_match
    for split in ("train", "validation", "test"):
        frame = pd.read_csv(prepared / f"{split}.csv")
        assert data_report["rows"][split] == len(frame)
        assert data_report["input_files"][split] == str(prepared / f"{split}.csv")
        assert list(frame.columns) == ["A", "B", "label"]
        assert set(frame.itertuples(index=False, name=None)).issubset(set(rows))
    assert all(
        details["normalized_any_field_overlap_count"] == 0
        for details in data_report["leakage"].values()
    )
    for split in ("validation", "test"):
        output = tmp_path / f"{split}_evaluation.json"
        run(ROOT / "scripts/evaluate_model.py", "--model-dir", runs[0],
            "--data-dir", prepared, "--split", split, "--output", output)
        metrics = json.loads(output.read_text("utf-8"))
        expected_rows = len(pd.read_csv(prepared / f"{split}.csv"))
        assert metrics["pair_classification"]["sample_count"] == expected_rows
        assert metrics["data_summary"]["pair_rows"] == expected_rows
        assert metrics["pair_classification"]["no_match_metrics_available"] is human_no_match
