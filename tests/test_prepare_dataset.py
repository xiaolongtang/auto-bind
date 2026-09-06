"""Contract tests for preparing real, field-disjoint evaluation datasets."""

from __future__ import annotations

from collections import Counter
import csv
from difflib import SequenceMatcher
import json
from pathlib import Path
import subprocess
import sys

import pytest

from metadata_matcher.prepare_dataset import prepare_dataset
from metadata_matcher.preprocess import normalize_field_name


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "validation", "test")
OUTPUT_FILES = {
    "train.csv",
    "validation.csv",
    "test.csv",
    "test_hard_cases.csv",
    "conflicting_pairs.csv",
    "rejected_rows.csv",
    "split_report.md",
    "split_statistics.json",
}
LABELS = ("DIRECT", "DERIVATION", "NO_MATCH")
Row = tuple[str, str, str]


def _write_csv(path: Path, rows: list[Row]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["A", "B", "label"])
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _triples(path: Path) -> list[Row]:
    return [(row["A"], row["B"], row["label"]) for row in _read_csv(path)]


def _component_rows(count: int = 20) -> list[Row]:
    """Equal-size components with every label, normalization, and B-to-A links."""
    return [
        row
        for index in range(count)
        for row in [
            (f"  Group{index:02d}Source  ", f"group{index:02d}_target", "DIRECT"),
            (f"group{index:02d}.target", f"group{index:02d}_aux", "DERIVATION"),
            (f"group{index:02d}_aux", f"group{index:02d}_tail", "NO_MATCH"),
        ]
    ]


def _run_cli(input_path: Path, output_dir: Path, *options: str) -> subprocess.CompletedProcess[str]:
    # -S disables third-party site-packages: preparation must not import the
    # model, torch, pandas, or any other optional training dependencies.
    return subprocess.run(
        [
            sys.executable,
            "-S",
            str(PROJECT_ROOT / "prepare_dataset.py"),
            "--input",
            str(input_path),
            "--output-dir",
            str(output_dir),
            *options,
        ],
        cwd=input_path.parent,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )


def _assert_complete_outputs(output_dir: Path) -> None:
    assert OUTPUT_FILES <= {path.name for path in output_dir.iterdir()}
    statistics = json.loads((output_dir / "split_statistics.json").read_text("utf-8"))
    assert isinstance(statistics, dict) and statistics
    assert (output_dir / "split_report.md").read_text("utf-8").startswith(
        "# Dataset Split Report"
    )
    for split in SPLITS:
        with (output_dir / f"{split}.csv").open(encoding="utf-8-sig", newline="") as handle:
            assert next(csv.reader(handle)) == ["A", "B", "label"]


def test_components_remain_whole_with_no_field_overlap_or_invented_rows(tmp_path: Path) -> None:
    source_rows = _component_rows()
    input_path = tmp_path / "data.csv"
    output_dir = tmp_path / "split"
    _write_csv(input_path, source_rows)

    statistics = prepare_dataset(input_path, output_dir, seed=42)

    assert isinstance(statistics, dict)
    _assert_complete_outputs(output_dir)
    actual = {split: _triples(output_dir / f"{split}.csv") for split in SPLITS}
    assert all(actual.values())
    assert Counter(row for rows in actual.values() for row in rows) == Counter(source_rows)
    fields = {
        split: {normalize_field_name(field) for row in rows for field in row[:2]}
        for split, rows in actual.items()
    }
    assert fields["train"].isdisjoint(fields["validation"])
    assert fields["train"].isdisjoint(fields["test"])
    assert fields["validation"].isdisjoint(fields["test"])
    destinations = {row: split for split, rows in actual.items() for row in rows}
    for start in range(0, len(source_rows), 3):
        assert len({destinations[row] for row in source_rows[start : start + 3]}) == 1
    for split, target in zip(SPLITS, (0.70, 0.15, 0.15)):
        assert abs(len(actual[split]) / len(source_rows) - target) <= 0.05 + 1e-9
        distribution = Counter(row[2] for row in actual[split])
        assert len(set(distribution.values())) == 1
        assert set(distribution) == set(LABELS)


def test_duplicate_conflict_and_invalid_rows_are_audited_without_changing_field_text(
    tmp_path: Path,
) -> None:
    clean_rows = _component_rows()
    normalized_label_row = ("  OriginalField  ", "Original.Target", " direct ")
    conflicting_rows = [
        ("conflict_source", "conflict_target", "DIRECT"),
        ("conflict_source", "conflict_target", "DIRECT"),
        ("conflict_source", "conflict_target", "DERIVATION"),
        ("NormalSource", "NormalTarget", "DIRECT"),
        ("normal.source", "normal-target", "NO_MATCH"),
    ]
    invalid_rows = [
        ("", "missing_a_target", "DIRECT"),
        ("missing_b_source", "   ", "DIRECT"),
        ("empty_label_source", "empty_label_target", ""),
        ("bad_label_source", "bad_label_target", "OTHER"),
        ("___ . -", "separator_target", "DIRECT"),
    ]
    input_path = tmp_path / "data.csv"
    output_dir = tmp_path / "split"
    _write_csv(
        input_path,
        clean_rows + [clean_rows[0], normalized_label_row] + conflicting_rows + invalid_rows,
    )

    prepare_dataset(input_path, output_dir)

    emitted = [row for split in SPLITS for row in _triples(output_dir / f"{split}.csv")]
    expected = clean_rows + [(normalized_label_row[0], normalized_label_row[1], "DIRECT")]
    assert Counter(emitted) == Counter(expected)
    assert Counter(_triples(output_dir / "conflicting_pairs.csv")) == Counter(conflicting_rows)
    assert len(_read_csv(output_dir / "rejected_rows.csv")) == len(invalid_rows)
    _assert_complete_outputs(output_dir)


def test_bom_and_extra_columns_are_accepted_but_split_schema_stays_three_columns(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "data.csv"
    source_rows = _component_rows()
    with input_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["A", "B", "label", "reviewer_note"])
        writer.writerows([*row, "human checked"] for row in source_rows)

    output_dir = tmp_path / "split"
    prepare_dataset(input_path, output_dir)

    _assert_complete_outputs(output_dir)
    assert Counter(
        row for split in SPLITS for row in _triples(output_dir / f"{split}.csv")
    ) == Counter(source_rows)


def test_seed_repeats_identical_splits_and_audit_files(tmp_path: Path) -> None:
    input_path = tmp_path / "data.csv"
    _write_csv(input_path, _component_rows())
    first, second = tmp_path / "first", tmp_path / "second"

    prepare_dataset(input_path, first, seed=137)
    prepare_dataset(input_path, second, seed=137)

    for name in OUTPUT_FILES:
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_explicit_synthetic_records_cannot_lose_provenance_in_output(tmp_path: Path) -> None:
    input_path = tmp_path / "data.csv"
    source_rows = _component_rows()
    with input_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["A", "B", "label", "synthetic"])
        writer.writerows([*row, "false"] for row in source_rows)
        writer.writerow(["fabricated_source", "fabricated_target", "NO_MATCH", "TRUE"])
        writer.writerow(["unknown_origin", "unknown_target", "DIRECT", ""])

    output_dir = tmp_path / "split"
    statistics = prepare_dataset(input_path, output_dir)

    assert Counter(
        row for split in SPLITS for row in _triples(output_dir / f"{split}.csv")
    ) == Counter(source_rows)
    rejected = _read_csv(output_dir / "rejected_rows.csv")
    assert {row["reason"] for row in rejected} == {"synthetic_record", "invalid_synthetic_marker"}
    assert statistics["data_quality"]["rejected_rows"] == 2


def test_leakage_validation_blocks_publishing_bad_splits(tmp_path: Path, monkeypatch) -> None:
    input_path = tmp_path / "data.csv"
    _write_csv(input_path, _component_rows())
    output_dir = tmp_path / "split"
    monkeypatch.setattr(
        "metadata_matcher.prepare_dataset.check_leakage",
        lambda splits: {"train_test": {"shared_field_count": 1, "shared_fields": ["price"]}},
    )

    with pytest.raises(RuntimeError, match="FIELD LEAKAGE ERROR"):
        prepare_dataset(input_path, output_dir)

    assert not output_dir.exists()


def test_no_match_is_never_synthesized_and_report_explains_missing_evaluation(
    tmp_path: Path,
) -> None:
    source_rows = [row for row in _component_rows() if row[2] != "NO_MATCH"]
    input_path = tmp_path / "data.csv"
    output_dir = tmp_path / "split"
    _write_csv(input_path, source_rows)

    prepare_dataset(input_path, output_dir)

    emitted = [row for split in SPLITS for row in _triples(output_dir / f"{split}.csv")]
    assert Counter(emitted) == Counter(source_rows)
    assert all(row[2] != "NO_MATCH" for row in emitted)
    assert (
        "Formal NO_MATCH evaluation is unavailable because the source dataset "
        "contains no human-verified NO_MATCH examples."
    ) in (output_dir / "split_report.md").read_text("utf-8")


def test_hard_cases_are_selected_only_from_test_and_honor_thresholds(tmp_path: Path) -> None:
    source_rows = [
        row
        for index in range(20)
        for row in [
            (f"aaaaaa{index}", f"zzzzzz{index}", "DIRECT"),
            (f"bbbbbb{index}", f"yyyyyy{index}", "DERIVATION"),
            (f"account_balance_{index}", f"account_balance_{index}_x", "NO_MATCH"),
        ]
    ]
    input_path = tmp_path / "data.csv"
    output_dir = tmp_path / "split"
    _write_csv(input_path, source_rows)

    prepare_dataset(
        input_path,
        output_dir,
        hard_positive_threshold=0.95,
        hard_negative_threshold=0.05,
    )

    test_rows = set(_triples(output_dir / "test.csv"))
    hard_rows = set(_triples(output_dir / "test_hard_cases.csv"))
    assert test_rows
    assert hard_rows <= test_rows
    expected = {
        row
        for row in test_rows
        if (
            row[2] in {"DIRECT", "DERIVATION"}
            and SequenceMatcher(None, normalize_field_name(row[0]), normalize_field_name(row[1])).ratio()
            < 0.95
        )
        or (
            row[2] == "NO_MATCH"
            and SequenceMatcher(None, normalize_field_name(row[0]), normalize_field_name(row[1])).ratio()
            > 0.05
        )
    }
    assert expected == test_rows
    assert expected <= hard_rows


def test_cli_runs_from_another_directory_without_training_dependencies(tmp_path: Path) -> None:
    input_path = tmp_path / "data.csv"
    output_dir = tmp_path / "split"
    _write_csv(input_path, _component_rows())

    result = _run_cli(input_path, output_dir, "--seed", "42")

    assert result.returncode == 0, result.stdout + result.stderr
    _assert_complete_outputs(output_dir)


@pytest.mark.parametrize(
    "options",
    [
        ("--train-ratio", "0.8"),
        ("--train-ratio", "-0.1"),
        ("--val-ratio", "nan"),
        ("--test-ratio", "inf"),
        ("--hard-positive-threshold", "1.1"),
        ("--hard-negative-threshold", "-0.01"),
    ],
)
def test_invalid_numeric_options_fail_before_writing_splits(
    tmp_path: Path, options: tuple[str, ...]
) -> None:
    input_path = tmp_path / "data.csv"
    output_dir = tmp_path / "split"
    _write_csv(input_path, _component_rows())

    result = _run_cli(input_path, output_dir, *options)

    assert result.returncode != 0
    assert not (output_dir / "train.csv").exists()


@pytest.mark.parametrize("header", ["A,label", "A,B", "B,label", "A,A,B,label"])
def test_missing_or_ambiguous_required_headers_are_rejected(tmp_path: Path, header: str) -> None:
    input_path = tmp_path / "data.csv"
    input_path.write_text(header + "\n", encoding="utf-8")
    output_dir = tmp_path / "split"

    result = _run_cli(input_path, output_dir)

    assert result.returncode != 0
    assert not (output_dir / "train.csv").exists()


@pytest.mark.parametrize("rows", [[], [("", "", "")]])
def test_empty_cleaned_input_still_emits_diagnostics_and_fails_cli(
    tmp_path: Path, rows: list[Row]
) -> None:
    input_path = tmp_path / "data.csv"
    output_dir = tmp_path / "split"
    _write_csv(input_path, rows)

    result = _run_cli(input_path, output_dir)

    assert result.returncode != 0
    _assert_complete_outputs(output_dir)
    assert all(not _triples(output_dir / f"{split}.csv") for split in SPLITS)


def test_giant_component_is_never_randomly_split_even_when_three_splits_are_impossible(
    tmp_path: Path,
) -> None:
    source_rows = [("shared_hub", f"target_{index}", LABELS[index % 3]) for index in range(30)]
    input_path = tmp_path / "data.csv"
    output_dir = tmp_path / "split"
    _write_csv(input_path, source_rows)

    result = _run_cli(input_path, output_dir)

    assert result.returncode != 0
    _assert_complete_outputs(output_dir)
    splits = [_triples(output_dir / f"{split}.csv") for split in SPLITS]
    assert sorted(map(len, splits)) == [0, 0, len(source_rows)]
    assert Counter(row for rows in splits for row in rows) == Counter(source_rows)
    report = (output_dir / "split_report.md").read_text("utf-8").lower()
    assert "component" in report
    assert "30" in report
