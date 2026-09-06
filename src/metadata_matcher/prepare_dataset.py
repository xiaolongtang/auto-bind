"""Audit human-labeled CSV pairs and split indivisible field graph components.

Entirely local, standard-library-only preparation. Raw A/B values are preserved;
the model's normalization is reused for graph nodes and annotation consistency.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
import csv
import json
import logging
import math
import os
from pathlib import Path
import random
import statistics
import tempfile
from typing import Any, Iterable, Mapping, Sequence

from .preprocess import normalize_field_name


LOGGER = logging.getLogger(__name__)
LABELS = ("DIRECT", "DERIVATION", "NO_MATCH")
SPLITS = ("train", "validation", "test")
COLUMNS = ("A", "B", "label")
NO_MATCH_LIMITATION = (
    "Formal NO_MATCH evaluation is unavailable because the source dataset "
    "contains no human-verified NO_MATCH examples."
)
OUTPUT_FILES = (
    "train.csv",
    "validation.csv",
    "test.csv",
    "test_hard_cases.csv",
    "conflicting_pairs.csv",
    "rejected_rows.csv",
    "split_report.md",
    "split_statistics.json",
)


@dataclass(frozen=True)
class PairRecord:
    """An original CSV record with cached normalized graph endpoints."""

    a: str
    b: str
    label: str
    source_row: int
    normalized_a: str
    normalized_b: str

    @property
    def pair(self) -> tuple[str, str]:
        return self.a, self.b

    @property
    def normalized_pair(self) -> tuple[str, str]:
        return self.normalized_a, self.normalized_b

    def csv_values(self) -> dict[str, str]:
        return {"A": self.a, "B": self.b, "label": self.label}


@dataclass
class Component:
    """Rows connected by normalized A/B fields, including transitive links."""

    component_id: int
    records: list[PairRecord]
    fields: set[str]
    label_counts: Counter[str]

    @property
    def size(self) -> int:
        return len(self.records)


class UnionFind:
    """Disjoint sets with iterative path compression and union by size."""

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.sizes: dict[str, int] = {}

    def find(self, field: str) -> str:
        if field not in self.parent:
            self.parent[field] = field
            self.sizes[field] = 1
        root = field
        while self.parent[root] != root:
            root = self.parent[root]
        while field != root:
            parent = self.parent[field]
            self.parent[field] = root
            field = parent
        return root

    def union(self, a: str, b: str) -> None:
        a_root, b_root = self.find(a), self.find(b)
        if a_root == b_root:
            return
        if self.sizes[a_root] < self.sizes[b_root]:
            a_root, b_root = b_root, a_root
        self.parent[b_root] = a_root
        self.sizes[a_root] += self.sizes[b_root]


def _distribution(records: Sequence[PairRecord]) -> dict[str, dict[str, Any]]:
    counts = Counter(row.label for row in records)
    return {
        label: {
            "count": counts[label],
            "ratio": counts[label] / len(records) if records else 0.0,
        }
        for label in LABELS
    }


def read_and_clean(
    input_path: Path,
) -> tuple[
    list[PairRecord], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]
]:
    """Audit invalid rows, quarantine every conflict occurrence, and dedup.

    No conflicting label is guessed. Normalized conflicts are also quarantined
    because the existing model rejects contradictory normalized pairs. Explicitly
    synthetic records cannot become human truth by dropping their marker column.
    """

    valid: list[PairRecord] = []
    rejected: list[dict[str, Any]] = []
    missing_counts: Counter[str] = Counter()
    invalid_labels: Counter[str] = Counter()
    original_labels: Counter[str] = Counter()
    rejection_reasons: Counter[str] = Counter()
    total = 0
    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, strict=True)
        headers = reader.fieldnames
        if headers is None:
            raise ValueError("Input CSV is empty; required header: A,B,label")
        if len(headers) != len(set(headers)):
            raise ValueError("Input CSV has duplicate column names")
        missing_columns = set(COLUMNS).difference(headers)
        if missing_columns:
            raise ValueError(
                f"Input CSV is missing required columns: {sorted(missing_columns)}"
            )
        for source_row, row in enumerate(reader, start=2):
            total += 1
            if None in row or any(value is None for value in row.values()):
                raise ValueError(
                    f"Malformed CSV record {source_row}: inconsistent column count"
                )
            a, b, original_label = (row[column] for column in COLUMNS)
            label = original_label.strip().upper()
            original_labels[label] += 1
            reasons: list[str] = []
            for column, value in (("A", a), ("B", b), ("label", original_label)):
                if not value.strip():
                    missing_counts[column] += 1
                    reasons.append(f"missing_{column}")
            if label and label not in LABELS:
                invalid_labels[label] += 1
                reasons.append("invalid_label")
            normalized_a, normalized_b = (
                normalize_field_name(a),
                normalize_field_name(b),
            )
            if a.strip() and not normalized_a:
                reasons.append("empty_normalized_A")
            if b.strip() and not normalized_b:
                reasons.append("empty_normalized_B")
            if "synthetic" in row:
                marker = row["synthetic"].strip().lower()
                if marker in {"true", "1", "yes"}:
                    reasons.append("synthetic_record")
                elif marker not in {"false", "0", "no"}:
                    reasons.append("invalid_synthetic_marker")
            if reasons:
                rejection_reasons.update(reasons)
                rejected.append(
                    {
                        "A": a,
                        "B": b,
                        "label": original_label,
                        "source_row": source_row,
                        "reason": ";".join(reasons),
                    }
                )
            else:
                valid.append(
                    PairRecord(a, b, label, source_row, normalized_a, normalized_b)
                )

    pair_counts = Counter(row.pair for row in valid)
    triples = Counter((row.a, row.b, row.label) for row in valid)
    raw_labels: dict[tuple[str, str], set[str]] = defaultdict(set)
    normalized_labels: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in valid:
        raw_labels[row.pair].add(row.label)
        normalized_labels[row.normalized_pair].add(row.label)
    raw_conflicts = {pair for pair, labels in raw_labels.items() if len(labels) > 1}
    normalized_conflicts = {
        pair for pair, labels in normalized_labels.items() if len(labels) > 1
    }
    conflicts: list[dict[str, Any]] = []
    clean: list[PairRecord] = []
    seen: set[tuple[str, str, str]] = set()
    duplicates_removed = 0
    for row in valid:
        if row.normalized_pair in normalized_conflicts:
            conflicts.append(
                {
                    **row.csv_values(),
                    "source_row": row.source_row,
                    "normalized_A": row.normalized_a,
                    "normalized_B": row.normalized_b,
                    "conflict_type": "raw_pair"
                    if row.pair in raw_conflicts
                    else "normalized_pair",
                }
            )
        else:
            triple = (row.a, row.b, row.label)
            if triple in seen:
                duplicates_removed += 1
            else:
                seen.add(triple)
                clean.append(row)

    quality = {
        "total_rows": total,
        "original_label_counts": {label: original_labels[label] for label in LABELS},
        "missing": {column: missing_counts[column] for column in COLUMNS},
        "rows_with_missing_values": sum(
            any(reason.startswith("missing_") for reason in row["reason"].split(";"))
            for row in rejected
        ),
        "invalid_label_counts": dict(sorted(invalid_labels.items())),
        "rejected_rows": len(rejected),
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
        "eligible_rows_before_conflicts_and_deduplication": len(valid),
        "duplicate_pair_groups": sum(count > 1 for count in pair_counts.values()),
        "duplicate_pair_rows": sum(count - 1 for count in pair_counts.values()),
        "exact_duplicate_rows": sum(count - 1 for count in triples.values()),
        "duplicate_rows_removed": duplicates_removed,
        "conflicting_raw_pairs": len(raw_conflicts),
        "conflicting_normalized_pairs": len(normalized_conflicts),
        "conflicting_rows": len(conflicts),
        "clean_rows": len(clean),
        "clean_label_distribution": _distribution(clean),
    }
    assert total == len(clean) + len(conflicts) + len(rejected) + duplicates_removed
    return clean, conflicts, rejected, quality


def build_components(records: Sequence[PairRecord]) -> list[Component]:
    """Connect every A/B pair regardless of label; never split an edge."""

    graph = UnionFind()
    for row in records:
        graph.union(row.normalized_a, row.normalized_b)
    groups: dict[str, list[PairRecord]] = defaultdict(list)
    for row in records:
        groups[graph.find(row.normalized_a)].append(row)
    return [
        Component(
            index,
            rows,
            {field for row in rows for field in row.normalized_pair},
            Counter(row.label for row in rows),
        )
        for index, rows in enumerate(groups.values(), start=1)
    ]


def split_components(
    components: Sequence[Component],
    ratios: Sequence[float],
    seed: int,
) -> tuple[dict[str, list[PairRecord]], dict[int, str]]:
    """Seeded largest-first greedy assignment, followed by improving moves.

    Balance row/label target errors. Components remain indivisible even when exact
    ratios or three nonempty sets are impossible. Work is linear per move pass.
    """

    rng = random.Random(seed)
    ordered = list(components)
    rng.shuffle(ordered)
    ordered.sort(key=lambda component: -component.size)
    total = sum(component.size for component in components)
    label_totals: Counter[str] = Counter()
    for component in components:
        label_totals.update(component.label_counts)
    active = [i for i, ratio in enumerate(ratios) if ratio > 0]
    row_counts = [0, 0, 0]
    class_counts: list[Counter[str]] = [Counter(), Counter(), Counter()]
    component_counts = [0, 0, 0]
    assignments: dict[int, int] = {}
    present_labels = [label for label in LABELS if label_totals[label]]

    def cost(index: int, size: int, labels: Mapping[str, int]) -> float:
        target = total * ratios[index]
        row_error = (size - target) ** 2 / max(target, 1.0) / max(total, 1)
        label_error = 0.0
        for label in present_labels:
            label_total = label_totals[label]
            label_target = label_total * ratios[index]
            label_error += (
                (labels.get(label, 0) - label_target) ** 2
                / max(label_target, 1.0)
                / label_total
            )
        return row_error + 0.2 * label_error / max(len(present_labels), 1)

    def delta(index: int, component: Component, sign: int) -> float:
        labels = {
            label: class_counts[index][label] + sign * component.label_counts[label]
            for label in LABELS
        }
        return cost(index, row_counts[index] + sign * component.size, labels) - cost(
            index, row_counts[index], class_counts[index]
        )

    def apply(index: int, component: Component, sign: int) -> None:
        row_counts[index] += sign * component.size
        component_counts[index] += sign
        for label in LABELS:
            class_counts[index][label] += sign * component.label_counts[label]

    for position, component in enumerate(ordered):
        empty = [index for index in active if component_counts[index] == 0]
        remaining = len(ordered) - position
        candidates = list(active)
        if len(ordered) >= len(active) and remaining == len(empty):
            candidates = empty
        rng.shuffle(candidates)
        selected = min(candidates, key=lambda index: delta(index, component, 1))
        assignments[component.component_id] = selected
        apply(selected, component, 1)

    for _ in range(10):
        moved = False
        for component in reversed(ordered):
            current = assignments[component.component_id]
            if component_counts[current] <= 1:
                continue
            best, improvement = current, -1e-12
            for candidate in active:
                if candidate == current:
                    continue
                change = delta(current, component, -1) + delta(candidate, component, 1)
                if change < improvement:
                    best, improvement = candidate, change
            if best != current:
                apply(current, component, -1)
                apply(best, component, 1)
                assignments[component.component_id] = best
                moved = True
        if not moved:
            break

    splits: dict[str, list[PairRecord]] = {name: [] for name in SPLITS}
    named_assignments = {}
    for component in components:
        name = SPLITS[assignments[component.component_id]]
        named_assignments[component.component_id] = name
        splits[name].extend(component.records)
    for rows in splits.values():
        rows.sort(key=lambda row: row.source_row)
    return splits, named_assignments


def check_leakage(
    splits: Mapping[str, Sequence[PairRecord]],
) -> dict[str, dict[str, Any]]:
    """Check A and B together for all three pairwise split intersections."""

    fields = {
        name: {field for row in rows for field in row.normalized_pair}
        for name, rows in splits.items()
    }
    result = {}
    for left, right in (
        ("train", "test"),
        ("train", "validation"),
        ("validation", "test"),
    ):
        shared = sorted(fields[left] & fields[right])
        result[f"{left}_{right}"] = {
            "shared_field_count": len(shared),
            "shared_fields": shared,
        }
    return result


def _is_subsequence(short: str, long: str) -> bool:
    iterator = iter(long)
    return all(
        any(character == candidate for candidate in iterator) for character in short
    )


def _possible_abbreviation(a: str, b: str) -> bool:
    """A conservative lexical hint, not an assertion of semantic truth."""

    a_tokens, b_tokens = a.split("_"), b.split("_")
    for short, long in ((a_tokens, b_tokens), (b_tokens, a_tokens)):
        if (
            len(short) == len(long)
            and any(
                1 < len(short_token) < len(long_token)
                and _is_subsequence(short_token, long_token)
                for short_token, long_token in zip(short, long)
            )
            and all(
                short_token == long_token
                or (len(short_token) >= 2 and _is_subsequence(short_token, long_token))
                for short_token, long_token in zip(short, long)
            )
        ):
            return True
        if (
            len(short) == 1
            and len(long) > 1
            and short[0] == "".join(t[0] for t in long)
        ):
            return True
    return False


def identify_hard_cases(
    records: Sequence[PairRecord],
    positive_threshold: float,
    negative_threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select original test rows using lexical hints without changing labels."""

    cases: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for row in records:
        a, b = row.normalized_pair
        similarity = SequenceMatcher(None, a, b, autojunk=False).ratio()
        reasons: list[str] = []
        if row.label in {"DIRECT", "DERIVATION"} and similarity < positive_threshold:
            reasons.append("hard_positive")
        if row.label == "NO_MATCH" and similarity > negative_threshold:
            reasons.append("hard_negative")
        if a == b and row.a != row.b:
            reasons.append("format_variant")
        elif row.label != "NO_MATCH" and similarity > negative_threshold:
            reasons.append("near_spelling_variant")
        if _possible_abbreviation(a, b):
            reasons.append("possible_abbreviation")
        if reasons:
            counts.update(reasons)
            cases.append(
                {
                    **row.csv_values(),
                    "source_row": row.source_row,
                    "similarity": round(similarity, 6),
                    "reasons": ";".join(reasons),
                }
            )
    return cases, {
        "total_rows": len(cases),
        "hard_positive": counts["hard_positive"],
        "hard_negative": counts["hard_negative"],
        "reason_counts": dict(sorted(counts.items())),
        "positive_similarity_threshold": positive_threshold,
        "negative_similarity_threshold": negative_threshold,
    }


def _component_statistics(
    components: Sequence[Component],
    assignments: Mapping[int, str],
    total: int,
) -> dict[str, Any]:
    sizes = sorted(component.size for component in components)
    largest = sorted(
        components, key=lambda component: (-component.size, component.component_id)
    )[:20]
    return {
        "count": len(components),
        "size_unit": "cleaned rows (not fields)",
        "size_distribution": {
            str(size): count for size, count in sorted(Counter(sizes).items())
        },
        "min_rows": min(sizes, default=0),
        "median_rows": statistics.median(sizes) if sizes else 0,
        "max_rows": max(sizes, default=0),
        "largest_ratio": max(sizes, default=0) / total if total else 0.0,
        "top_20": [
            {
                "component_id": component.component_id,
                "rows": component.size,
                "fields": len(component.fields),
                "ratio": component.size / total,
                "split": assignments[component.component_id],
                "label_counts": {
                    label: component.label_counts[label] for label in LABELS
                },
                "sample_fields": sorted(component.fields)[:10],
            }
            for component in largest
        ],
    }


def render_report(report: Mapping[str, Any]) -> str:
    """Render a complete Markdown audit, including component size distribution."""

    quality = report["data_quality"]
    lines = [
        "# Dataset Split Report",
        "",
        "## Original Dataset",
        "",
        f"- Source: `{report['input_path']}`",
        f"- Total rows: {quality['total_rows']}",
        "- Labels are trimmed and uppercased; A/B text remains unchanged.",
        "- Original class ratios use all input rows, including rejected/duplicate/conflicting rows.",
        "",
        "| Label | Count | Ratio |",
        "| --- | ---: | ---: |",
    ]
    for label, count in quality["original_label_counts"].items():
        ratio = count / quality["total_rows"] if quality["total_rows"] else 0
        lines.append(f"| {label} | {count} | {ratio:.2%} |")
    lines.extend(
        [
            "",
            "## Data Quality",
            "",
            f"- Missing A / B / label: {quality['missing']['A']} / {quality['missing']['B']} / {quality['missing']['label']}",
            f"- Rows with missing values: {quality['rows_with_missing_values']}",
            f"- Unsupported labels: `{json.dumps(quality['invalid_label_counts'], ensure_ascii=False)}`",
            f"- Rejected rows: {quality['rejected_rows']} (see rejected_rows.csv)",
            f"- Rejection reason counts (may overlap): `{json.dumps(quality['rejection_reasons'], ensure_ascii=False)}`",
            f"- Duplicate pair groups: {quality['duplicate_pair_groups']}; excess pair rows: {quality['duplicate_pair_rows']}",
            f"- Exact duplicate rows: {quality['exact_duplicate_rows']}; removed outside conflicts: {quality['duplicate_rows_removed']}",
            f"- Conflicting raw pairs: {quality['conflicting_raw_pairs']}",
            f"- Conflicting normalized pairs (includes raw conflicts): {quality['conflicting_normalized_pairs']}",
            f"- Quarantined conflicting rows: {quality['conflicting_rows']} (see conflicting_pairs.csv)",
            f"- Clean rows available for splitting: {quality['clean_rows']}",
            "- Duplicate/conflict counts apply to rows with valid fields, labels and provenance markers.",
            "- All occurrences of conflicting pairs are quarantined, including duplicates; no label is guessed.",
            "- Normalized conflicts are also quarantined to satisfy the model's existing annotation validation.",
            "- Row accounting: original = rejected + quarantined conflicts + removed duplicates + clean rows.",
            "- source_row is the one-based CSV record number including the header, not a physical line number.",
            "",
            "## Split Strategy",
            "",
            "Random row splitting can expose the same field (including spelling/case/separator aliases) in training and testing, inflating accuracy.",
            "Every normalized field is a graph node. Every labeled pair, including NO_MATCH, is an undirected edge.",
            "Connected components are assigned whole to train, validation or test; field disjointness takes priority over exact ratios.",
            "Normalization uses Unicode NFKC, trim, CamelCase splitting, lowercase, separator replacement and underscore cleanup, shared with the model.",
            f"A seeded largest-first greedy algorithm balances row/label target errors, followed by up to 10 improving move passes. Seed: {report['seed']}.",
            "This is deterministic for the same input order and parameters; the heuristic does not guarantee a global optimum or class coverage.",
            "No synthetic rows are generated. Validation/test/hard cases contain only retained source records.",
            "",
            "## Dataset Sizes",
            "",
            "Ratios below use the cleaned, nonconflicting dataset as the denominator.",
            "",
            "| Dataset | Rows | Target | Actual | Components |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, split in report["splits"].items():
        lines.append(
            f"| {name} | {split['rows']} | {split['target_ratio']:.2%} | {split['actual_ratio']:.2%} | {split['components']} |"
        )
    lines.extend(
        [
            "",
            "## Label Distribution",
            "",
            "| Dataset | Label | Count | Ratio within dataset |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for name, split in report["splits"].items():
        for label, values in split["label_distribution"].items():
            lines.append(
                f"| {name} | {label} | {values['count']} | {values['ratio']:.2%} |"
            )
    lines.extend(["", "## Leakage Check", ""])
    for name, check in report["leakage_check"].items():
        lines.append(
            f"- {name.replace('_', '/').title()} shared normalized fields: {check['shared_field_count']}"
        )
    components = report["connected_components"]
    lines.extend(
        [
            "",
            "## Connected Components",
            "",
            f"- Component count: {components['count']}",
            f"- Largest component: {components['max_rows']} rows ({components['largest_ratio']:.2%})",
            f"- Minimum / median / maximum rows: {components['min_rows']} / {components['median_rows']} / {components['max_rows']}",
            "- Giant components often arise from shared names such as id, date, type or price. They are never broken to improve ratios.",
            "",
            "### Component size distribution",
            "",
            "| Rows per component | Number of components |",
            "| ---: | ---: |",
        ]
    )
    for size, count in components["size_distribution"].items():
        lines.append(f"| {size} | {count} |")
    lines.extend(
        [
            "",
            "### Largest 20 components",
            "",
            "| ID | Rows | Fields | Share | Split | DIRECT | DERIVATION | NO_MATCH |",
            "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
        ]
    )
    for component in components["top_20"]:
        counts = component["label_counts"]
        lines.append(
            f"| {component['component_id']} | {component['rows']} | {component['fields']} | {component['ratio']:.2%} | {component['split']} | {counts['DIRECT']} | {counts['DERIVATION']} | {counts['NO_MATCH']} |"
        )
    hard = report["hard_test_cases"]
    lines.extend(
        [
            "",
            "## Hard Test Cases",
            "",
            f"- Selected original test rows: {hard['total_rows']}",
            f"- Hard positives (DIRECT/DERIVATION similarity < {hard['positive_similarity_threshold']}): {hard['hard_positive']}",
            f"- Hard negatives (NO_MATCH similarity > {hard['negative_similarity_threshold']}): {hard['hard_negative']}",
            f"- Reason counts (may overlap): `{json.dumps(hard['reason_counts'], ensure_ascii=False)}`",
            "- Similarity is SequenceMatcher on normalized names (autojunk=False). Format, abbreviation and near-spelling flags are potential cases for manual review, not proven typos or semantics.",
            "- Hard cases are a subset of test.csv, not an independent evaluation sample.",
            "",
            "## Limitations",
            "",
            "- Human verification is an input requirement; the CSV and this program cannot establish annotation provenance. Confirm source labels were manually verified before reporting formal metrics.",
            "- The split prevents normalized field reuse, but cannot guarantee semantic, business-system, alias or temporal disjointness.",
            "- A/B direction is retained; reverse pairs are not automatically treated as equivalent annotations.",
            f"- Source NO_MATCH rows: {quality['original_label_counts']['NO_MATCH']}; clean NO_MATCH rows: {quality['clean_label_distribution']['NO_MATCH']['count']}.",
            f"- Validation NO_MATCH rows: {report['splits']['validation']['label_distribution']['NO_MATCH']['count']}; test NO_MATCH rows: {report['splits']['test']['label_distribution']['NO_MATCH']['count']}.",
            f"- Structural status: **{report['status']}**. Ready for the existing training pipeline: **{report['ready_for_training']}**.",
        ]
    )
    lines.extend(f"- {warning}" for warning in report["warnings"])
    if not report["warnings"]:
        lines.append("- No additional structural warnings.")
    return "\n".join(lines) + "\n"


def _write_csv(
    path: Path, columns: Sequence[str], rows: Iterable[Mapping[str, Any]]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _validate_parameters(ratios: Sequence[float], thresholds: Sequence[float]) -> None:
    if any(not math.isfinite(ratio) or ratio < 0 or ratio > 1 for ratio in ratios):
        raise ValueError("Split ratios must be finite numbers in [0, 1]")
    if not math.isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("Train, validation and test ratios must sum to 1")
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in thresholds):
        raise ValueError("Hard-case similarity thresholds must be finite and in [0, 1]")


def prepare_dataset(
    input_path: str | Path = "data.csv",
    output_dir: str | Path = "data_split",
    *,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    hard_positive_threshold: float = 0.4,
    hard_negative_threshold: float = 0.7,
    example_data: bool = False,
) -> dict[str, Any]:
    """Write the CSV/report artifacts and return their structured statistics.

    Invalid rows and conflicts are audited and excluded; invalid schema/parameters
    raise an error. An unusable split still produces artifacts with status=unusable
    (CLI exit 2). Zero-ratio splits are allowed, but the training pipeline needs all
    three sets to be nonempty. No source records are generated or augmented.
    """

    input_path, output_dir = Path(input_path), Path(output_dir)
    ratios = (train_ratio, val_ratio, test_ratio)
    _validate_parameters(ratios, (hard_positive_threshold, hard_negative_threshold))
    if input_path.resolve() in {(output_dir / name).resolve() for name in OUTPUT_FILES}:
        raise ValueError("Input path must not be one of the generated output files")
    clean, conflicts, rejected, quality = read_and_clean(input_path)
    components = build_components(clean)
    splits, assignments = split_components(components, ratios, seed)
    leakage = check_leakage(splits)
    if any(check["shared_field_count"] for check in leakage.values()):
        raise RuntimeError(
            "FIELD LEAKAGE ERROR: shared normalized fields detected; outputs not published"
        )
    hard_cases, hard_stats = identify_hard_cases(
        splits["test"], hard_positive_threshold, hard_negative_threshold
    )
    component_stats = _component_statistics(components, assignments, len(clean))
    warnings: list[str] = []
    if rejected:
        warnings.append(
            f"{len(rejected)} invalid or explicitly synthetic rows were excluded; inspect rejected_rows.csv."
        )
    if conflicts:
        warnings.append(
            f"{len(conflicts)} conflicting rows require manual annotation review; inspect conflicting_pairs.csv."
        )
    if not clean:
        warnings.append(
            "No valid nonconflicting rows remain; training/evaluation is unavailable."
        )
    if component_stats["largest_ratio"] > max(ratios):
        warnings.append(
            "A giant connected component exceeds the largest target split. Exact target ratios are impossible without field leakage; it remains intact."
        )
    if len(components) < sum(ratio > 0 for ratio in ratios):
        warnings.append(
            "There are fewer connected components than requested nonempty splits. At least one requested dataset must remain empty; no random fallback is used."
        )
    split_stats = {}
    for index, (name, rows) in enumerate(splits.items()):
        actual = len(rows) / len(clean) if clean else 0.0
        split_stats[name] = {
            "rows": len(rows),
            "target_ratio": ratios[index],
            "actual_ratio": actual,
            "components": sum(split == name for split in assignments.values()),
            "normalized_field_count": len(
                {field for row in rows for field in row.normalized_pair}
            ),
            "label_distribution": _distribution(rows),
        }
        if not rows and ratios[index] > 0:
            warnings.append(
                f"{name} is empty; it cannot support training or formal evaluation."
            )
        if abs(actual - ratios[index]) > 0.05:
            warnings.append(
                f"{name} actual ratio {actual:.2%} differs from target {ratios[index]:.2%} by more than 5 percentage points because whole components are indivisible."
            )
        absent = [
            label
            for label in LABELS
            if not split_stats[name]["label_distribution"][label]["count"]
        ]
        if absent:
            warnings.append(
                f"{name} lacks these source label classes: {', '.join(absent)}. Three-class coverage is incomplete."
            )
    if quality["original_label_counts"]["NO_MATCH"] == 0:
        warnings.append(NO_MATCH_LIMITATION)
    elif quality["clean_label_distribution"]["NO_MATCH"]["count"] == 0:
        warnings.append(
            "All source NO_MATCH rows were excluded during cleaning; formal NO_MATCH evaluation is unavailable."
        )
    for name in ("validation", "test"):
        if split_stats[name]["label_distribution"]["NO_MATCH"]["count"] == 0:
            warnings.append(
                f"Formal NO_MATCH evaluation is unavailable in {name}: it contains no retained human-verified NO_MATCH examples. No synthetic negatives were added."
            )
    if example_data:
        warnings.append(
            "EXAMPLE DATA ONLY: these records are demonstration/test fixtures, not verified enterprise ground truth; metrics are not formal evaluation evidence."
        )
    requested_nonempty = bool(clean) and all(
        splits[name] for name, ratio in zip(SPLITS, ratios) if ratio > 0
    )
    report = {
        "schema_version": 1,
        "input_path": str(input_path),
        "seed": seed,
        "example_data": example_data,
        "strategy": "normalized_field_disjoint_connected_components",
        "status": "unusable"
        if not requested_nonempty
        else ("warning" if warnings else "ok"),
        "ready_for_training": all(bool(splits[name]) for name in SPLITS),
        "data_quality": quality,
        "splits": split_stats,
        "leakage_check": leakage,
        "connected_components": component_stats,
        "hard_test_cases": hard_stats,
        "warnings": warnings,
    }
    # Stage all files before publishing; reading/validation errors cannot
    # overwrite a previous split. Each individual file replacement is atomic.
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=output_dir) as temporary:
        staging = Path(temporary)
        for name, rows in splits.items():
            _write_csv(
                staging / f"{name}.csv", COLUMNS, (row.csv_values() for row in rows)
            )
        _write_csv(
            staging / "test_hard_cases.csv",
            (*COLUMNS, "source_row", "similarity", "reasons"),
            hard_cases,
        )
        _write_csv(
            staging / "conflicting_pairs.csv",
            (*COLUMNS, "source_row", "normalized_A", "normalized_B", "conflict_type"),
            conflicts,
        )
        _write_csv(
            staging / "rejected_rows.csv", (*COLUMNS, "source_row", "reason"), rejected
        )
        (staging / "split_report.md").write_text(
            render_report(report), encoding="utf-8"
        )
        (staging / "split_statistics.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        for filename in OUTPUT_FILES:
            os.replace(staging / filename, output_dir / filename)
    LOGGER.info(
        "Input rows: %d | DIRECT: %d | DERIVATION: %d | NO_MATCH: %d",
        quality["total_rows"],
        *(quality["original_label_counts"][label] for label in LABELS),
    )
    LOGGER.info(
        "Duplicate pair rows: %d | conflicting raw/normalized pairs: %d/%d | conflicting rows: %d",
        quality["duplicate_pair_rows"],
        quality["conflicting_raw_pairs"],
        quality["conflicting_normalized_pairs"],
        quality["conflicting_rows"],
    )
    LOGGER.info(
        "Component size distribution (rows: count): %s",
        component_stats["size_distribution"],
    )
    LOGGER.info(
        "Largest 20 components (id, rows, split): %s",
        [
            (item["component_id"], item["rows"], item["split"])
            for item in component_stats["top_20"]
        ],
    )
    for name, stats in split_stats.items():
        LOGGER.info(
            "%s: %d rows (%.2f%%; target %.2f%%)",
            name.title(),
            stats["rows"],
            stats["actual_ratio"] * 100,
            stats["target_ratio"] * 100,
        )
    for name, check in leakage.items():
        LOGGER.info(
            "%s shared fields: %d",
            name.replace("_", "/").title(),
            check["shared_field_count"],
        )
    for warning in warnings:
        LOGGER.warning("%s", warning)
    LOGGER.info(
        "Dataset status: %s | Ready for training: %s | Report: %s",
        report["status"],
        report["ready_for_training"],
        output_dir / "split_report.md",
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Clean labeled pairs and create offline field-disjoint train/validation/test datasets."
    )
    parser.add_argument("--input", type=Path, default=Path("data.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data_split"))
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--hard-positive-threshold",
        type=float,
        default=0.4,
        help="Select DIRECT/DERIVATION with normalized string similarity below this threshold",
    )
    parser.add_argument(
        "--hard-negative-threshold",
        type=float,
        default=0.7,
        help="Select NO_MATCH with normalized string similarity above this threshold",
    )
    parser.add_argument(
        "--example-data",
        action="store_true",
        help="Mark reports as demonstration data, unsuitable for formal evaluation",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        report = prepare_dataset(
            args.input,
            args.output_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            hard_positive_threshold=args.hard_positive_threshold,
            hard_negative_threshold=args.hard_negative_threshold,
            example_data=args.example_data,
        )
    except (OSError, UnicodeError, csv.Error, ValueError, RuntimeError) as exc:
        LOGGER.error("Dataset preparation failed: %s", exc)
        return 2
    return 2 if report["status"] == "unusable" else 0


if __name__ == "__main__":
    raise SystemExit(main())
