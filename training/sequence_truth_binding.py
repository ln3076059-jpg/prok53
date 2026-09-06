"""Reconstruct final truth from exact canonical sequences in an external lock."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from training.build_event_truth_from_sequences import (
    CONTEXT_FIELDNAMES, EVENT_FIELDNAMES, convert_sequence_to_context, convert_sequence_to_events,
)
from training.common import sha256_file, stable_json_hash
from training.validate_event_sequence_annotations import load_schema, validate_annotation


def sequence_bindings(external_lock: dict) -> list[dict]:
    groups = external_lock.get("sequence_annotations")
    if not isinstance(groups, dict) or not groups:
        raise ValueError("external lock is missing canonical sequence bindings")
    bindings = []
    seen = set()
    for sample_id, records in groups.items():
        if not isinstance(records, list) or not records:
            raise ValueError("external lock has empty or invalid sequence bindings")
        for record in records:
            if not isinstance(record, dict) or not record.get("path") or not record.get("sha256"):
                raise ValueError("invalid canonical sequence path/SHA binding")
            path = str(Path(record["path"]).resolve())
            if path in seen:
                raise ValueError("duplicate canonical sequence binding")
            seen.add(path)
            bindings.append({"sample_id": sample_id, "path": path, "sha256": record["sha256"]})
    return sorted(bindings, key=lambda row: (row["sample_id"], row["path"], row["sha256"]))


def verify_truth_source_binding(
    external_lock: dict, csv_path: Path, kind: str, truth_lock: dict | None = None,
) -> dict:
    required = (
        external_lock.get("require_canonical_sequence_annotations") is True
        or "sequence_annotations" in external_lock
        or "source_sequence_set_sha256" in external_lock
        or (truth_lock is not None and "source_sequence_set_sha256" in truth_lock)
    )
    if not required:
        return {}  # Historical/custom locks without canonical sequence sources.
    bindings = sequence_bindings(external_lock)
    digest = stable_json_hash(bindings)
    if (
        external_lock.get("require_canonical_sequence_annotations") is True
        or "source_sequence_set_sha256" in external_lock
    ) and external_lock.get("source_sequence_set_sha256") != digest:
        raise ValueError("external source_sequence_set_sha256 mismatch")
    schema = load_schema(
        Path(__file__).resolve().parents[1] / "datasets/schemas/v2_event_sequence_annotation.schema.json"
    )
    if kind == "event":
        fields, convert = EVENT_FIELDNAMES, convert_sequence_to_events
    elif kind == "context":
        fields, convert = CONTEXT_FIELDNAMES, convert_sequence_to_context
    else:
        raise ValueError(f"unknown truth kind: {kind}")
    expected = []
    for binding in bindings:
        path = Path(binding["path"])
        if not path.is_file() or sha256_file(path) != binding["sha256"]:
            raise ValueError("canonical sequence source missing or SHA256 mismatch")
        annotation = json.loads(path.read_text(encoding="utf-8"))
        errors = validate_annotation(
            annotation, schema,
            require_review_evidence=external_lock.get("require_sequence_review_evidence") is True,
        )
        if errors:
            raise ValueError("invalid frozen canonical sequence: " + "; ".join(errors))
        if annotation["review_provenance"].get("identity_manifest_sha256") != external_lock.get("identity_manifest_sha256"):
            raise ValueError("canonical sequence identity manifest SHA256 mismatch")
        expected.extend(convert(annotation))

    def canonical_rows(rows):
        normalized = [
            {field: "" if row.get(field) is None else str(row[field]) for field in fields}
            for row in rows
        ]
        return sorted(normalized, key=lambda row: json.dumps(row, sort_keys=True))

    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if set(reader.fieldnames or []) != set(fields) or len(reader.fieldnames or []) != len(fields):
            raise ValueError(f"{kind} CSV columns do not match canonical exporter")
        actual = list(reader)
    if any(set(row) != set(fields) or any(v is None for v in row.values()) for row in actual):
        raise ValueError(f"{kind} CSV has malformed rows")
    if canonical_rows(actual) != canonical_rows(expected):
        raise ValueError(f"{kind} CSV does not match frozen canonical sequence rows")
    provenance = {
        "source_sequence_set_sha256": digest,
        "source_sequence_bindings": bindings,
        "canonical_truth_rows_sha256": stable_json_hash(canonical_rows(expected)),
    }
    if truth_lock is not None:
        for field, expected_value in provenance.items():
            if truth_lock.get(field) != expected_value:
                raise ValueError(f"{kind} truth lock {field} does not match external canonical sources")
    return provenance
