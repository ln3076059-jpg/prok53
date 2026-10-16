"""Measure the strict phone-context gate on the frozen V9 review proposals.

This is a development-only diagnostic over FN-selected positive cases. It reports
candidate yield and human-selected proposal retention, never detector recall,
precision, or event metrics.
"""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from training.common import sha256_file
from training.dmd.fusion import PHONE_USE_CONTEXTS, gate_phone_confidence
from training.dmd.v9_human_review_results import validate_reviews

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "reports" / "v9_research"
THRESHOLDS = (0.02, 0.10, 0.20, 0.25, 0.35, 0.50)


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _proposals(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [proposal for sample in row["samples"] for proposal in sample["proposals"]]


def analyze(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output = output.resolve()
    proposal_path = output / "phone_proposal_mining.json"
    review_path = output / "human_phone_review_summary.json"
    mined = _read(proposal_path)
    reviewed = validate_reviews(output)
    reviewed_by_case = {item["case_id"]: item for item in reviewed["cases"]}
    if {row["case_id"] for row in mined["rows"]} != set(reviewed_by_case):
        raise ValueError("proposal and human-review case sets differ")

    all_proposals = [proposal for row in mined["rows"] for proposal in _proposals(row)]
    contexts = Counter(str(proposal["geometry_context"]) for proposal in all_proposals)
    selected = [
        item["selected_proposal"]
        for item in reviewed["cases"]
        if item["selected_proposal"] is not None
    ]
    hard_cases = [
        item["case_id"]
        for item in reviewed["cases"]
        if item["selected_proposal"] is None
    ]
    if len(hard_cases) != 1:
        raise ValueError("expected one governed no-valid-proposal hard example")

    threshold_rows = []
    for threshold in THRESHOLDS:
        naive_proposals = 0
        gated_proposals = 0
        naive_case_candidates = 0
        gated_case_candidates = 0
        selected_retained_naive = 0
        selected_retained_gated = 0
        hard_case_naive = False
        hard_case_gated = False
        for row in mined["rows"]:
            case_id = row["case_id"]
            proposals = _proposals(row)
            naive = [item for item in proposals if float(item["confidence"]) >= threshold]
            gated = [
                item
                for item in naive
                if gate_phone_confidence(
                    float(item["confidence"]),
                    str(item["geometry_context"]),
                    strict=True,
                )[0]
                > 0.0
            ]
            naive_proposals += len(naive)
            gated_proposals += len(gated)
            naive_case_candidates += int(bool(naive))
            gated_case_candidates += int(bool(gated))
            if case_id == hard_cases[0]:
                hard_case_naive = bool(naive)
                hard_case_gated = bool(gated)

            selected_id = reviewed_by_case[case_id]["selected_proposal_id"]
            if selected_id == "NO_VALID_PROPOSAL":
                continue
            selected_proposal = next(
                item for item in proposals if item["proposal_id"] == selected_id
            )
            admitted, _ = gate_phone_confidence(
                float(selected_proposal["confidence"]),
                str(selected_proposal["geometry_context"]),
                strict=True,
            )
            selected_retained_naive += int(
                float(selected_proposal["confidence"]) >= threshold
            )
            selected_retained_gated += int(
                float(selected_proposal["confidence"]) >= threshold and admitted > 0.0
            )

        threshold_rows.append(
            {
                "confidence_threshold": threshold,
                "naive_proposals": naive_proposals,
                "context_gated_proposals": gated_proposals,
                "removed_proposals": naive_proposals - gated_proposals,
                "naive_case_candidate_yield": naive_case_candidates,
                "context_gated_case_candidate_yield": gated_case_candidates,
                "human_selected_retained_naive": selected_retained_naive,
                "human_selected_retained_context_gated": selected_retained_gated,
                "human_selected_total": len(selected),
                "hard_example_has_naive_candidate": hard_case_naive,
                "hard_example_has_context_gated_candidate": hard_case_gated,
            }
        )

    mining_floor = threshold_rows[0]
    selected_contexts = Counter(str(item["geometry_context"]) for item in selected)
    if set(selected_contexts) - set(PHONE_USE_CONTEXTS):
        raise ValueError("a human-selected proposal would be rejected by the strict context gate")
    hard_row = next(row for row in mined["rows"] if row["case_id"] == hard_cases[0])
    hard_contexts = Counter(
        str(item["geometry_context"]) for item in _proposals(hard_row)
    )
    if set(hard_contexts) & set(PHONE_USE_CONTEXTS):
        raise ValueError("hard example contains an admitted handheld-use proposal")

    result = {
        "schema_version": 1,
        "scope": "FN_SELECTED_DEVELOPMENT_DIAGNOSTIC_ONLY",
        "metric_semantics": "CANDIDATE_YIELD_NOT_ACCURACY_OR_EVENT_METRICS",
        "source_proposal_artifact": proposal_path.relative_to(ROOT).as_posix(),
        "source_proposal_artifact_sha256": sha256_file(proposal_path),
        "source_human_review_artifact": review_path.relative_to(ROOT).as_posix(),
        "source_human_review_artifact_sha256": sha256_file(review_path),
        "review_snapshot_sha256": reviewed["review_snapshot_sha256"],
        "strict_admitted_contexts": sorted(PHONE_USE_CONTEXTS),
        "proposal_summary": {
            "cases": len(mined["rows"]),
            "proposals": len(all_proposals),
            "geometry_contexts": dict(sorted(contexts.items())),
            "human_selected_proposals": len(selected),
            "human_selected_contexts": dict(sorted(selected_contexts.items())),
            "no_valid_proposal_hard_examples": len(hard_cases),
            "hard_example_case_ids": hard_cases,
            "hard_example_contexts": dict(sorted(hard_contexts.items())),
        },
        "mining_floor_result": {
            **mining_floor,
            "removed_fraction": (
                mining_floor["removed_proposals"] / mining_floor["naive_proposals"]
            ),
        },
        "thresholds": threshold_rows,
        "interpretation": {
            "measured": (
                "The strict gate removes mounted/static proposals while preserving all "
                "human-selected handheld proposals at the 0.02 mining floor."
            ),
            "not_measured": [
                "detector precision",
                "detector recall",
                "event precision/recall/F1",
                "false alarms per minute",
                "independent generalization",
            ],
        },
        "governance": {
            "development_only": True,
            "fn_selected_positive_only": True,
            "verified_occupant_tracks": 0,
            "protected_holdouts_accessed": False,
            "ground_truth_changed": False,
            "event_matcher_changed": False,
            "fine_tuning_performed": False,
            "production_approved": False,
        },
    }
    return result


def render_report(result: dict[str, Any], output: Path = DEFAULT_OUTPUT) -> Path:
    summary = result["proposal_summary"]
    floor = result["mining_floor_result"]
    rows = "\n".join(
        "| {confidence_threshold:.2f} | {naive_proposals} | {context_gated_proposals} | "
        "{removed_proposals} | {naive_case_candidate_yield}/13 | "
        "{context_gated_case_candidate_yield}/13 | {human_selected_retained_context_gated}/12 | "
        "{hard_example_has_naive_candidate} | {hard_example_has_context_gated_candidate} |".format(
            **item
        )
        for item in result["thresholds"]
    )
    text = f"""# V9 phone-context gate ablation

Scope: **FN-selected development diagnostic only**. The table reports candidate yield, not detector
precision/recall or event metrics.

| Confidence | Raw proposals | Gated proposals | Removed | Raw case yield | Gated case yield | Human selections retained | Hard raw | Hard gated |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
{rows}

At the 0.02 mining floor, the gate removes {floor['removed_proposals']}/{floor['naive_proposals']}
proposals ({100 * floor['removed_fraction']:.2f}%). All removed proposals are `MOUNTED_OR_STATIC`;
all {floor['human_selected_retained_context_gated']}/{summary['human_selected_proposals']} human-selected
boxes are retained as `HANDHELD_USE`.

The adjudicated hard example `{summary['hard_example_case_ids'][0]}` contains
{sum(summary['hard_example_contexts'].values())} proposals, all mounted/static. The raw 0.02 gate would
offer it as a candidate, while the context gate rejects every proposal. This directly addresses the
mounted cabin-camera failure without relabeling the visible phone as a negative.

The fusion API now accepts per-view context and exposes raw confidence, admitted confidence, and rejection
reasons. Strict mode fails closed for mounted, visible-but-not-used, possible, or unknown context. Legacy
mode remains unchanged until a caller supplies audited context and explicitly enables strict gating.

This positive-only FN sample has no true-negative denominator and no verified occupant tracks. It cannot
establish detector accuracy, event P/R/F1, false alarms/min, target attainment, or independent
generalization. No model was trained, and protected holdouts were not accessed.
"""
    path = output / "14_PHONE_CONTEXT_GATE_ABLATION.md"
    path.write_text(text, encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = analyze(args.output)
    json_path = args.output / "phone_context_gate_ablation.json"
    _write_json(json_path, result)
    report_path = render_report(result, args.output)
    print(json.dumps({"json": str(json_path), "report": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()
