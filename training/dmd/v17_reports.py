"""Render concise V17 reports from machine-readable audit evidence."""

# ruff: noqa: E501

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/v17_ai85_recovery"


def load(name: str) -> dict:
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def save(name: str, body: str) -> None:
    (OUT / name).write_text(body.strip() + "\n", encoding="utf-8")


def render() -> None:
    access = load("dataset_access_action_plan.json")
    inventory = load("approved_data_inventory.json")
    benchmark = load("phone_benchmark_protocol.json")
    split = load("group_split_manifest.json")
    sahi = load("sahi_ablation.json")
    action = load("action_experiments.json")
    association = load("association_fusion_analysis.json")
    temporal = load("temporal_event_analysis.json")
    event = load("event_metrics.json")
    seatbelt = load("seatbelt_review_status.json")

    blocker_rows = "\n".join(
        f"| {item['dataset_id']} | {item['status']} | {', '.join(item['blockers'])} |"
        for item in access["datasets"]
    )
    save("01_DATA_ACCESS_ACTION_PLAN.md", f"""
# Dataset access action plan

V17 re-audited the 28,555-row inventory and separated blockers by cause. Zero rows are approved;
zero new images, subjects, videos or events were added. No permission flag was changed.

| Dataset | Status | Exact blockers |
|---|---|---|
{blocker_rows}

The machine-readable plan gives the official action link, exact permission, operations allowed
after approval, expected Roadwatch contribution and evidence receipt for every priority source.
The next highest-value actions are: obtain 100-Driver written consent; accept State Farm rules
in the user's own account; establish explicit DMD fitting rights; or approve a license-checked
COCO/Open Images subset. Tokens and credentials are not evidence of accepted terms.
""")
    save("02_ANNOTATION_BENCHMARK_AND_SPLIT.md", f"""
# Annotation benchmark and split

`training/dmd/phone_benchmark.py` now rejects incomplete truth. A positive image needs original
dimensions, SHA-256, license, context, human approval and in-bounds pixel boxes. A zero-box image
is accepted only when a human confirms the `ALL_VISIBLE_PHONES` annotation scope. Driver,
passenger and unknown associations remain distinct.

Current benchmark status: **{benchmark['status']}**. Positive images: 0; reviewed negatives: 0;
phone boxes: 0. AP calculation is therefore prohibited.

`training/split_dataset.py` now builds connected components across SHA, near-duplicate cluster,
subject, video, session and original asset before assigning a split. It blocks `gZ-37` and
`gE-28` before materialization and runs a second preflight. Current split status is
**{split['status']}** because approved records are zero; group-disjoint validation is not claimed.
""")
    save("03_SAHI_AND_ACTION_RESULTS.md", f"""
# SAHI and action results

The V15 sliced adapter SHA-256 is `{sahi['code_sha256']}`. Its geometry diagnostic uses six
384×384 tiles at 25% overlap. It measured adapter overhead only and excluded model inference.
D0-D4 AP50/AP50-95, negative-frame FP rate and small-phone recall remain blocked. SAHI has not
demonstrated a Phone AP increase.

No action fitting ran. The inherited development-only E3 metrics remain Macro-F1
**{action['inherited_development_only']['macro_f1']:.2%}**, SAFE recall
**{action['inherited_development_only']['safe_recall']:.2%}**, PHONE recall
**{action['inherited_development_only']['phone_recall']:.2%}** and OTHER recall
**{action['inherited_development_only']['other_recall']:.2%}**. There is still no valid fitted
head/scaler. SAFE recall did not improve in V17.
""")
    save("04_ASSOCIATION_TEMPORAL_AND_SEATBELT.md", f"""
# Association, temporal events and seatbelt

`training/dmd/candidate_flow.py` provides ordered stage decisions with mandatory reason codes,
GT-positive rejection counts, admitted-negative counts and UNKNOWN coverage. Accuracy analysis
is {association['status']} because no authorized cohort has occupant-role, object and event truth
together. NEEDS_REVIEW remains outside confirmed TP metrics.

The candidate temporal configuration requires at least
{temporal['active_candidate_config']['min_positive_seconds']:.2f}s and
{temporal['active_candidate_config']['min_observations']} observations. Evidence spans below that
duration cannot activate. V17 did not tune the threshold or matcher because a valid replay set
does not exist.

Seatbelt remains blocked by {seatbelt['pending']} pending human reviews. No automatic approval or
training occurred. The inherited development Macro-F1 is
**{seatbelt['best_inherited_development_macro_f1']:.2%}** and UNCERTAIN recall is
**{seatbelt['uncertain_recall']:.2%}**.
""")
    save("05_METRICS_AND_TESTS.md", f"""
# Metrics and tests

V17 did not change predictions: TP={event['tp']}, FP={event['fp']}, FN={event['fn']}, precision
={event['precision']:.2%}, recall={event['recall']:.2%}, F1={event['f1']:.2%}, FA/min
={event['false_alarms_per_minute']:.4f}. Deltas are all zero. These are inherited development
metrics, not a new independent evaluation.

Executed test counts and build results are stored in `test_results.json` with raw logs. Runtime
for D0-D4 and the full pipeline is not reported because no valid inference benchmark ran.
""")
    save("06_FINAL_SCIENTIFIC_DECISION.md", f"""
# Final scientific decision

**Production: REJECT. Independent AI85: NOT ESTABLISHED.**

V17 delivers three concrete engineering improvements: leakage-safe component splitting before
materialization, a fail-closed phone positive/negative benchmark contract, and stage-level
candidate-flow instrumentation. They are retained as tested engineering controls, while their
three research experiments remain INCONCLUSIVE because synthetic contract tests do not establish
accuracy.

Approved samples: **{inventory['approved_rows']}**. New valid data: **0**. Group-disjoint locked
test: **not established**. Phone AP improvement: **not measured**. SAFE recall improvement: **0**.
Event TP/FP/FN change: **0/0/0**. Seatbelt improvement: **0**. The current event F1
({event['f1']:.2%}) does not exceed 85%.

The project needs a human authorization receipt for a selected dataset, human-reviewed phone
positive/negative truth, subject/session lineage, and completion of the 500 seatbelt reviews.
Only after those receipts exist can Roadwatch freeze a split and run D0-D4 or fitting without
turning missing evidence into a metric claim.
""")


if __name__ == "__main__":
    render()
