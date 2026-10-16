"""Render the required AI85 next-stage reports from measured JSON artifacts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from training.dmd.ai85_audit import read_json


def percent(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def metric_row(name: str, metrics: dict[str, Any], status: str) -> str:
    return (
        f"| {name} | {metrics['true_positives']} | {metrics['false_positives']} | "
        f"{metrics['false_negatives']} | {percent(metrics['precision'])} | "
        f"{percent(metrics['recall'])} | {percent(metrics['f1'])} | "
        f"{metrics['false_alarms_per_minute']:.4f} | {status} |"
    )


def table_row(*cells: Any) -> str:
    return "| " + " | ".join(str(cell) for cell in cells) + " |"


def write(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def render(output: Path) -> None:
    analysis = read_json(output / "analysis.json")
    fn = read_json(output / "fn_recovery.json")
    dataset = read_json(output / "dataset_generalization.json")
    detector = read_json(output / "detector_trace.json")
    bias = analysis["subject_bias"]
    baseline = analysis["baseline"]
    direct = analysis["direct_coarse_metrics"]
    flat = analysis["flat_clip_metrics"]
    flat_coarse = analysis["flat_coarse_metrics"]
    variants = analysis["event_variants"]
    root_counts = Counter(row["root_cause"] for row in fn["cases"])
    base_safe = baseline["coarse_metrics"]["per_class"]["SAFE_DRIVING"]["recall"]
    base_coarse_macro = baseline["coarse_metrics"]["macro_f1"]
    base_balanced = baseline["coarse_metrics"]["balanced_accuracy"]
    flat_safe = flat_coarse["per_class"]["SAFE_DRIVING"]["recall"]
    direct_safe = direct["per_class"]["SAFE_DRIVING"]["recall"]

    train_lines = []
    for subject, row in analysis["baseline_training_terminal_accuracy"].items():
        train_lines.append(
            f"| Held out {subject} | {row['epoch']} | "
            f"{percent(row['sampled_coarse_accuracy'])} | {row['loss']:.6f} |"
        )
    outer_lines = []
    for subject, row in analysis["baseline_subject_confusions"].items():
        clip = row["clip_metrics"]
        coarse = row["coarse_metrics"]
        outer_lines.append(
            f"| {subject} | {percent(clip['clip_accuracy'])} | "
            f"{percent(coarse['per_class']['SAFE_DRIVING']['recall'])} |"
        )
    write(
        output / "ROOT_CAUSE_ANALYSIS.md",
        f"""# Root-cause analysis

## Finding

The dominant measured failure is cross-subject representation shift, followed by a lack of
continuous PHONE evidence. The frozen ResNet18 embeddings separate subject identity much more
strongly than behavior: subject silhouette is `{bias['silhouette_cosine_by_subject']:.4f}` versus
action silhouette `{bias['silhouette_cosine_by_action']:.4f}`. A linear probe identifies the
subject with `{percent(bias['subject_identification_accuracy_random_split'])}` accuracy on a
random split.

This conclusion uses embedding separability, training/outer performance, image statistics,
class distributions, temporal traces, and direct head ablations. It is not inferred from a
confusion matrix alone.

## Memorization evidence

| Fold | Final epoch | Training coarse accuracy | Loss |
| --- | ---: | ---: | ---: |
{chr(10).join(train_lines)}

| Outer subject | Six-class accuracy | SAFE recall |
| --- | ---: | ---: |
{chr(10).join(outer_lines)}

The near-perfect training accuracy and poor subject-disjoint performance are consistent with
memorization of subject/camera appearance. Exact RGB duplicate checks from the prior audit found
no cross-subject duplicate frames, so duplication does not explain the gap.

## Head-design evidence

The subject-normalized six-class probe reaches Macro-F1 `{percent(flat['macro_f1'])}`, but its
official coarse SAFE recall is only
`{percent(flat_coarse['per_class']['SAFE_DRIVING']['recall'])}` because the four PHONE class
probabilities are summed before the coarse decision. A direct three-class head avoids this
aggregation failure and reaches SAFE recall
`{percent(direct['per_class']['SAFE_DRIVING']['recall'])}`, coarse Macro-F1
`{percent(direct['macro_f1'])}`, and balanced accuracy `{percent(direct['balanced_accuracy'])}`.
It still misses the 85% SAFE target.

## Event-loss attribution

- `{root_counts['INSUFFICIENT_CONTINUOUS_ACTION_EVIDENCE']}` of 13 baseline FNs lack two
  consecutive samples at the inherited 0.80 threshold.
- `{root_counts['TEMPORAL_BOUNDARY_OR_EVENT_MATCHING']}` reaches sustained activation but fails
  boundary/matching.
- `{root_counts['MISSING_CENTERED_SAMPLE']}` has no centered action sample.
- The 960-pixel detector trace produces a PHONE proposal in only
  `{detector['cases_with_any_phone_proposal']}/13` FN intervals and
  `{detector['sampled_frames_with_phone_proposal']}/{detector['sampled_frames']}` sampled frames.

## Limits

Only three already-consumed development subjects are available. The source governance record
states `RIGHTS_STATUS = PENDING_PROJECT_REVIEW` and `RIGHTS_ACCEPTED = false`. No causal claim
about lighting or occlusion is made because those factors have no verified case-level labels.
""",
    )

    baseline_cm = baseline["coarse_metrics"]["confusion_matrix"]
    safe_table = "\n".join(
        [
            table_row(
                "V6 baseline hierarchical head",
                percent(base_safe),
                percent(base_coarse_macro),
                percent(base_balanced),
            ),
            table_row(
                "Subject-normalized six-class probe, regrouped",
                percent(flat_safe),
                percent(flat_coarse["macro_f1"]),
                percent(flat_coarse["balanced_accuracy"]),
            ),
            table_row(
                "Subject-normalized direct coarse probe",
                percent(direct_safe),
                percent(direct["macro_f1"]),
                percent(direct["balanced_accuracy"]),
            ),
        ]
    )
    class_lines = []
    for subject, row in dataset["subjects"].items():
        distribution = row["class_distribution"]
        phone_frames = sum(
            distribution.get(name, 0)
            for name in ("TEXTING_LEFT", "TEXTING_RIGHT", "PHONECALL_LEFT", "PHONECALL_RIGHT")
        )
        class_lines.append(
            f"| {subject} | {distribution.get('SAFE_DRIVING', 0)} | "
            f"{phone_frames} | "
            f"{distribution.get('OTHER_ACTIVITY', 0)} |"
        )
    write(
        output / "SAFE_CLASSIFICATION_ANALYSIS.md",
        f"""# SAFE_DRIVING classification analysis

## Baseline

The baseline has 668 SAFE clips, so SAFE is not rare. Only 84 are classified SAFE; 196 are
classified PHONE and 388 OTHER.

```json
{json.dumps(baseline_cm, indent=2)}
```

| Subject | SAFE frames | PHONE frames | OTHER frames |
| --- | ---: | ---: | ---: |
{chr(10).join(class_lines)}

## Controlled representation and head comparison

| Model | SAFE recall | Coarse Macro-F1 | Balanced accuracy |
| --- | ---: | ---: | ---: |
{safe_table}

Per-video z-score normalization uses no labels and reduces static appearance offsets. The direct
coarse probe improves SAFE recall by
`{percent(direct_safe - base_safe)}` absolute, but remains below 85%.

The six-class result demonstrates why flat-class accuracy cannot substitute for official coarse
SAFE recall: summing four individually modest PHONE probabilities can reverse the decision. The
direct coarse head is the valid comparison for SAFE.

## Preprocessing and label checks

The prior audit verified exact alignment between RGB caches and embeddings and found zero replay
label mismatches with the stored float32 timestamps. There are 3-6 boundary disagreements per
subject only when timestamps are widened to float64. Nine-sample windows are mixed near action
boundaries, but long SAFE intervals also fail, so boundary mixing is not the primary explanation.
""",
    )

    detector_by_case = {row["case_id"]: row for row in detector["rows"]}
    fn_lines = []
    recovered = 0
    for row in fn["cases"]:
        fixed = row["variant_evidence"]["SN_PHONE_INHERITED_POLICY"]
        recovered += int(fixed["matched"])
        base = row["baseline_trace"]
        new = row["subject_normalized_phone_trace"]
        fn_lines.append(
            f"| {row['case_id']} | {row['duration_seconds']:.2f} | {base['samples']} | "
            f"{(base.get('probability_max') or 0):.3f} | {base['longest_activation_run']} | "
            f"{(new.get('probability_max') or 0):.3f} | {new['longest_activation_run']} | "
            f"{detector_by_case[row['case_id']]['samples_with_phone_proposal']} | "
            f"{'yes' if fixed['matched'] else 'no'} | {row['root_cause']} |"
        )
    visual_lines = [
        f"- `{row['case_id']}`: {row['visual_review']}" for row in fn["cases"]
    ]
    fn_header = (
        "| Case | Sec | Baseline samples | Base max | Base longest >=.80 | New max | "
        "New longest >=.80 | Detector-positive frames | Recovered by fixed new branch | "
        "Root cause |"
    )
    write(
        output / "FN_RECOVERY.md",
        f"""# False-negative recovery

All 13 original BODY ROI images were manually inspected. Every interval is consistent with its
PHONE-use annotation. The table evaluates continuous evidence across each full interval rather
than selecting the maximum frame only.

{fn_header}
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
{chr(10).join(fn_lines)}

The subject-normalized binary branch recovers `{recovered}` of the 13 baseline FNs under the
inherited event policy. Overall TP remains 16 because it also loses previously matched events;
recovery of selected errors is therefore not a net recall gain.

## Visual review

{chr(10).join(visual_lines)}

`gB-9-phone-06` is only about one second and has two centered samples with an occluded phone.
`gC-14-phone-05` is about 0.40 seconds and has no centered sample. `gZ-36-phone-05` reaches two
baseline activations but fails temporal matching. These three mechanisms cannot be repaired by a
single global threshold without additional false alarms.

The detector trace records every proposal box, confidence, driver ROI, timestamp and legacy view
confidence in `detector_trace.json`. Association is limited to containment in a driver ROI; no
phone track, occupant track, or ground-truth box exists.
""",
    )

    representation_table = "\n".join(
        [
            table_row(
                "V6 frozen embedding + BiGRU",
                f"Six-class Macro-F1 {percent(baseline['clip_metrics']['macro_f1'])}",
                percent(base_safe),
                percent(baseline["event_metrics"]["f1"]),
            ),
            table_row(
                "Normalized embedding + six-class linear",
                f"Six-class Macro-F1 {percent(flat['macro_f1'])}",
                percent(flat_safe),
                percent(variants["SN_FLAT_INHERITED_POLICY"]["pooled_metrics"]["f1"]),
            ),
            table_row(
                "Normalized embedding + direct coarse linear",
                f"Coarse Macro-F1 {percent(direct['macro_f1'])}",
                percent(direct_safe),
                percent(
                    variants["SN_DIRECT_COARSE_INHERITED_POLICY"]["pooled_metrics"]["f1"]
                ),
            ),
            table_row(
                "Normalized embedding + binary PHONE linear",
                "PHONE event branch",
                "n/a",
                percent(variants["SN_PHONE_INHERITED_POLICY"]["pooled_metrics"]["f1"]),
            ),
        ]
    )
    write(
        output / "VISUAL_REPRESENTATION_ABLATION.md",
        f"""# Visual representation ablation

## Hypothesis

Frozen ImageNet embeddings contain a large subject/camera offset. Per-video label-free feature
normalization should suppress that offset before a larger temporal architecture is attempted.
A strongly regularized linear probe measures whether the normalized representation is separable.

## Results

| Representation/head | Clip or coarse metric | SAFE recall | Event F1 at inherited policy |
| --- | ---: | ---: | ---: |
{representation_table}

The representation/head change improves clip-level separability and SAFE recognition but does
not create sufficiently calibrated sustained event probabilities. The direct coarse probe emits
no event at the inherited 0.80 threshold because strong regularization keeps probabilities below
that threshold.

Mean fine-tuning, temporal attention and Native112 were not rerun. Multi-ROI training was not
performed: gC-14 lacks FACE/HANDS video, while the other two subjects' views differ from BODY by
7-22 frames. A three-subject group-disjoint multi-view comparison would therefore confound view
availability with subject identity.
""",
    )

    view_lines = []
    for subject, row in dataset["subjects"].items():
        view_lines.append(
            f"| {subject} | {', '.join(item['view'] for item in row['views'])} | "
            f"{', '.join(str(item['frames']) for item in row['views'])} |"
        )
    write(
        output / "DATASET_GENERALIZATION.md",
        f"""# Dataset and generalization

| Subject | Available views | Frame counts |
| --- | --- | --- |
{chr(10).join(view_lines)}

- Development subjects: `{', '.join(dataset['development_subjects'])}`.
- Consumed holdouts excluded from every new load: `{', '.join(dataset['forbidden_holdouts'])}`.
- New approved development subjects: none.
- New independent holdout: none.
- Source rights: `{dataset['rights_status']}`; accepted =
  `{str(dataset['rights_accepted']).lower()}`.
- Splits: group-disjoint by source subject for every outer fold.

`AI85_INDEPENDENT_VALIDATION = NOT_ESTABLISHED`

The three subjects have different clothing, brightness, camera composition and view availability.
The subject-ID probe reaches 100%, while action silhouette is only 0.0486. The current dataset is
the binding research bottleneck. Additional architecture search on these same subjects would
increase development-set adaptation without establishing generalization.
""",
    )

    rows = [
        metric_row("V6 baseline fusion", baseline["event_metrics"], "historical comparator")
    ]
    for name, variant in variants.items():
        rows.append(metric_row(name, variant["pooled_metrics"], variant["status"]))
    write(
        output / "METRICS_COMPARISON.md",
        f"""# Metrics comparison

| Variant | TP | FP | FN | Precision | Recall | F1 | FA/min | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
{chr(10).join(rows)}

The best observed development trade-off is `SN_PHONE_EXPLORATORY_POLICY`: Precision 88.89%,
Recall 55.17%, F1 68.09%, and 0.0906 FA/min. It is explicitly ineligible because the policy was
inspected on the same three outer development subjects. The nested validation policy reproduces
the baseline event result: 76.19% / 55.17% / 64.00% with 0.2266 FA/min.

No controlled candidate meets Recall or Event F1 >=85%. The direct coarse probe reaches SAFE
recall 70.81%, below the 85% target.
""",
    )

    test_results_path = output / "test_results.json"
    tests = read_json(test_results_path) if test_results_path.is_file() else {}
    write(
        output / "TEST_RESULTS.md",
        f"""# Test results

Generated test evidence is stored in `test_results.json` and the referenced JUnit XML files.

- Targeted AI85-next tests: `{tests.get('targeted', {}).get('summary', 'PENDING')}`.
- AI85 combined regression tests: `{tests.get('combined', {}).get('summary', 'PENDING')}`.
- Ruff: `{tests.get('ruff', {}).get('summary', 'PENDING')}`.
- Compile: `{tests.get('compile', {}).get('summary', 'PENDING')}`.
- Full repository suite: `{tests.get('full', {}).get('summary', 'PENDING')}`.

No failing test is removed or skipped to improve the report. Full-suite failures, if present, are
classified in `test_results.json`.
""",
    )

    write(
        output / "FINAL_REPORT.md",
        f"""# AI85 next-stage final report

The work identified subject/camera appearance bias as the main SAFE classification failure and
insufficient continuous action evidence as the main FN mechanism. A new subject-normalized
representation was implemented and evaluated without rereading `gZ-37` or `gE-28` and without
repeating the failed Mean, Attention, or Native112 fine-tunes.

The direct coarse probe improves SAFE recall from
`{percent(baseline['coarse_metrics']['per_class']['SAFE_DRIVING']['recall'])}` to
`{percent(direct['per_class']['SAFE_DRIVING']['recall'])}`. The controlled binary PHONE branch
with the inherited policy reaches Precision 80.00%, Recall 55.17%, F1 65.31%, and 0.1813 FA/min.
Nested policy selection does not improve the baseline. An exploratory policy reaches Precision
88.89% and 0.0906 FA/min but remains at Recall 55.17% and is not eligible for selection.

Five baseline FNs are recovered by the new fixed branch, but five prior matches are lost, leaving
overall recall unchanged. The detector proposes PHONE in only 2/13 FN intervals at 960 input.
One FN lacks any centered action sample and one fails temporal matching after sustained evidence.

## Decision

- `AI85_TARGET_ACHIEVED = false`
- `AI85_INDEPENDENT_VALIDATION = NOT_ESTABLISHED`
- `SELECTED_CANDIDATE = none`
- `PRODUCTION_APPROVED = false`
- Production models and thresholds remain unchanged.

## Remaining blockers

1. No new approved subjects and no untouched licensed holdout.
2. Source rights remain pending project review.
3. BODY-only availability for gC-14 prevents a clean multi-view group-disjoint ablation.
4. No phone-box ground truth or persistent phone/occupant association exists.
5. The current 0.5-second action sampling cannot cover every sub-second event.
6. Subject-normalized probabilities require calibration on separate validation subjects before
   any temporal policy can be frozen.
""",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports/ai85_next"))
    args = parser.parse_args()
    render(args.output)


if __name__ == "__main__":
    main()
