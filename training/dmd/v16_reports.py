"""Render the evidence-backed V16 data-centric research report bundle."""

# ruff: noqa: E501

from __future__ import annotations

import json
from pathlib import Path

from training.common import sha256_file

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/v16_data_ai85"


def load(name: str) -> dict:
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def save(name: str, body: str) -> None:
    (OUT / name).write_text(body.strip() + "\n", encoding="utf-8")


def render() -> None:
    registry = load("dataset_registry.json")
    data = load("data_manifest.json")
    metrics = load("metrics_comparison.json")
    ready = load("candidate_readiness.json")
    event = metrics["phone_event_controlled_29"]
    action = metrics["clip_selected_e3"]["pooled"]
    seatbelt = metrics["seatbelt"]
    rows = registry["registry"]
    status_lines = "\n".join(
        f"| {row['dataset_id']} | {row['access_status']} | "
        f"{', '.join(row['allowed_operations']) or 'none'} | No |"
        for row in rows
    )
    local_lines = "\n".join(
        f"| `{item['manifest']}` | {item['records']} | {item['unique_sha256']} | "
        f"{item['source_groups']} | {item['approved_for_training']} |"
        for item in data["existing_ingest_manifests"]
    )

    save(
        "01_V15_CODE_AUDIT.md",
        f"""
# V15 code and evidence audit

V16 started from local commit `{registry['source_head']}` and preserved the user-owned
`prompt.md` change. The V15 implementation is present in `training/dmd/sliced_phone.py` and is
wired into `AutonomousPhonePipeline.detect_roi`. It contains overlapping tile generation,
coordinate projection back to the source ROI, edge penalties, containment handling and NMS.
No second SAHI adapter was created.

The V15 detector checkpoints remain unchanged. The specialist checkpoint SHA-256 is
`840a29cb2151b881279cdabe25b03b28c5dcf40a43464edd6b672c8851f77d54`; the auxiliary
checkpoint SHA-256 is `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`.
V15 reported 40 targeted and 629 full tests. V16 reruns are recorded in
`test_results.json`, not inferred from that historical claim.

The accuracy limitation remains decisive: V15 has engineering tests and a diagnostic proposal
yield result, but no approved positive-box plus reviewed-negative detector validation set.
Therefore V15 did not establish Phone AP or an event-metric improvement.
""",
    )
    save(
        "02_DATASET_RESEARCH_AND_LICENSES.md",
        """
# Dataset research and licenses

Nine requested sources were checked against official owner pages on 2026-10-18. A public URL
was treated as discovery evidence only. No V16 media download occurred.

- **DMD 2026:** the owner lists RGB real-car data for 14 subjects, OpenLABEL/VCD 5+ labels and
  CC BY-NC-ND 4.0. The project permission record still says `RIGHTS_ACCEPTED=false`; processing
  or fitting permissions were not changed. `gZ-37` and `gE-28` remain protected.
- **100-Driver:** the official page reports more than 470,000 images, 100 drivers, five vehicles
  and four cameras. The official custom form requires prior PIC Lab consent, research-only use
  and no redistribution. Repository code being MIT does not license its data.
- **State Farm:** access remains subject to Kaggle competition rules and account acceptance.
  The existing local registry also rejects it pending legal review. No credential or API token
  was used.
- **Drive&Act:** the official page reports 12 hours, 29 long sequences, five views and 83
  hierarchical activities under research-use conditions. A phone-equivalent label was not
  established, so no automatic PHONE_USE mapping is allowed.
- **COCO 2017:** useful for generic cell-phone boxes, but image-level rights and the selected
  subset still require verification. The full archive was deliberately not downloaded.
- **Open Images V7:** official metadata identifies `/m/050k8` as `Mobile phone`. Verified
  negatives and per-image licenses are required; missing boxes alone do not make a negative.
- **Roboflow candidates:** all three pages display CC BY 4.0, but original-image provenance,
  augmentation lineage and label meaning remain unresolved. The Eco3cc page does not prove a
  consistent exported UNCERTAIN class.

User actions needed before access changes: submit the official 100-Driver license form; accept
the State Farm competition rules through the user's own Kaggle account; provide an authorized
DMD rights record; and complete provenance/label review for Roboflow assets. Credentials must
not be committed or pasted into reports.

Authorization procedure:

1. For 100-Driver, download the official licensing PDF, describe this non-commercial research,
   sign it with the responsible institution/researcher, send it to the PIC Lab contact printed
   on the form, and retain their written approval reference. Only then may a human set the local
   registry to `DOWNLOAD_AUTHORIZED` with `allowed_operations: [download]`.
2. For State Farm, the user must sign into Kaggle and accept the competition rules personally.
   Record the accepted rules/version and permitted operations; store the token only in the
   environment. Do not put a token in YAML, a command transcript, Git or this report.
3. For DMD, an authorized human must update the separate project permission record. Dataset
   access from another source cannot alter `RIGHTS_ACCEPTED` or authorize model fitting.
4. For each Roboflow candidate, sample original assets, trace their origin/license and inspect
   exact exported classes before allowing ingestion beyond quarantine. Public project licensing
   alone does not approve the assets for Roadwatch training.
""",
    )
    save(
        "03_DATASET_SELECTION_MATRIX.md",
        f"""
# Dataset selection matrix

| Dataset | V16 status | Allowed now | Download/train selected |
|---|---|---|---|
{status_lines}

Selection result: zero datasets passed the combined rights, provenance, approved-label and
group-split gates. V16 performed {registry['v16_downloads_performed']} downloads and selected
none for fitting. This is a fail-closed decision, not a claim that the sources lack scientific
value.
""",
    )
    save(
        "04_DATA_INGESTION_AND_QA.md",
        f"""
# Data ingestion and QA

V16 reused `training/download_source.py`, `training/ingest_dataset.py` and
`training/split_dataset.py`. CLI defaults now point at the repository's actual `dataset/`
layout. The downloader now requires the exact `DOWNLOAD_AUTHORIZED` registry state, a verified
license and an explicit `download` operation before any network function is called.

| Existing manifest | Rows | Unique SHA-256 | Groups | Approved |
|---|---:|---:|---:|---:|
{local_lines}

The inventory contains **{data['total_records']}** records and
**{data['total_approved_for_training']}** approved training records. V16 added zero original
images, subjects, videos and events. Existing manifests retain exact SHA-256, perceptual hashes,
source-group IDs, decode checks, raw labels and pending review states. Raw media was not
committed, overwritten or copied into this report bundle.
""",
    )
    save(
        "05_ANNOTATION_AND_SPLIT_PROTOCOL.md",
        """
# Annotation and split protocol

`training/data_governance.py` defines the requested phone visibility, phone action, context,
occupant-role and three-state seatbelt vocabularies. Every canonical annotation retains source
ID, asset ID, source hash, original label, mapping version, review state and license reference.
Video intervals require a real video ID and valid start/end times; static images cannot acquire
synthetic event timestamps. Positive PHONE_BOX rows require valid normalized coordinates.

Approval requires an authenticated reviewer. UNKNOWN is never converted to SAFE_DRIVING,
visible phone evidence is not automatically phone use, and invisible belt evidence remains
UNCERTAIN_OR_OCCLUDED. Human QA requires double annotation on a sampled subset, disagreement
adjudication and immutable review receipts.

The split validator checks exact content, source group, effective group, subject, video and
original-asset lineage across splits. Augmentations must follow their original asset. `gZ-37`
and `gE-28` are forbidden in train or validation. Because there are zero approved governed
records, no train/validation/locked-test split is claimed as established.

Planning target: group-disjoint data from at least 20 new consenting participants when feasible,
with roughly 100 independently held-out phone events plus enough reviewed negative driving time
to estimate false alarms/minute. These are collection targets, not current sample counts.
""",
    )
    save(
        "06_PHONE_DETECTION_RESULTS.md",
        """
# Phone detection results

The V15 tiled implementation was verified at code and regression-test level: tile bounds,
overlap, source-coordinate mapping, edge handling, duplicate suppression, ROI clipping and
unchanged checkpoint identity are covered. D0-D4 accuracy ablations were not run because there
is no approved validation set containing human-checked phone boxes and reviewed in-scope
negative frames.

Phone AP50, AP50-95, small-object recall and negative-frame FP rate remain **not measured**.
The 13 historical FN-selected frames were not reused as the principal detector validation set.
Proposal yield was not reported as recall. No detector was promoted and no active weights or
thresholds changed.
""",
    )
    save(
        "07_ACTION_RECOGNITION_RESULTS.md",
        f"""
# Action recognition results

No A0-A6 fitting was authorized. The reproducible development-only E3 result therefore remains:
clip Macro-F1 **{action['macro_f1']:.2%}**, balanced accuracy **{action['balanced_accuracy']:.2%}**,
SAFE recall **{action['safe_recall']:.2%}**, PHONE recall **{action['phone_recall']:.2%}** and
OTHER recall **{action['other_recall']:.2%}** on 2,625 samples from the repeatedly used subjects
`gB-9`, `gC-14` and `gZ-36`.

These results are development-only and do not demonstrate cross-dataset or independent-subject
generalization. No synthetic fitted model, scaler or claimed accuracy was created.
""",
    )
    save(
        "08_ASSOCIATION_FUSION_ANALYSIS.md",
        """
# Association and fusion analysis

The existing driver-phone association, autonomous phone, online candidate, fusion and event
modules were retained. A new candidate-flow accuracy run would require verified occupant role,
phone-object truth and temporal event truth on the same authorized samples. Those conditions
are absent, so counts at Detected → Candidate → Driver-associated → Handheld-evidence →
Action-supported → Fusion-admitted → Temporal-activated → Event-matched were not fabricated.

The V15 context replay remains historical development evidence: it removed one FP without
recovering a TP on a two-subject slice. It is not an independent validation result. Ambiguous
driver/passenger or mounted-device evidence must enter a review state rather than be counted as
a confirmed TP.
""",
    )
    save(
        "09_TEMPORAL_EVENT_RESULTS.md",
        f"""
# Temporal event results

No new temporal candidate was admitted, so the frozen controlled 29-event development result is
unchanged: TP={event['true_positives']}, FP={event['false_positives']}, FN={event['false_negatives']},
precision={event['precision']:.2%}, recall={event['recall']:.2%}, F1={event['f1']:.2%}, and
FA/min={event['false_alarms_per_minute']:.4f} across {event['evaluated_minutes']:.4f} minutes.

V16 recovered 0 TP, lost 0 TP, added 0 FP and removed 0 FP because it did not alter predictions.
Duration-bin recall, onset error and emission latency were not recomputed without new real event
annotations. The matcher, ground truth and denominators were unchanged.
""",
    )
    save(
        "10_SEATBELT_RESULTS.md",
        f"""
# Seatbelt results

No seatbelt training was run. All 500 queued candidates still require authenticated human
review, so they cannot become training truth automatically. The best existing development-only
Macro-F1 remains **{seatbelt['best_development_macro_f1']:.2%}** (S2). Its UNCERTAIN recall is
**{seatbelt['variants'][2]['uncertain_recall']:.2%}**, false-unfastened rate is
**{seatbelt['variants'][2]['false_unfastened_rate']:.2%}**, and abstention rate is
**{seatbelt['variants'][2]['abstention_rate']:.2%}**.

The three-state policy remains FASTENED, UNFASTENED and UNCERTAIN_OR_OCCLUDED. Occluded belt
evidence is not converted to UNFASTENED, and pending samples stay in the denominator of any
future coverage/abstention report.
""",
    )
    save(
        "11_END_TO_END_METRICS.md",
        f"""
# End-to-end metrics

| Metric | V15 verified development baseline | V16 | Change |
|---|---:|---:|---:|
| Phone event precision | {event['precision']:.2%} | {event['precision']:.2%} | 0 |
| Phone event recall | {event['recall']:.2%} | {event['recall']:.2%} | 0 |
| Phone event F1 | {event['f1']:.2%} | {event['f1']:.2%} | 0 |
| False alarms/min | {event['false_alarms_per_minute']:.4f} | {event['false_alarms_per_minute']:.4f} | 0 |
| SAFE clip recall | {action['safe_recall']:.2%} | {action['safe_recall']:.2%} | 0 |
| Seatbelt Macro-F1 | {seatbelt['best_development_macro_f1']:.2%} | {seatbelt['best_development_macro_f1']:.2%} | 0 |
| Phone AP | not established | not established | n/a |

There is no independent holdout result. Classification and event metrics remain separate. V16
did not select thresholds on a locked test or change production behavior.
""",
    )
    save(
        "12_RUNTIME_AND_TESTS.md",
        """
# Runtime and tests

Machine-readable command results are in `test_results.json`; raw output is retained in the
adjacent `.log` files. `reproduce.ps1` reruns the audit, report renderer, compilation, Ruff,
targeted tests, full tests, backend smoke, frontend build and bundle verification.

No model inference or fitting ran in V16, so a new detector p50/p95, RAM, VRAM or training
throughput number does not exist. Historical runtime is not relabeled as a V16 benchmark.
The V16 engineering change is exercised without any network access by the fail-closed downloader
test and by annotation/split validation tests.
""",
    )
    save(
        "13_FINAL_AI_LEAD_DECISION.md",
        f"""
# Final AI lead decision

**Decision: REJECT production promotion and reject an AI85 claim.**

V16 verified all nine requested dataset candidates and strengthened the acquisition and
annotation boundary, but zero sources currently satisfy all rights, provenance, human-label and
group-split requirements. The existing inventory has {data['total_records']} rows and zero
approved training rows. No V16 media was downloaded, no protected subject was accessed, no
model was fitted, and no active checkpoint or event logic changed.

The tiled detector is technically implemented but has not demonstrated an AP improvement. Phone
event recall/F1, SAFE recall, seatbelt Macro-F1 and FA/min are unchanged. Independent AI85 is
not established.

Next admissible work is: obtain written access for a selected source; record license and allowed
operations in the registry; review provenance; annotate real positive boxes and scoped negatives;
double-review a QA subset; build a subject/session/original-asset-disjoint split; lock the test;
then run D0-D4 before any fitting. Seatbelt work additionally needs completion of the 500 pending
human reviews. Production approval remains a separate human decision.

Readiness snapshot: `{json.dumps(ready, sort_keys=True)}`
""",
    )

    # The report renderer hashes its own evidence inputs; the verifier adds the final bundle hash.
    evidence = {
        name: sha256_file(OUT / name)
        for name in (
            "dataset_registry.json",
            "license_audit.json",
            "data_manifest.json",
            "label_mapping.json",
            "annotation_qa.json",
            "group_split_manifest.json",
            "experiment_registry.json",
            "metrics_comparison.json",
            "model_manifest.json",
            "candidate_readiness.json",
        )
    }
    (OUT / "report_evidence_hashes.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    render()
