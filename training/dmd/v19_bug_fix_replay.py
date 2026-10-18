"""ROADWATCH V19 — Controlled Event Replay R0-R5 Runner.

Evaluates the exact event-level impact of V18 bug fixes:
- E1: Phone context normalization (HANDHELD / HANDHELD_USE / HANDHELD_PHONE_USE)
- E2: Temporal engine state preservation across unadmitted frames
- R4: Integration of original-resolution / auxiliary phone proposals from mining
- R5: Engineering candidate with tuned temporal hysteresis

Conducted on the development cohort (gC-14 and gZ-36, 20 GT events, 15.27 min),
with historical reconciliation against the 3-subject 29-event baseline (including gB-9).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from backend.ai.events import Observation, TemporalEventEngine
from training.dmd.adapter import parse_dmd_openlabel
from training.dmd.temporal_eval import (
    TemporalMatchingPolicy,
    match_temporal_events,
)

ROOT = Path(__file__).resolve().parents[2]

REPLAY_INPUTS = (
    ("gC-14", ROOT / "reports" / "cache_observations_sub_14.json"),
    ("gZ-36", ROOT / "reports" / "cache_observations_sub_36.json"),
)

PROPOSALS_PATH = ROOT / "reports" / "v9_research" / "phone_proposal_mining.json"
CONFIG_PATH = ROOT / "models" / "temporal_config_v2.json"
POLICY_PATH = ROOT / "reports" / "DMD_EVENT_MATCHING_POLICY_FREEZE.json"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_stream(
    observations: list[Observation],
    gt_intervals: list[dict[str, Any]],
    total_seconds: float,
    temporal_config: dict[str, Any],
    matching_policy: TemporalMatchingPolicy,
    reset_on_unadmitted: bool = False,
    e1_fix: bool = True,
    proposal_injections: dict[float, float] | None = None,
) -> dict[str, Any]:
    """Run temporal engine with controlled bug conditions."""
    supported = {
        "window_seconds",
        "min_positive_seconds",
        "min_observations",
        "cooldown_seconds",
        "smoothing_alpha",
        "feature_positive_score",
        "positive_ratio",
        "gap_tolerance_seconds",
        "candidate_threshold",
        "activation_threshold",
        "release_threshold",
        "occlusion_bridge_seconds",
    }
    engine_kwargs = {k: v for k, v in temporal_config.items() if k in supported}
    engine = TemporalEventEngine(**engine_kwargs)

    HANDHELD_CONTEXTS = (
        frozenset({"HANDHELD", "HANDHELD_USE", "HANDHELD_PHONE_USE"})
        if e1_fix
        else frozenset({"HANDHELD_USE"})
    )

    generated_candidates = []

    for obs in observations:
        # Check proposal injection for R4/R5
        conf = obs.confidence
        ctx = obs.phone_context

        if proposal_injections:
            # Check if there is an auxiliary proposal near this timestamp (within 0.5s)
            for p_time, p_conf in proposal_injections.items():
                if abs(obs.timestamp - p_time) <= 0.5:
                    if p_conf > conf:
                        conf = p_conf
                        ctx = "HANDHELD_USE"
                    break

        admitted = (conf > 0.0) and (ctx in HANDHELD_CONTEXTS)

        if not admitted:
            if reset_on_unadmitted:
                # E2 bug: recreates engine on unadmitted frame
                engine = TemporalEventEngine(**engine_kwargs)
            # Empty / unadmitted observation fed
            empty_obs = Observation(
                timestamp=obs.timestamp,
                class_name="phone",
                confidence=0.0,
                track_id=obs.track_id,
                occupant_role="driver",
                vehicle_context_id=obs.vehicle_context_id,
                phone_context="UNKNOWN",
                fusion_score=0.0,
                phone_hand_proximity=0.0,
                phone_face_proximity=0.0,
                pose_confidence=0.0,
                occupant_role_confidence=1.0,
                vehicle_context_confidence=1.0,
            )
            evs = engine.add(empty_obs)
        else:
            effective_obs = Observation(
                timestamp=obs.timestamp,
                class_name="phone",
                confidence=conf,
                track_id=obs.track_id,
                occupant_role="driver",
                vehicle_context_id=obs.vehicle_context_id,
                phone_context=ctx,
                fusion_score=conf,
                phone_hand_proximity=obs.phone_hand_proximity,
                phone_face_proximity=obs.phone_face_proximity,
                pose_confidence=obs.pose_confidence,
                occupant_role_confidence=1.0,
                vehicle_context_confidence=1.0,
            )
            evs = engine.add(effective_obs)

        for ev in evs:
            if ev.event_type == "PHONE":
                generated_candidates.append({
                    "event_type": ev.event_type,
                    "start_seconds": ev.start_timestamp,
                    "end_seconds": ev.end_timestamp,
                    "confidence": ev.confidence,
                    "duration": round(ev.end_timestamp - ev.start_timestamp, 2),
                })

    # Deduplicate candidates across cooldown
    deduped_preds = []
    cooldown = float(temporal_config.get("cooldown_seconds", 3.0))
    for cand in generated_candidates:
        if not deduped_preds:
            deduped_preds.append(dict(cand))
        else:
            last = deduped_preds[-1]
            if cand["start_seconds"] <= last["end_seconds"] + cooldown:
                if cand["end_seconds"] > last["end_seconds"]:
                    last["end_seconds"] = cand["end_seconds"]
                    last["duration"] = round(last["end_seconds"] - last["start_seconds"], 2)
            else:
                deduped_preds.append(dict(cand))

    matched, unmatched_preds, unmatched_gts = match_temporal_events(
        gt_intervals=gt_intervals,
        predicted_events=deduped_preds,
        policy=matching_policy,
    )

    matched_gt_set = set()
    for m in matched:
        gt_item = m["ground_truth"]
        matched_gt_set.add((float(gt_item["start_seconds"]), float(gt_item["end_seconds"])))

    tp = len(matched_gt_set)
    fn = len(unmatched_gts)
    total_gt = len(gt_intervals)
    duplicate_preds = max(0, len(matched) - tp)
    fp = len(unmatched_preds) + duplicate_preds

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / total_gt if total_gt > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    eval_mins = total_seconds / 60.0 if total_seconds > 0 else 1.0
    fa_per_min = fp / eval_mins

    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "total_gt_events": total_gt,
        "evaluated_minutes": round(eval_mins, 2),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "false_alarms_per_minute": round(fa_per_min, 4),
        "matched_gt_count": tp,
        "unmatched_pred_count": len(unmatched_preds),
        "unmatched_gt_count": fn,
        "matched_events": matched,
    }


def run_replays() -> dict[str, Any]:
    temporal_cfg = _read_json(CONFIG_PATH)
    policy_data = _read_json(POLICY_PATH)
    policy = TemporalMatchingPolicy(
        iou_threshold=float(policy_data["iou_threshold"]),
        onset_tolerance_seconds=float(policy_data["onset_tolerance_seconds"]),
        offset_tolerance_seconds=float(policy_data["offset_tolerance_seconds"]),
    )

    # Load proposals mining for R4/R5
    proposals_data = _read_json(PROPOSALS_PATH)
    proposals_by_subject: dict[str, dict[float, float]] = {"gC-14": {}, "gZ-36": {}}
    for row in proposals_data.get("rows", []):
        sub = row.get("subject_id")
        if sub in proposals_by_subject:
            for s in row.get("samples", []):
                t = float(s["timestamp_seconds"])
                for p in s.get("proposals", []):
                    conf = float(p.get("confidence", 0.0))
                    if conf > proposals_by_subject[sub].get(t, 0.0):
                        proposals_by_subject[sub][t] = conf

    # Load cached subject streams and GT
    subject_streams = {}
    for sub, cache_path in REPLAY_INPUTS:
        cache = _read_json(cache_path)
        obs = [Observation(**item) for item in cache["observations"]]
        diag = cache["diagnostics"]
        sub_dir = ROOT / "dataset" / "external_dmd" / "subjects" / sub
        ann_paths = sorted(sub_dir.glob("*_ann_distraction.json"))
        ann = parse_dmd_openlabel(
            ann_paths[0],
            fps=float(diag["fps"]),
            total_frames=int(diag["total_frames"]),
        )
        gt = [i.to_dict() for i in ann.phone_intervals]
        subject_streams[sub] = {
            "observations": obs,
            "gt": gt,
            "total_seconds": float(diag["total_seconds"]),
        }

    # Define R0 - R5 configurations
    # R0: Pre-V18 (E1 bug active, E2 bug active)
    # R1: Only E1 fix enabled (E1 fix=True, E2 bug active=True)
    # R2: Only E2 fix enabled (E1 fix=False, E2 bug active=False)
    # R3: E1 + E2 fixes enabled (E1 fix=True, E2 bug active=False)
    # R4: E1 + E2 + original-resolution proposals injected
    # R5: Engineering candidate (E1 + E2 + proposals + tuned min_positive_seconds=0.8)

    experiments = {
        "R0": {
            "name": "Pre-V18 Frozen Baseline (E1 bug + E2 bug active)",
            "e1_fix": False,
            "reset_on_unadmitted": True,
            "proposals": False,
            "temporal_config": temporal_cfg,
        },
        "R1": {
            "name": "Only E1 Fix Enabled (Context Normalized, E2 bug active)",
            "e1_fix": True,
            "reset_on_unadmitted": True,
            "proposals": False,
            "temporal_config": temporal_cfg,
        },
        "R2": {
            "name": "Only E2 Fix Enabled (Strict Context, E2 Engine Preserved)",
            "e1_fix": False,
            "reset_on_unadmitted": False,
            "proposals": False,
            "temporal_config": temporal_cfg,
        },
        "R3": {
            "name": "E1 + E2 Fixes Enabled (Normalized Context + Preserved Engine)",
            "e1_fix": True,
            "reset_on_unadmitted": False,
            "proposals": False,
            "temporal_config": temporal_cfg,
        },
        "R4": {
            "name": "E1 + E2 + Original-Resolution Phone Proposals",
            "e1_fix": True,
            "reset_on_unadmitted": False,
            "proposals": True,
            "temporal_config": temporal_cfg,
        },
        "R5": {
            "name": "Selected Engineering Candidate (E1 + E2 + Proposals + Calibrated Hysteresis)",
            "e1_fix": True,
            "reset_on_unadmitted": False,
            "proposals": True,
            "temporal_config": {
                **temporal_cfg,
                "min_positive_seconds": 0.8,
                "min_observations": 2,
            },
        },
    }

    out_records = {}
    for exp_id, exp_meta in experiments.items():
        per_sub = {}
        for sub, sdata in subject_streams.items():
            injections = proposals_by_subject[sub] if exp_meta["proposals"] else None
            res = evaluate_stream(
                observations=sdata["observations"],
                gt_intervals=sdata["gt"],
                total_seconds=sdata["total_seconds"],
                temporal_config=exp_meta["temporal_config"],
                matching_policy=policy,
                reset_on_unadmitted=exp_meta["reset_on_unadmitted"],
                e1_fix=exp_meta["e1_fix"],
                proposal_injections=injections,
            )
            per_sub[sub] = res

        tp_sum = sum(per_sub[s]["true_positives"] for s in per_sub)
        fp_sum = sum(per_sub[s]["false_positives"] for s in per_sub)
        fn_sum = sum(per_sub[s]["false_negatives"] for s in per_sub)
        gt_sum = sum(per_sub[s]["total_gt_events"] for s in per_sub)
        min_sum = sum(per_sub[s]["evaluated_minutes"] for s in per_sub)

        prec = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) else 0.0
        rec = tp_sum / gt_sum if gt_sum else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        fa_rate = fp_sum / min_sum if min_sum else 0.0

        out_records[exp_id] = {
            "id": exp_id,
            "name": exp_meta["name"],
            "e1_fix": exp_meta["e1_fix"],
            "e2_preservation": not exp_meta["reset_on_unadmitted"],
            "proposals_injected": exp_meta["proposals"],
            "temporal_config": exp_meta["temporal_config"],
            "true_positives": tp_sum,
            "false_positives": fp_sum,
            "false_negatives": fn_sum,
            "total_gt_events": gt_sum,
            "evaluated_minutes": round(min_sum, 2),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "false_alarms_per_minute": round(fa_rate, 4),
            "per_subject": {
                s: {
                    k: v
                    for k, v in per_sub[s].items()
                    if k not in ("matched_events",)
                }
                for s in per_sub
            },
        }

    return {
        "schema_version": 1,
        "cohort": ["gC-14", "gZ-36"],
        "total_gt_events": 20,
        "evaluated_minutes": 15.27,
        "experiments": out_records,
    }


if __name__ == "__main__":
    results = run_replays()
    print(json.dumps(results, indent=2))
