"""Dense original-video sampling with frozen weights and unchanged event matcher.

Run once per output directory. Labels are loaded only AFTER visual extraction.
Existing 15-frame embeddings are reused exactly; new centers use the identical
64px -> 112px preprocessing and 15-frame temporal context spacing.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import psutil
import torch
import yaml
from torch import nn
from torchvision.models import resnet18

from training.dmd.adapter import parse_dmd_openlabel
from training.dmd.adaptive_sampling import (
    SamplingConfig,
    VisualSampler,
    confirmed_events,
    padded_roi,
)
from training.dmd.ai85_audit import artifact, baseline_inputs, read_json, write_json
from training.dmd.ai85_next import DEVELOPMENT_SUBJECTS, aggregate_event_metrics
from training.dmd.holdout_guard import assert_not_consumed_holdout_for_development
from training.dmd.run_v5_experiments import _ground_truth, discover_external_dmd_root
from training.dmd.v5_video_action import (
    classification_metrics,
    label_timestamps,
    merge_evidence_events,
    probabilities_to_events,
)
from training.dmd.v6_action_hierarchy import (
    HierarchicalResNetGRUClassifier,
    hierarchical_probabilities,
)

VARIANTS = ("T0", "T1", "T2", "T3", "T4", "T5")


def timing(values: list[float]) -> dict:
    return {
        "count": len(values),
        "total_seconds": float(sum(values)),
        "p50_ms": float(np.percentile(values, 50) * 1000) if values else None,
        "p95_ms": float(np.percentile(values, 95) * 1000) if values else None,
    }


def extract(subject: str, output: Path, sampling: SamplingConfig | None = None) -> dict:
    assert_not_consumed_holdout_for_development(subject, "ai85_plus_visual_extraction")
    if subject not in DEVELOPMENT_SUBJECTS:
        raise ValueError("subject is not in the frozen development allowlist")
    sampling = sampling or SamplingConfig()
    if (sampling.base_stride, sampling.dense_stride) != (15, 3):
        raise ValueError(
            "this inherited-context experiment requires base stride 15 and dense stride 3"
        )
    destination = output / f"{subject}_dense.npz"
    if destination.exists():
        raise FileExistsError(destination)
    root = discover_external_dmd_root()
    video = next((root / "subjects" / subject).glob("*_rgb_body.mp4"))
    old_path = root / "derivatives/v5_resnet18_embeddings" / f"v5_resnet18_{subject}.npz"
    rgb_path = root / "derivatives/v5_rgb_cache" / f"v5_rgb_{subject}_64.npz"
    with np.load(old_path) as old, np.load(rgb_path) as rgb:
        old_embeddings = old["embeddings"].copy()
        old_indices = rgb["frame_indices"].copy()
        old_times = old["timestamps"].astype(np.float64)
    lookup = {int(frame): i for i, frame in enumerate(old_indices)}
    weights = Path(torch.hub.get_dir()) / "checkpoints/resnet18-f37072fd.pth"
    model = resnet18(weights=None)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    model.fc = nn.Identity()
    model.eval().requires_grad_(False)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.arange(0, count, 3)
    embeddings = np.empty((len(indices), 512), np.float16)
    masks = np.zeros((len(indices), len(VARIANTS)), bool)
    motion = np.zeros(len(indices), np.float32)
    timestamps = indices / fps
    sampler = VisualSampler(sampling)
    batch, batch_indices = [], []
    decode_times, encode_times, motion_times = [], [], []
    started = time.perf_counter()
    peak_rss = 0

    def flush() -> None:
        nonlocal peak_rss
        if not batch:
            return
        t0 = time.perf_counter()
        tensor = torch.from_numpy(np.stack(batch)).permute(0, 3, 1, 2).float() / 255
        tensor = torch.nn.functional.interpolate(
            tensor, (112, 112), mode="bilinear", align_corners=False
        )
        with torch.inference_mode():
            embeddings[batch_indices] = model((tensor - mean) / std).numpy().astype(np.float16)
        encode_times.append(time.perf_counter() - t0)
        peak_rss = max(peak_rss, psutil.Process().memory_info().rss)
        batch.clear()
        batch_indices.clear()

    try:
        for frame_index in range(count):
            t0 = time.perf_counter()
            if not cap.grab():
                raise ValueError(f"early EOF at {frame_index}")
            if frame_index % 3:
                decode_times.append(time.perf_counter() - t0)
                continue
            ok, frame = cap.retrieve()
            if not ok:
                raise ValueError(f"decode failed at {frame_index}")
            decode_times.append(time.perf_counter() - t0)
            j = frame_index // 3
            roi = padded_roi(frame.shape, (0, 0.10, 0.65, 0.95))
            crop = frame[roi[1] : roi[3], roi[0] : roi[2]]
            t0 = time.perf_counter()
            decision = sampler.observe(frame_index, float(timestamps[j]), crop)
            motion_times.append(time.perf_counter() - t0)
            masks[j] = [decision[v] for v in VARIANTS]
            motion[j] = decision["motion"]
            if frame_index in lookup:
                k = lookup[frame_index]
                embeddings[j] = old_embeddings[k]
                timestamps[j] = old_times[k]
            else:
                crop = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA)
                batch.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                batch_indices.append(j)
                if len(batch) >= 16:
                    flush()
            if j and j % 512 == 0:
                print(f"dense {subject}: {j}/{len(indices)}", flush=True)
        flush()
    finally:
        cap.release()
    # Ground truth cannot influence sampler decisions or extraction indices.
    annotation_path = next((root / "subjects" / subject).glob("*_ann_distraction.json"))
    labels, _ = label_timestamps(timestamps, parse_dmd_openlabel(annotation_path))
    # Preserve original float32-boundary label semantics at inherited centers.
    with np.load(old_path) as old:
        for frame, k in lookup.items():
            labels[frame // 3] = old["labels"][k]
    np.savez_compressed(
        destination,
        embeddings=embeddings,
        frame_indices=indices,
        timestamps=timestamps,
        labels=labels,
        masks=masks,
        motion=motion,
    )
    result = {
        "sampling_config": asdict(sampling),
        "subject": subject,
        "fps": fps,
        "video_frames": count,
        "dense_samples": len(indices),
        "inherited_embeddings": len(old_indices),
        "visual_labels_used": False,
        "decode": timing(decode_times),
        "motion": timing(motion_times),
        "encoder_batches": timing(encode_times),
        "batch_size": 16,
        "peak_rss_mb": peak_rss / 1024**2,
        "seconds": time.perf_counter() - started,
        "video": artifact(video),
        "annotation": artifact(annotation_path),
        "weights": artifact(weights),
        "inherited_cache": artifact(old_path),
        "output": artifact(destination),
    }
    write_json(output / f"{subject}_extraction.json", result)
    return result


def matched_keys(evaluation: dict) -> set:
    return {
        (m["ground_truth"]["start_seconds"], m["ground_truth"]["end_seconds"])
        for m in evaluation["matches"]
    }


def run(output: Path) -> dict:
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "temporal_results.json").exists():
        raise FileExistsError("use a new output directory; measured history is immutable")
    config = yaml.safe_load(Path("models/model_config_ai85_plus.yaml").read_text())
    sampling = SamplingConfig(**{key: config["sampling"][key] for key in asdict(SamplingConfig())})
    if (sampling.base_stride, sampling.dense_stride) != (15, 3):
        raise ValueError("changing temporal grid requires a separately validated context mapping")
    baseline, old_predictions, durations = baseline_inputs()
    historical = baseline["experiments"][1]
    root = discover_external_dmd_root()
    truths = {s: _ground_truth(root / "subjects", s) for s in DEVELOPMENT_SUBJECTS}
    object_report = read_json("reports/v5_research_upgrade/v5_experiment_results.json")
    obj = next(e for e in object_report["experiments"] if e["experiment_id"] == "B0_V4_CURRENT")
    predictions = {v: {} for v in VARIANTS}
    frame_predictions = {v: [] for v in VARIANTS}
    frame_labels = {v: [] for v in VARIANTS}
    subjects = {}
    for s in DEVELOPMENT_SUBJECTS:
        dense_path = output / f"{s}_dense.npz"
        extraction = (
            read_json(output / f"{s}_extraction.json")
            if dense_path.exists()
            else extract(s, output, sampling)
        )
        saved_sampling = extraction.get("sampling_config", asdict(SamplingConfig()))
        if saved_sampling != asdict(sampling):
            raise ValueError(
                "cached extraction sampling config differs; use a fresh output directory"
            )
        with np.load(dense_path) as raw:
            dense = {k: raw[k].copy() for k in raw.files}
        checkpoint = Path(config["action"]["checkpoint"].format(subject=s))
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload["heldout_subject"] != s or set(payload["training_subjects"]) != set(
            DEVELOPMENT_SUBJECTS
        ) - {s}:
            raise ValueError("checkpoint split mismatch")
        model = HierarchicalResNetGRUClassifier()
        model.load_state_dict(payload["model_state_dict"])
        model.eval().requires_grad_(False)
        # 20 dense steps = 60 original frames, the unchanged 4x15-frame context.
        centers = np.arange(20, len(dense["embeddings"]) - 20)
        probabilities, head_times = [], []
        for start in range(0, len(centers), 64):
            local = centers[start : start + 64]
            windows = local[:, None] + np.arange(-20, 21, 5)[None, :]
            t0 = time.perf_counter()
            with torch.inference_mode():
                coarse, phone = model(
                    torch.from_numpy(dense["embeddings"][windows].astype(np.float32))
                )
                probabilities.append(hierarchical_probabilities(coarse, phone).numpy())
            head_times.append(time.perf_counter() - t0)
        probabilities = np.concatenate(probabilities)
        times = dense["timestamps"][centers]
        labels = dense["labels"][centers]
        selection = dense["masks"][centers]
        # Original T0 ended at the last inherited embedding with complete context.
        t0mask = selection[:, 0] & (times <= old_predictions[s]["timestamps"][-1] + 1e-6)
        selection[:, 0] = t0mask
        max_error = float(np.abs(probabilities[t0mask] - old_predictions[s]["probabilities"]).max())
        if max_error > 2e-6:
            raise ValueError(f"T0 probabilities failed reconciliation: {s}: {max_error}")
        objects = obj["per_subject"][s]["evaluation_evidence"]["predicted_events"]
        counts = {}
        for col, variant in enumerate(VARIANTS):
            mask = selection[:, col]
            if variant == "T5":
                events = confirmed_events(times[mask], probabilities[mask, 1:5].sum(1))
            else:
                events = probabilities_to_events(
                    times[mask],
                    probabilities[mask],
                    threshold=0.8,
                    minimum_samples=2,
                    merge_gap_seconds=2.0,
                )
            predictions[variant][s] = merge_evidence_events(objects, events, gap_seconds=2.0)
            frame_predictions[variant].append(probabilities[mask])
            frame_labels[variant].append(labels[mask])
            needed = np.unique((centers[mask, None] + np.arange(-20, 21, 5)).ravel())
            counts[variant] = {
                "selected_centers": int(mask.sum()),
                "required_unique_context_embeddings": len(needed),
                "action_head_calls_ratio_to_T0": float(mask.sum() / t0mask.sum()),
                "center_lookahead_seconds": 60 / extraction["fps"],
                "event_smoothing_extra_lookahead_max_seconds": float(np.diff(times[mask]).max())
                if variant != "T5"
                else 0.0,
            }
        np.savez_compressed(
            output / f"{s}_predictions.npz",
            timestamps=times,
            labels=labels,
            probabilities=probabilities,
            masks=selection,
            frame_indices=dense["frame_indices"][centers],
        )
        subjects[s] = {
            "extraction": extraction,
            "checkpoint": artifact(checkpoint),
            "T0_max_probability_error": max_error,
            "sampling": counts,
            "temporal_head_batches": timing(head_times),
        }
        print(f"temporal completed {s}", flush=True)
    variants = []
    for variant in VARIANTS:
        pooled, per_subject = aggregate_event_metrics(truths, predictions[variant], durations)
        labels = np.concatenate(frame_labels[variant])
        probs = np.concatenate(frame_predictions[variant])
        classification = classification_metrics(labels, probs)
        coarse = np.stack((probs[:, 0], probs[:, 1:5].sum(1), probs[:, 5]), axis=1).argmax(1)
        safe_recall = float((coarse[labels == 0] == 0).mean())
        recovered, lost, new_fp, removed_fp = [], [], [], []
        buckets = {
            k: {"total": 0, "matched": 0, "sampled": 0} for k in ("<1", "1-3", "3-10", ">10")
        }
        for s, evaluation in per_subject.items():
            original = historical["per_subject"][s]
            old, new = matched_keys(original), matched_keys(evaluation)
            recovered.extend([[s, *k] for k in sorted(new - old)])
            lost.extend([[s, *k] for k in sorted(old - new)])

            def fp_key(e):
                return e["start_seconds"], e["end_seconds"]

            oldfp = {fp_key(e) for e in original["unmatched_predictions"]}
            newfp = {fp_key(e) for e in evaluation["unmatched_predictions"]}
            new_fp.extend([[s, *k] for k in sorted(newfp - oldfp)])
            removed_fp.extend([[s, *k] for k in sorted(oldfp - newfp)])
            with np.load(output / f"{s}_predictions.npz") as p:
                selected_times = p["timestamps"][p["masks"][:, VARIANTS.index(variant)]]
            for truth in truths[s]:
                a, b = truth["start_seconds"], truth["end_seconds"]
                duration = b - a
                bucket = buckets[
                    "<1"
                    if duration < 1
                    else "1-3"
                    if duration <= 3
                    else "3-10"
                    if duration <= 10
                    else ">10"
                ]
                bucket["total"] += 1
                bucket["matched"] += int((a, b) in new)
                bucket["sampled"] += int(np.any((selected_times >= a) & (selected_times <= b)))
        for b in buckets.values():
            b["recall"] = b["matched"] / b["total"] if b["total"] else None
        variants.append(
            {
                "id": variant,
                "pooled": pooled,
                "per_subject": per_subject,
                "classification": classification,
                "coarse_safe_recall": safe_recall,
                "recovered": recovered,
                "lost": lost,
                "new_fp_intervals": new_fp,
                "removed_fp_intervals": removed_fp,
                "duration_buckets": buckets,
                "duplicates": sum(p["duplicate_events"] for p in per_subject.values()),
                "selection_eligible": False,
            }
        )
    if any(
        variants[0]["pooled"][k] != historical["pooled_metrics"][k]
        for k in ("true_positives", "false_positives", "false_negatives")
    ):
        raise ValueError("T0 event counts did not reproduce historical baseline")
    result = {
        "status": "DEVELOPMENT_ONLY_NO_SELECTION",
        "config": artifact(Path("models/model_config_ai85_plus.yaml")),
        "subjects": subjects,
        "variants": variants,
        "performance_scope": "shared dense extraction and head inference; not full-system FPS",
        "limitations": [
            "All three subjects were repeatedly consumed for development.",
            "Sampling is causal; centered BiGRU and inherited smoothing require lookahead.",
            "Adaptive masking saves potential head calls; dense benchmark extracted all contexts.",
            "Count context embeddings; selected-center counts are not encoder savings.",
            "New/removed FP comparison is by exact interval, not semantic identity.",
        ],
    }
    write_json(output / "temporal_results.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports/ai85_plus/temporal"))
    args = parser.parse_args()
    run(args.output)
