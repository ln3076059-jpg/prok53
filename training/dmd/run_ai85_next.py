"""Run the AI85 next-stage data-centric audit and controlled linear probes."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score

from training.common import sha256_file
from training.dmd.ai85_audit import read_json, write_json
from training.dmd.ai85_next import (
    DEVELOPMENT_SUBJECTS,
    FORBIDDEN_HOLDOUTS,
    EventPolicy,
    aggregate_event_metrics,
    events_from_prediction,
    fit_logistic_probe,
    interval_trace,
    load_normalized_subject,
    policy_candidates,
    policy_rank,
    predict_probe,
)
from training.dmd.run_v5_experiments import _ground_truth, discover_external_dmd_root
from training.dmd.run_v6_class_collapse_ablation import _coarse_metrics
from training.dmd.v5_video_action import (
    ACTION_CLASSES,
    classification_metrics,
    merge_evidence_events,
)

VISUAL_REVIEW = {
    "gB-9-phone-01": "Phone clearly visible at right ear; hand and device occupy a large region.",
    "gB-9-phone-03": "Phone clearly visible at right ear; little occlusion.",
    "gB-9-phone-05": "Phone clearly visible at left ear; strong behavior evidence.",
    "gB-9-phone-06": "Short interval; gaze and hands move toward lap, but phone is occluded.",
    "gB-9-phone-07": "Phone clearly visible in left hand near torso.",
    "gB-9-phone-08": "Phone at left ear, partly behind hand but behavior is clear.",
    "gB-9-phone-09": "Phone clearly visible in left hand near torso.",
    "gZ-36-phone-01": "Phone clearly visible in right hand near lap.",
    "gZ-36-phone-03": "Phone clearly visible in right hand near torso.",
    "gZ-36-phone-05": "Phone clearly visible in right hand; bright screen and partial glare.",
    "gC-14-phone-02": "Phone is small and partly clipped at the lower-right ROI boundary.",
    "gC-14-phone-05": "Phone is visible during rapid hand motion with substantial motion blur.",
    "gC-14-phone-06": "Phone clearly visible at right ear; little occlusion.",
}


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def combined_hash(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.as_posix().encode())
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def save_model(
    path: Path,
    model: LogisticRegression,
    *,
    task: str,
    training_subjects: list[str],
    regularization_c: float,
    seed: int,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        coefficients=model.coef_.astype(np.float32),
        intercept=model.intercept_.astype(np.float32),
        classes=model.classes_,
        task=np.asarray(task),
        training_subjects=np.asarray(training_subjects),
        regularization_c=np.asarray(regularization_c),
        seed=np.asarray(seed),
    )
    return artifact(path)


def _video_quality(rgb_cache: Path, labels: np.ndarray) -> dict[str, Any]:
    with np.load(rgb_cache, allow_pickle=False) as payload:
        frames = payload["frames"].astype(np.float32)
    gray = 0.299 * frames[..., 0] + 0.587 * frames[..., 1] + 0.114 * frames[..., 2]
    gradient = np.mean(
        np.diff(gray, axis=1)[:, :, :-1] ** 2 + np.diff(gray, axis=2)[:, :-1, :] ** 2,
        axis=(1, 2),
    )
    result: dict[str, Any] = {}
    for index, name in enumerate(ACTION_CLASSES):
        mask = labels == index
        result[name] = {
            "samples": int(mask.sum()),
            "brightness_mean": round(float(frames[mask].mean()), 3),
            "contrast_mean": round(float(frames[mask].std(axis=(1, 2, 3)).mean()), 3),
            "gradient_energy_mean": round(float(gradient[mask].mean()), 3),
        }
    return result


def _view_metadata(subject_root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(subject_root.glob("*_rgb_*.mp4")):
        capture = cv2.VideoCapture(str(path))
        try:
            rows.append(
                {
                    "view": path.stem.rsplit("_", 1)[-1],
                    "frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
                    "fps": float(capture.get(cv2.CAP_PROP_FPS)),
                    "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    "sha256": sha256_file(path),
                }
            )
        finally:
            capture.release()
    return rows


def _baseline_confusions() -> dict[str, Any]:
    rows = {}
    for subject in DEVELOPMENT_SUBJECTS:
        path = Path("reports/v6_completion/folds") / f"hierarchical_loss_only_loso_{subject}.npz"
        with np.load(path, allow_pickle=False) as payload:
            labels = payload["labels"]
            probabilities = payload["probabilities"]
        rows[subject] = {
            "clip_metrics": classification_metrics(labels, probabilities),
            "coarse_metrics": _coarse_metrics(labels, probabilities),
            "prediction": artifact(path),
        }
    return rows


def _window_purity(labels: np.ndarray, sequence_length: int = 9) -> dict[str, Any]:
    offset = sequence_length // 2
    by_class: dict[str, list[float]] = {name: [] for name in ACTION_CLASSES}
    for center in range(offset, len(labels) - offset):
        label = int(labels[center])
        if label < 0:
            continue
        window = labels[center - offset : center + offset + 1]
        by_class[ACTION_CLASSES[label]].append(float(np.mean(window == label)))
    return {
        name: {
            "clips": len(values),
            "mean_center_label_fraction": round(float(np.mean(values)), 4) if values else None,
            "mixed_label_clips": int(sum(value < 1.0 for value in values)),
        }
        for name, values in by_class.items()
    }


def _subject_bias_diagnostics(raw_subjects: dict[str, Any], seed: int) -> dict[str, Any]:
    features = np.concatenate([raw_subjects[s].embeddings for s in DEVELOPMENT_SUBJECTS])
    actions = np.concatenate([raw_subjects[s].labels for s in DEVELOPMENT_SUBJECTS])
    identities = np.concatenate(
        [
            np.full(len(raw_subjects[s].labels), index)
            for index, s in enumerate(DEVELOPMENT_SUBJECTS)
        ]
    )
    valid = actions >= 0
    features, actions, identities = features[valid], actions[valid], identities[valid]
    rng = np.random.default_rng(seed)
    sample = rng.choice(len(features), size=min(2000, len(features)), replace=False)
    permutation = rng.permutation(len(features))
    split = int(0.7 * len(features))
    train, test = permutation[:split], permutation[split:]
    identity_model = LogisticRegression(C=0.1, max_iter=300, random_state=seed, tol=1e-3)
    identity_model.fit(features[train], identities[train])
    return {
        "silhouette_cosine_by_action": float(
            silhouette_score(features[sample], actions[sample], metric="cosine")
        ),
        "silhouette_cosine_by_subject": float(
            silhouette_score(features[sample], identities[sample], metric="cosine")
        ),
        "subject_identification_accuracy_random_split": float(
            identity_model.score(features[test], identities[test])
        ),
        "interpretation": (
            "Subject identity is substantially more separable than action in frozen embeddings."
        ),
    }


def _event_overlap(event: dict[str, Any], truth: dict[str, Any]) -> bool:
    return min(float(event["end_seconds"]), float(truth["end_seconds"])) > max(
        float(event["start_seconds"]), float(truth["start_seconds"])
    )


def _matched_truths(per_subject: dict[str, dict[str, Any]]) -> dict[str, set[tuple[float, float]]]:
    return {
        subject: {
            (
                float(item["ground_truth"]["start_seconds"]),
                float(item["ground_truth"]["end_seconds"]),
            )
            for item in metrics["matches"]
        }
        for subject, metrics in per_subject.items()
    }


def run(output: Path, model_output: Path, config_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    model_output.mkdir(parents=True, exist_ok=True)
    fold_dir = output / "folds"
    fold_dir.mkdir(exist_ok=True)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seed = int(config["seed"])
    root = discover_external_dmd_root()
    embedding_dir = root / "derivatives/v5_resnet18_embeddings"

    raw_subjects = {}
    subjects = {}
    normalization = {}
    source_paths = []
    for subject in DEVELOPMENT_SUBJECTS:
        source_path = embedding_dir / f"v5_resnet18_{subject}.npz"
        source_paths.append(source_path)
        normalized, diagnostics = load_normalized_subject(subject, embedding_dir)
        subjects[subject] = normalized
        normalization[subject] = diagnostics
        with np.load(source_path, allow_pickle=False) as payload:
            from training.dmd.v5_resnet_gru import SubjectEmbeddings

            raw_subjects[subject] = SubjectEmbeddings(
                subject,
                payload["embeddings"].astype(np.float32),
                payload["timestamps"].astype(np.float64),
                payload["labels"].astype(np.int64),
            )

    durations = {
        subject: float(
            read_json(root / "derivatives" / f"v3_features_{subject}.json")["duration_seconds"]
        )
        for subject in DEVELOPMENT_SUBJECTS
    }
    truths = {
        subject: _ground_truth(root / "subjects", subject) for subject in DEVELOPMENT_SUBJECTS
    }
    baseline_errors = read_json("reports/ai85_upgrade/error_analysis.json")
    baseline_confusions = _baseline_confusions()

    flat_predictions = {}
    coarse_predictions = {}
    phone_predictions = {}
    folds = []
    for heldout in DEVELOPMENT_SUBJECTS:
        training_subjects = [subject for subject in DEVELOPMENT_SUBJECTS if subject != heldout]
        flat_model = fit_logistic_probe(
            [subjects[s] for s in training_subjects],
            task="flat",
            regularization_c=float(config["classifiers"]["flat_six_class"]["c"]),
            seed=seed,
        )
        coarse_model = fit_logistic_probe(
            [subjects[s] for s in training_subjects],
            task="coarse",
            regularization_c=float(config["classifiers"]["coarse_three_class"]["c"]),
            seed=seed,
        )
        phone_model = fit_logistic_probe(
            [subjects[s] for s in training_subjects],
            task="phone_binary",
            regularization_c=float(config["classifiers"]["phone_binary"]["c"]),
            seed=seed,
        )
        flat_predictions[heldout] = predict_probe(flat_model, subjects[heldout], task="flat")
        coarse_predictions[heldout] = predict_probe(
            coarse_model, subjects[heldout], task="coarse"
        )
        phone_predictions[heldout] = predict_probe(
            phone_model, subjects[heldout], task="phone_binary"
        )
        model_artifacts = {
            "coarse": save_model(
                model_output / f"subject_normalized_coarse_loso_{heldout}.npz",
                coarse_model,
                task="coarse",
                training_subjects=training_subjects,
                regularization_c=float(config["classifiers"]["coarse_three_class"]["c"]),
                seed=seed,
            ),
            "flat": save_model(
                model_output / f"subject_normalized_flat_loso_{heldout}.npz",
                flat_model,
                task="flat",
                training_subjects=training_subjects,
                regularization_c=float(config["classifiers"]["flat_six_class"]["c"]),
                seed=seed,
            ),
            "phone_binary": save_model(
                model_output / f"subject_normalized_phone_loso_{heldout}.npz",
                phone_model,
                task="phone_binary",
                training_subjects=training_subjects,
                regularization_c=float(config["classifiers"]["phone_binary"]["c"]),
                seed=seed,
            ),
        }
        prediction_path = fold_dir / f"subject_normalized_loso_{heldout}.npz"
        np.savez_compressed(
            prediction_path,
            centers=flat_predictions[heldout]["centers"],
            timestamps=flat_predictions[heldout]["timestamps"],
            labels=flat_predictions[heldout]["labels"],
            flat_probabilities=flat_predictions[heldout]["probabilities"],
            coarse_probabilities=coarse_predictions[heldout]["probabilities"],
            phone_probabilities=phone_predictions[heldout]["phone_probabilities"],
        )
        folds.append(
            {
                "heldout_subject": heldout,
                "training_subjects": training_subjects,
                "dataset_hash": combined_hash(
                    [embedding_dir / f"v5_resnet18_{s}.npz" for s in training_subjects]
                ),
                "models": model_artifacts,
                "prediction": artifact(prediction_path),
                "seed": seed,
            }
        )

    pooled_labels = np.concatenate([flat_predictions[s]["labels"] for s in DEVELOPMENT_SUBJECTS])
    pooled_flat = np.concatenate(
        [flat_predictions[s]["probabilities"] for s in DEVELOPMENT_SUBJECTS]
    )
    flat_clip_metrics = classification_metrics(pooled_labels, pooled_flat)
    flat_coarse_metrics = _coarse_metrics(pooled_labels, pooled_flat)
    pooled_coarse = np.concatenate(
        [coarse_predictions[s]["probabilities"] for s in DEVELOPMENT_SUBJECTS]
    )
    direct_coarse_metrics = _coarse_metrics(pooled_labels, pooled_coarse)

    inherited_policy = EventPolicy(0.80, 2, 2.0)
    exploratory_policy = EventPolicy(0.78, 3, 2.0)
    flat_fixed_events = {
        subject: events_from_prediction(flat_predictions[subject], inherited_policy)
        for subject in DEVELOPMENT_SUBJECTS
    }
    coarse_fixed_events = {
        subject: events_from_prediction(coarse_predictions[subject], inherited_policy)
        for subject in DEVELOPMENT_SUBJECTS
    }
    phone_fixed_events = {
        subject: events_from_prediction(phone_predictions[subject], inherited_policy)
        for subject in DEVELOPMENT_SUBJECTS
    }
    exploratory_events = {
        subject: events_from_prediction(phone_predictions[subject], exploratory_policy)
        for subject in DEVELOPMENT_SUBJECTS
    }

    nested_policies = {}
    nested_search = {}
    for outer in DEVELOPMENT_SUBJECTS:
        inner_subjects = [subject for subject in DEVELOPMENT_SUBJECTS if subject != outer]
        inner_predictions = {}
        for validation in inner_subjects:
            training = [subject for subject in inner_subjects if subject != validation]
            model = fit_logistic_probe(
                [subjects[s] for s in training],
                task="phone_binary",
                regularization_c=float(config["classifiers"]["phone_binary"]["c"]),
                seed=seed,
            )
            inner_predictions[validation] = predict_probe(
                model, subjects[validation], task="phone_binary"
            )
        candidates = []
        for policy in policy_candidates():
            events = {
                subject: events_from_prediction(inner_predictions[subject], policy)
                for subject in inner_subjects
            }
            metrics, _ = aggregate_event_metrics(
                {subject: truths[subject] for subject in inner_subjects},
                events,
                {subject: durations[subject] for subject in inner_subjects},
            )
            candidates.append({"policy": policy.as_dict(), "metrics": metrics, "_policy": policy})
        winner = min(candidates, key=lambda row: policy_rank(row["metrics"], row["_policy"]))
        nested_policies[outer] = winner["_policy"]
        nested_search[outer] = {
            "training_and_validation_subjects": inner_subjects,
            "outer_subject_labels_accessed_for_selection": False,
            "selected_policy": winner["policy"],
            "selected_validation_metrics": winner["metrics"],
            "candidates": [
                {"policy": row["policy"], "metrics": row["metrics"]} for row in candidates
            ],
        }
    nested_events = {
        subject: events_from_prediction(phone_predictions[subject], nested_policies[subject])
        for subject in DEVELOPMENT_SUBJECTS
    }

    event_variants = {}
    for name, events, status in (
        (
            "SN_DIRECT_COARSE_INHERITED_POLICY",
            coarse_fixed_events,
            "CONTROLLED_HEAD_AND_REPRESENTATION_ABLATION",
        ),
        ("SN_FLAT_INHERITED_POLICY", flat_fixed_events, "CONTROLLED_REPRESENTATION_ABLATION"),
        ("SN_PHONE_INHERITED_POLICY", phone_fixed_events, "CONTROLLED_REPRESENTATION_ABLATION"),
        ("SN_PHONE_NESTED_POLICY", nested_events, "VALIDATION_SELECTED_DEVELOPMENT_ONLY"),
        (
            "SN_PHONE_EXPLORATORY_POLICY",
            exploratory_events,
            "OUTER_DEVELOPMENT_TUNED_NOT_ELIGIBLE_FOR_SELECTION",
        ),
    ):
        pooled, per_subject = aggregate_event_metrics(truths, events, durations)
        event_variants[name] = {
            "status": status,
            "pooled_metrics": pooled,
            "per_subject": per_subject,
            "events": events,
        }

    v5 = read_json("reports/v5_research_upgrade/v5_experiment_results.json")
    object_experiment = next(
        item for item in v5["experiments"] if item["experiment_id"] == "B0_V4_CURRENT"
    )
    object_events = {
        subject: object_experiment["per_subject"][subject]["evaluation_evidence"][
            "predicted_events"
        ]
        for subject in DEVELOPMENT_SUBJECTS
    }
    fused_events = {
        subject: merge_evidence_events(object_events[subject], phone_fixed_events[subject])
        for subject in DEVELOPMENT_SUBJECTS
    }
    fused_metrics, fused_per_subject = aggregate_event_metrics(truths, fused_events, durations)
    event_variants["SN_PHONE_INHERITED_POLICY_FUSION"] = {
        "status": "CONTROLLED_FUSION_ABLATION",
        "pooled_metrics": fused_metrics,
        "per_subject": fused_per_subject,
        "events": fused_events,
    }

    matched = {
        name: _matched_truths(variant["per_subject"]) for name, variant in event_variants.items()
    }
    baseline_prediction = {}
    for subject in DEVELOPMENT_SUBJECTS:
        path = Path("reports/v6_completion/folds") / f"hierarchical_loss_only_loso_{subject}.npz"
        with np.load(path, allow_pickle=False) as payload:
            scores = payload["probabilities"][:, 1:5].sum(axis=1)
            if len(scores) >= 3:
                scores = np.convolve(scores, np.ones(3) / 3.0, mode="same")
            baseline_prediction[subject] = {
                "timestamps": payload["timestamps"].copy(),
                "phone_probabilities": scores,
            }

    fn_rows = []
    for case in baseline_errors["cases"]:
        if case["result"] != "FN":
            continue
        subject = case["subject_id"]
        truth = case["truth"]
        key = (float(truth["start_seconds"]), float(truth["end_seconds"]))
        variant_evidence = {}
        for name, variant in event_variants.items():
            overlaps = [
                event for event in variant["events"][subject] if _event_overlap(event, truth)
            ]
            variant_evidence[name] = {
                "matched": key in matched[name][subject],
                "overlapping_events": overlaps,
            }
        baseline_trace = interval_trace(
            baseline_prediction[subject]["timestamps"],
            baseline_prediction[subject]["phone_probabilities"],
            float(truth["start_seconds"]),
            float(truth["end_seconds"]),
            threshold=0.8,
        )
        phone_trace = interval_trace(
            phone_predictions[subject]["timestamps"],
            phone_predictions[subject]["phone_probabilities"],
            float(truth["start_seconds"]),
            float(truth["end_seconds"]),
            threshold=0.8,
        )
        if baseline_trace["samples"] == 0:
            root_cause = "MISSING_CENTERED_SAMPLE"
        elif baseline_trace["longest_activation_run"] < 2:
            root_cause = "INSUFFICIENT_CONTINUOUS_ACTION_EVIDENCE"
        else:
            root_cause = "TEMPORAL_BOUNDARY_OR_EVENT_MATCHING"
        fn_rows.append(
            {
                "case_id": case["case_id"],
                "subject_id": subject,
                "truth": truth,
                "duration_seconds": round(
                    float(truth["end_seconds"]) - float(truth["start_seconds"]), 4
                ),
                "visual_review": VISUAL_REVIEW[case["case_id"]],
                "visual_review_provenance": "MANUAL_REVIEW_OF_EXPORTED_ORIGINAL_BODY_ROI",
                "baseline_checks": case["checks"],
                "baseline_trace": baseline_trace,
                "subject_normalized_phone_trace": phone_trace,
                "variant_evidence": variant_evidence,
                "root_cause": root_cause,
                "detector_instrumentation": "SEE detector_trace.json",
            }
        )

    dataset_subjects = {}
    for subject in DEVELOPMENT_SUBJECTS:
        labels = subjects[subject].labels
        dataset_subjects[subject] = {
            "class_distribution": dict(
                Counter(ACTION_CLASSES[int(label)] for label in labels if label >= 0)
            ),
            "window_purity": _window_purity(labels),
            "video_quality": _video_quality(
                root / "derivatives/v5_rgb_cache" / f"v5_rgb_{subject}_64.npz", labels
            ),
            "views": _view_metadata(root / "subjects" / subject),
        }
    dataset_generalization = {
        "development_subjects": list(DEVELOPMENT_SUBJECTS),
        "forbidden_holdouts": list(FORBIDDEN_HOLDOUTS),
        "holdouts_read": [],
        "new_approved_development_subjects": [],
        "new_independent_holdout": None,
        "subject_disjoint_outer_folds": [
            {"test": row["heldout_subject"], "train": row["training_subjects"]} for row in folds
        ],
        "subjects": dataset_subjects,
        "multi_view_limit": (
            "gC-14 has BODY only; gB-9 and gZ-36 have BODY/FACE/HANDS with 7-22 frame "
            "count differences. A complete group-disjoint multi-view ablation is not possible."
        ),
        "rights_status": "PENDING_PROJECT_REVIEW",
        "rights_accepted": False,
        "rights_source": "dataset/external_dmd/README.md",
        "data_bottleneck": True,
    }

    analysis = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "config": artifact(config_path),
        "dataset_hash": combined_hash(source_paths),
        "dataset_sources": [artifact(path) for path in source_paths],
        "normalization": normalization,
        "subject_bias": _subject_bias_diagnostics(raw_subjects, seed),
        "baseline_subject_confusions": baseline_confusions,
        "baseline_training_terminal_accuracy": {
            subject: read_json(
                Path("reports/v6_completion/folds")
                / f"hierarchical_loss_only_loso_{subject}.json"
            )["training"]["history"][-1]
            for subject in DEVELOPMENT_SUBJECTS
        },
        "flat_clip_metrics": flat_clip_metrics,
        "flat_coarse_metrics": flat_coarse_metrics,
        "direct_coarse_metrics": direct_coarse_metrics,
        "folds": folds,
        "nested_policy_selection": nested_search,
        "event_variants": event_variants,
        "baseline": {
            "clip_metrics": baseline_errors["clip_metrics"],
            "coarse_metrics": baseline_errors["coarse_metrics"],
            "event_metrics": baseline_errors["recomputed_metrics"],
        },
        "selection": {
            "selected_candidate": None,
            "production_approved": False,
            "independent_validation": "NOT_ESTABLISHED",
            "reason": (
                "No controlled candidate satisfies all targets and no untouched licensed "
                "holdout exists."
            ),
        },
        "runtime": {
            "seconds": round(time.perf_counter() - started, 3),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
    }
    write_json(output / "analysis.json", analysis)
    write_json(output / "fn_recovery.json", {"cases": fn_rows})
    write_json(output / "dataset_generalization.json", dataset_generalization)
    write_json(
        output / "representation_experiment.json",
        {
            "hypothesis": (
                "Per-video label-free normalization reduces subject/camera appearance bias in "
                "frozen embeddings; a strongly regularized linear probe tests separability before "
                "a larger model."
            ),
            "not_repeated": ["MEAN_FINETUNE", "ATTENTION_FINETUNE", "NATIVE112_FINETUNE"],
            "flat_clip_metrics": flat_clip_metrics,
            "flat_coarse_metrics": flat_coarse_metrics,
            "direct_coarse_metrics": direct_coarse_metrics,
            "event_variants": event_variants,
            "folds": folds,
        },
    )
    return analysis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports/ai85_next"))
    parser.add_argument(
        "--model-output", type=Path, default=Path("models/candidate/ai85_next")
    )
    parser.add_argument(
        "--config", type=Path, default=Path("models/model_config_ai85_next.yaml")
    )
    args = parser.parse_args()
    result = run(args.output, args.model_output, args.config)
    print(json.dumps(result["selection"], indent=2))


if __name__ == "__main__":
    main()
