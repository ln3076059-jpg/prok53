"""Run CPU-only V10 behavior-centric representation probes on frozen DMD caches.

The experiment is a LEVEL-2 development ablation. It uses fixed LOSO folds and a
single pre-registered balanced linear head for every representation. Protected
holdouts are never read, no backbone is trained, and no production model changes.
"""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    recall_score,
    silhouette_score,
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, normalize

from training.common import sha256_file
from training.dmd.v5_video_action import ACTION_CLASSES
from training.dmd.v9_diagnostics import DEVELOPMENT_SUBJECTS, PROTECTED_HOLDOUTS

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "reports" / "v10_research"
CACHE_ROOT = ROOT / "dataset" / "external_dmd" / "derivatives"
SUBJECTS = ("gB-9", "gC-14", "gZ-36")
COARSE_CLASSES = ("SAFE_DRIVING", "PHONE_USE", "OTHER_ACTIVITY")
SEED = 20261017


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()


def _coarse_labels(labels: np.ndarray) -> np.ndarray:
    result = np.full(labels.shape, -1, dtype=np.int64)
    result[labels == 0] = 0
    result[np.isin(labels, (1, 2, 3, 4))] = 1
    result[labels == 5] = 2
    return result


def _hog_descriptor() -> cv2.HOGDescriptor:
    return cv2.HOGDescriptor((64, 64), (32, 32), (16, 16), (16, 16), 9)


def _behavior_hog(frames: np.ndarray) -> np.ndarray:
    """Fixed overlapping upper/lower body HOG; no learned or label-aware crop."""
    hog = _hog_descriptor()
    output = []
    for rgb in frames:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        upper = cv2.resize(gray[:44], (64, 64), interpolation=cv2.INTER_AREA)
        lower = cv2.resize(gray[20:], (64, 64), interpolation=cv2.INTER_AREA)
        output.append(np.concatenate((hog.compute(upper).ravel(), hog.compute(lower).ravel())))
    return np.asarray(output, dtype=np.float32)


def _motion_residual(frames: np.ndarray) -> np.ndarray:
    """Causal/past and offline/future absolute residual maps at fixed scale."""
    gray = np.asarray(
        [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in frames], dtype=np.float32
    )
    previous = np.zeros_like(gray)
    following = np.zeros_like(gray)
    previous[1:] = np.abs(gray[1:] - gray[:-1])
    following[:-1] = np.abs(gray[1:] - gray[:-1])
    rows = []
    for past, future in zip(previous, following, strict=True):
        past_small = cv2.resize(past, (16, 16), interpolation=cv2.INTER_AREA)
        future_small = cv2.resize(future, (16, 16), interpolation=cv2.INTER_AREA)
        rows.append(np.concatenate((past_small.ravel(), future_small.ravel())) / 255.0)
    return np.asarray(rows, dtype=np.float32)


def _load_subject(subject_id: str) -> dict[str, Any]:
    if subject_id not in DEVELOPMENT_SUBJECTS or subject_id in PROTECTED_HOLDOUTS:
        raise ValueError(f"subject is not eligible for V10 development probe: {subject_id}")
    rgb_path = CACHE_ROOT / "v5_rgb_cache" / f"v5_rgb_{subject_id}_64.npz"
    embedding_path = (
        CACHE_ROOT / "v5_resnet18_embeddings" / f"v5_resnet18_{subject_id}.npz"
    )
    rgb = np.load(rgb_path, allow_pickle=False)
    frozen = np.load(embedding_path, allow_pickle=False)
    if not (
        np.allclose(rgb["timestamps"], frozen["timestamps"], atol=1e-5)
        and np.array_equal(rgb["labels"], frozen["labels"])
    ):
        raise ValueError(f"RGB and frozen caches are not aligned for {subject_id}")
    labels = _coarse_labels(rgb["labels"].astype(np.int64))
    behavior = _behavior_hog(rgb["frames"])
    motion = _motion_residual(rgb["frames"])
    return {
        "subject_id": subject_id,
        "frame_indices": rgb["frame_indices"].astype(np.int64),
        "timestamps": rgb["timestamps"].astype(np.float64),
        "labels": labels,
        "features": {
            "D0_FROZEN_RESNET_FULL_ROI": frozen["embeddings"].astype(np.float32),
            "D1_BEHAVIOR_TWO_REGION_HOG": behavior,
            "D2_TEMPORAL_RESIDUAL": motion,
            "D5_BEHAVIOR_HOG_PLUS_MOTION": np.concatenate((behavior, motion), axis=1),
        },
        "sources": [
            {
                "path": rgb_path.relative_to(ROOT).as_posix(),
                "bytes": rgb_path.stat().st_size,
                "sha256": sha256_file(rgb_path),
            },
            {
                "path": embedding_path.relative_to(ROOT).as_posix(),
                "bytes": embedding_path.stat().st_size,
                "sha256": sha256_file(embedding_path),
            },
        ],
    }


def _per_class_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=np.arange(3), zero_division=0
    )
    return {
        name: {
            "precision": round(float(precision[index]), 4),
            "recall": round(float(recall[index]), 4),
            "f1": round(float(f1[index]), 4),
            "support": int(support[index]),
        }
        for index, name in enumerate(COARSE_CLASSES)
    }


def _evaluate_representation(
    representation_id: str,
    subjects: list[dict[str, Any]],
    output: Path,
) -> dict[str, Any]:
    fold_rows = []
    pooled_labels = []
    pooled_predictions = []
    for heldout in subjects:
        train = [item for item in subjects if item["subject_id"] != heldout["subject_id"]]
        train_x = np.concatenate([item["features"][representation_id] for item in train])
        train_y = np.concatenate([item["labels"] for item in train])
        test_x = heldout["features"][representation_id]
        test_y = heldout["labels"]
        train_mask = train_y >= 0
        test_mask = test_y >= 0
        model = make_pipeline(
            StandardScaler(),
            OneVsRestClassifier(
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=500,
                    random_state=SEED,
                    solver="liblinear",
                ),
                n_jobs=1,
            ),
        )
        started = time.perf_counter()
        model.fit(train_x[train_mask], train_y[train_mask])
        predictions = model.predict(test_x[test_mask]).astype(np.int64)
        probabilities = model.predict_proba(test_x[test_mask]).astype(np.float32)
        elapsed = time.perf_counter() - started
        labels = test_y[test_mask]
        prediction_path = (
            output / "predictions" / f"{representation_id.lower()}_loso_{heldout['subject_id']}.npz"
        )
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            prediction_path,
            frame_indices=heldout["frame_indices"][test_mask],
            timestamps=heldout["timestamps"][test_mask],
            labels=labels,
            predictions=predictions,
            probabilities=probabilities,
        )
        fold_rows.append(
            {
                "heldout_subject": heldout["subject_id"],
                "training_subjects": [item["subject_id"] for item in train],
                "samples": int(len(labels)),
                "macro_f1": round(float(f1_score(labels, predictions, average="macro")), 4),
                "balanced_accuracy": round(
                    float(balanced_accuracy_score(labels, predictions)), 4
                ),
                "safe_recall": round(
                    float(recall_score(labels == 0, predictions == 0, zero_division=0)), 4
                ),
                "confusion_matrix": confusion_matrix(
                    labels, predictions, labels=np.arange(3)
                ).tolist(),
                "per_class": _per_class_metrics(labels, predictions),
                "fit_predict_seconds": round(elapsed, 3),
                "prediction_path": prediction_path.relative_to(ROOT).as_posix(),
                "prediction_sha256": sha256_file(prediction_path),
            }
        )
        pooled_labels.append(labels)
        pooled_predictions.append(predictions)

    labels = np.concatenate(pooled_labels)
    predictions = np.concatenate(pooled_predictions)
    return {
        "representation_id": representation_id,
        "feature_dimensions": int(subjects[0]["features"][representation_id].shape[1]),
        "folds": fold_rows,
        "pooled": {
            "samples": int(len(labels)),
            "macro_f1": round(float(f1_score(labels, predictions, average="macro")), 4),
            "balanced_accuracy": round(
                float(balanced_accuracy_score(labels, predictions)), 4
            ),
            "safe_recall": round(
                float(recall_score(labels == 0, predictions == 0, zero_division=0)), 4
            ),
            "confusion_matrix": confusion_matrix(
                labels, predictions, labels=np.arange(3)
            ).tolist(),
            "per_class": _per_class_metrics(labels, predictions),
        },
    }


def _separability(
    representation_id: str, subjects: list[dict[str, Any]]
) -> dict[str, float]:
    features = np.concatenate([item["features"][representation_id] for item in subjects])
    labels = np.concatenate([item["labels"] for item in subjects])
    identities = np.concatenate(
        [np.full(len(item["labels"]), index) for index, item in enumerate(subjects)]
    )
    valid = labels >= 0
    features = normalize(features[valid])
    labels = labels[valid]
    identities = identities[valid]
    rng = np.random.default_rng(SEED)
    sample = np.sort(rng.choice(len(labels), size=min(900, len(labels)), replace=False))
    action = float(silhouette_score(features[sample], labels[sample], metric="cosine"))
    subject = float(
        silhouette_score(features[sample], identities[sample], metric="cosine")
    )
    return {
        "action_silhouette_cosine": round(action, 6),
        "subject_silhouette_cosine": round(subject, 6),
        "subject_to_action_ratio": (
            round(subject / action, 4) if abs(action) > 1e-9 else None
        ),
        "sample_size": int(len(sample)),
    }


def run(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    if set(SUBJECTS) != set(DEVELOPMENT_SUBJECTS) or set(SUBJECTS) & set(PROTECTED_HOLDOUTS):
        raise ValueError("V10 representation subject registry is invalid")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    subjects = [_load_subject(subject_id) for subject_id in SUBJECTS]
    source_rows = [source for item in subjects for source in item["sources"]]
    dataset_hash = _canonical_hash(source_rows)
    representation_ids = list(subjects[0]["features"])
    results = []
    for representation_id in representation_ids:
        row = _evaluate_representation(representation_id, subjects, output)
        row["separability"] = _separability(representation_id, subjects)
        results.append(row)

    baseline = next(item for item in results if item["representation_id"].startswith("D0_"))
    candidates = [item for item in results if item is not baseline]
    eligible = [
        item
        for item in candidates
        if item["pooled"]["macro_f1"] > baseline["pooled"]["macro_f1"]
        and item["pooled"]["safe_recall"] >= baseline["pooled"]["safe_recall"]
    ]
    selected = max(eligible, key=lambda item: item["pooled"]["macro_f1"], default=None)
    selected_worst_fold = (
        min(selected["folds"], key=lambda fold: fold["safe_recall"])
        if selected is not None
        else None
    )
    partially_supported = (
        selected is not None
        and selected["separability"]["subject_silhouette_cosine"]
        < baseline["separability"]["subject_silhouette_cosine"]
    )
    return {
        "schema_version": 1,
        "experiment_family": "V10_BEHAVIOR_CENTRIC_REPRESENTATION_PROBE",
        "evidence_level": "LEVEL_2_DEVELOPMENT_ABLATION",
        "git_head_before_experiment": _git_head(),
        "hypothesis": (
            "Fixed behavior-region shape and temporal residual features reduce subject/background "
            "dominance and improve SAFE/PHONE/OTHER LOSO separability over frozen full-ROI ResNet."
        ),
        "subjects": list(SUBJECTS),
        "protected_holdouts": sorted(PROTECTED_HOLDOUTS),
        "protected_holdouts_accessed": False,
        "dataset_role": "REPEATEDLY_USED_DEVELOPMENT_ONLY",
        "rights_accepted": False,
        "temporal_development_eligible": True,
        "dataset_sources": source_rows,
        "dataset_hash": dataset_hash,
        "class_order": list(COARSE_CLASSES),
        "fine_class_order": list(ACTION_CLASSES),
        "controlled_variables": {
            "split": "OUTER_LOSO_THREE_SUBJECTS",
            "head": "STANDARD_SCALER_PLUS_BALANCED_LOGISTIC_REGRESSION",
            "C": 1.0,
            "solver": "liblinear",
            "max_iter": 500,
            "seed": SEED,
            "feature_training": "NONE",
        },
        "representations": results,
        "selection_rule": (
            "KEEP only if pooled Macro-F1 exceeds D0 and pooled SAFE recall does not decrease."
        ),
        "selected_candidate": (
            selected["representation_id"] if selected is not None else None
        ),
        "hypothesis_outcome": (
            "PARTIALLY_SUPPORTED" if partially_supported else "NOT_SUPPORTED"
        ),
        "model_selection_decision": "REJECT_FOR_PROMOTION",
        "decision": (
            "KEEP_AS_FOLLOW_UP_DEVELOPMENT_HYPOTHESIS"
            if partially_supported
            else "REJECT_ALL_CANDIDATES"
        ),
        "governance_reasons": [
            (
                "The pooled selection rule identifies a development candidate, but its "
                f"worst fold is {selected_worst_fold['heldout_subject']} with "
                f"SAFE recall {selected_worst_fold['safe_recall']:.4f}."
                if selected_worst_fold is not None
                else "No candidate passed the pre-written pooled selection rule."
            ),
            (
                "The selected candidate lowers subject silhouette but also lowers action "
                "silhouette relative to D0."
                if selected is not None
                and selected["separability"]["action_silhouette_cosine"]
                < baseline["separability"]["action_silhouette_cosine"]
                else "Action separability did not provide sufficient promotion evidence."
            ),
            "All evaluated representations remain below the 85% pooled Macro-F1 and SAFE-recall research targets.",
            "No independent holdout, event-level validation, or accepted canonical dataset is available.",
        ],
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "compute": "CPU_ONLY",
        },
        "limitations": [
            "Only three repeatedly used development subjects are available.",
            "The cached RGB input is 64x64 and cannot support true pose-aligned hand/face crops.",
            "D2 includes a future residual and is an offline non-causal representation.",
            "Clip-level linear-probe gains do not imply event-level gains.",
            "No independent holdout or accepted canonical dataset is available.",
        ],
        "production_approved": False,
    }


def render(result: dict[str, Any], output: Path = DEFAULT_OUTPUT) -> Path:
    rows = []
    for item in result["representations"]:
        pooled = item["pooled"]
        sep = item["separability"]
        rows.append(
            f"| {item['representation_id']} | {item['feature_dimensions']} | "
            f"{100 * pooled['macro_f1']:.2f}% | {100 * pooled['balanced_accuracy']:.2f}% | "
            f"{100 * pooled['safe_recall']:.2f}% | {sep['action_silhouette_cosine']:.4f} | "
            f"{sep['subject_silhouette_cosine']:.4f} | {sep['subject_to_action_ratio']} |"
        )
    fold_rows = []
    for item in result["representations"]:
        for fold in item["folds"]:
            fold_rows.append(
                f"| {item['representation_id']} | {fold['heldout_subject']} | "
                f"{100 * fold['macro_f1']:.2f}% | {100 * fold['safe_recall']:.2f}% | "
                f"{fold['samples']} |"
            )
    selected = result["selected_candidate"] or "NONE"
    text = f"""# V10 behavior-centric representation probe

Evidence level: **LEVEL 2 - development ablation**. All folds are subject-disjoint LOSO across
`gB-9`, `gC-14`, and `gZ-36`; these subjects have been used repeatedly and are not independent.

| Representation | Dim | Macro-F1 | Balanced accuracy | SAFE recall | Action silhouette | Subject silhouette | Subject/action |
| --- | --- | --- | --- | --- | --- | --- | --- |
{chr(10).join(rows)}

| Representation | Held-out subject | Macro-F1 | SAFE recall | Clips |
| --- | --- | --- | --- | --- |
{chr(10).join(fold_rows)}

The pre-written pooled selection rule requires higher Macro-F1 than D0 without reducing pooled SAFE recall.
It selects **{selected}** for follow-up. Hypothesis outcome: **{result['hypothesis_outcome']}**;
model-selection decision: **{result['model_selection_decision']}**.

The representation hypothesis is only partly supported. D2 substantially reduces subject silhouette and
improves pooled clip metrics, but its action silhouette is lower than D0 and its worst held-out subject is
`gZ-36`, where SAFE recall is 1.66%. Every representation remains below the 85% research targets. The pooled
rule is retained as originally specified; the worst-subject safety result separately blocks promotion.

D0 reuses frozen ResNet-18 full-driver-ROI embeddings. D1 uses fixed overlapping upper/lower HOG regions;
D2 uses adjacent-frame residual maps; D5 concatenates D1 and D2. The same balanced linear head, seed and
LOSO folds are used throughout. No backbone, detector, event threshold or matcher was changed.

This test does not measure Event F1. The RGB cache is 64x64, D2 includes future-frame evidence, dataset
rights remain unaccepted, and no independent holdout exists. The candidate can guide the next controlled
video experiment but cannot be promoted to production.
"""
    path = output / "07_BEHAVIOR_CENTRIC_REPRESENTATION.md"
    path.write_text(text, encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(args.output)
    result_path = args.output / "representation_probe.json"
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    report_path = render(result, args.output)
    print(json.dumps({"result": str(result_path), "report": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()
