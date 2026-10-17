"""Run pre-registered V11 causal-motion and behavior-ROI development ablations."""

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
    silhouette_score,
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, normalize

from training.common import sha256_file
from training.dmd.behavior_roi import BehaviorRoiExtractor, Box, RoiStrategy
from training.dmd.causal_motion import CausalMotionConfig, CausalMotionEncoder
from training.dmd.v9_diagnostics import DEVELOPMENT_SUBJECTS, PROTECTED_HOLDOUTS

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "reports" / "v11_research"
CACHE_ROOT = ROOT / "dataset" / "external_dmd" / "derivatives"
SUBJECTS = ("gB-9", "gC-14", "gZ-36")
CLASSES = ("SAFE_DRIVING", "PHONE_USE", "OTHER_ACTIVITY")
SEED = 20261017
EXPERIMENT_PLAN = {
    "V11-A_FROZEN_RESNET": "Existing 512-D full-driver frozen ResNet representation",
    "V11-B_CAUSAL_LAG1": "Past-only t minus t-1 residual; one sample lag equals 0.504 s",
    "V11-C_CAUSAL_MULTISCALE": "Past-only residuals at 1, 3, 7 samples (0.504, 1.512, 3.528 s)",
    "V11-D_BEHAVIOR_MULTI_ROI": "Fixed face, torso, left-hand and right-hand ROI HOG",
    "V11-E_ROI_PLUS_CAUSAL": "V11-D appearance concatenated with V11-C causal motion",
}


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


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
    return cv2.HOGDescriptor((32, 32), (16, 16), (8, 8), (8, 8), 9)


def _multi_roi_hog(frames: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    extractor = BehaviorRoiExtractor(output_size=(32, 32), padding=0.12)
    hog = _hog_descriptor()
    features = []
    provenance: dict[str, Any] | None = None
    driver = Box(0, 0, frames.shape[2], frames.shape[1])
    for frame in frames:
        crops = extractor.extract(
            frame,
            driver,
            RoiStrategy.MULTI_REGION,
            source_camera="DMD_BODY_64_CACHE",
            driver_track_id=None,
        )
        if provenance is None:
            provenance = {
                "regions": [crop.provenance.as_dict() for crop in crops],
                "note": "Fixed camera-normalized regions; missing landmarks remain UNKNOWN.",
            }
        vectors = []
        for crop in crops:
            gray = cv2.cvtColor(crop.image, cv2.COLOR_RGB2GRAY)
            vectors.append(hog.compute(gray).ravel())
        features.append(np.concatenate(vectors))
    return np.asarray(features, dtype=np.float32), provenance or {}


def _load_subject(subject_id: str) -> dict[str, Any]:
    if subject_id not in DEVELOPMENT_SUBJECTS or subject_id in PROTECTED_HOLDOUTS:
        raise ValueError(f"subject is not eligible for V11 development: {subject_id}")
    rgb_path = CACHE_ROOT / "v5_rgb_cache" / f"v5_rgb_{subject_id}_64.npz"
    embedding_path = CACHE_ROOT / "v5_resnet18_embeddings" / f"v5_resnet18_{subject_id}.npz"
    with np.load(rgb_path, allow_pickle=False) as rgb:
        frames = rgb["frames"].copy()
        frame_indices = rgb["frame_indices"].astype(np.int64)
        timestamps = rgb["timestamps"].astype(np.float64)
        source_labels = rgb["labels"].astype(np.int64)
    with np.load(embedding_path, allow_pickle=False) as frozen:
        if not (
            np.allclose(timestamps, frozen["timestamps"], atol=1e-5)
            and np.array_equal(source_labels, frozen["labels"])
        ):
            raise ValueError(f"RGB and embedding caches are not aligned for {subject_id}")
        embeddings = frozen["embeddings"].astype(np.float32)
    lag1 = CausalMotionEncoder(
        CausalMotionConfig(lags=(1,), map_size=(16, 16), include_statistics=True)
    ).encode_sequence(frames)
    multiscale = CausalMotionEncoder(
        CausalMotionConfig(lags=(1, 3, 7), map_size=(12, 12), include_statistics=True)
    ).encode_sequence(frames)
    roi, roi_provenance = _multi_roi_hog(frames)
    features = {
        "V11-A_FROZEN_RESNET": embeddings,
        "V11-B_CAUSAL_LAG1": lag1,
        "V11-C_CAUSAL_MULTISCALE": multiscale,
        "V11-D_BEHAVIOR_MULTI_ROI": roi,
        "V11-E_ROI_PLUS_CAUSAL": np.concatenate((roi, multiscale), axis=1),
    }
    del frames
    return {
        "subject_id": subject_id,
        "frame_indices": frame_indices,
        "timestamps": timestamps,
        "labels": _coarse_labels(source_labels),
        "features": features,
        "roi_provenance": roi_provenance,
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


def _class_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
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
        for index, name in enumerate(CLASSES)
    }


def _metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    per_class = _class_metrics(labels, predictions)
    return {
        "samples": int(len(labels)),
        "macro_f1": round(float(f1_score(labels, predictions, average="macro")), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(labels, predictions)), 4),
        "safe_recall": per_class["SAFE_DRIVING"]["recall"],
        "phone_recall": per_class["PHONE_USE"]["recall"],
        "other_recall": per_class["OTHER_ACTIVITY"]["recall"],
        "confusion_matrix": confusion_matrix(labels, predictions, labels=np.arange(3)).tolist(),
        "per_class": per_class,
    }


def _evaluate(representation_id: str, subjects: list[dict[str, Any]], output: Path) -> dict[str, Any]:
    folds = []
    pooled_labels = []
    pooled_predictions = []
    for heldout in subjects:
        training = [row for row in subjects if row["subject_id"] != heldout["subject_id"]]
        train_x = np.concatenate([row["features"][representation_id] for row in training])
        train_y = np.concatenate([row["labels"] for row in training])
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
        prediction_path = output / "predictions" / f"{representation_id.lower()}_loso_{heldout['subject_id']}.npz"
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            prediction_path,
            frame_indices=heldout["frame_indices"][test_mask],
            timestamps=heldout["timestamps"][test_mask],
            labels=labels,
            predictions=predictions,
            probabilities=probabilities,
        )
        row = _metrics(labels, predictions)
        row.update(
            {
                "heldout_subject": heldout["subject_id"],
                "training_subjects": [item["subject_id"] for item in training],
                "fit_predict_seconds": round(elapsed, 3),
                "prediction_path": prediction_path.relative_to(ROOT).as_posix(),
                "prediction_sha256": sha256_file(prediction_path),
            }
        )
        folds.append(row)
        pooled_labels.append(labels)
        pooled_predictions.append(predictions)
    pooled = _metrics(np.concatenate(pooled_labels), np.concatenate(pooled_predictions))
    return {
        "representation_id": representation_id,
        "description": EXPERIMENT_PLAN.get(representation_id, representation_id),
        "feature_dimensions": int(subjects[0]["features"][representation_id].shape[1]),
        "folds": folds,
        "pooled": pooled,
        "worst_subject": {
            "macro_f1": min(fold["macro_f1"] for fold in folds),
            "safe_recall": min(fold["safe_recall"] for fold in folds),
            "phone_recall": min(fold["phone_recall"] for fold in folds),
            "other_recall": min(fold["other_recall"] for fold in folds),
        },
    }


def _separability(representation_id: str, subjects: list[dict[str, Any]]) -> dict[str, Any]:
    features = np.concatenate([row["features"][representation_id] for row in subjects])
    labels = np.concatenate([row["labels"] for row in subjects])
    identities = np.concatenate(
        [np.full(len(row["labels"]), index) for index, row in enumerate(subjects)]
    )
    valid = labels >= 0
    features = normalize(features[valid])
    labels = labels[valid]
    identities = identities[valid]
    rng = np.random.default_rng(SEED)
    sample = np.sort(rng.choice(len(labels), size=min(750, len(labels)), replace=False))
    action = float(silhouette_score(features[sample], labels[sample], metric="cosine"))
    subject = float(silhouette_score(features[sample], identities[sample], metric="cosine"))
    return {
        "action_silhouette_cosine": round(action, 6),
        "subject_silhouette_cosine": round(subject, 6),
        "subject_to_action_ratio": round(subject / action, 4) if abs(action) > 1e-9 else None,
        "sample_size": int(len(sample)),
    }


def run(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    if set(SUBJECTS) != set(DEVELOPMENT_SUBJECTS) or set(SUBJECTS) & set(PROTECTED_HOLDOUTS):
        raise ValueError("V11 subject registry violates holdout policy")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    extraction_started = time.perf_counter()
    subjects = [_load_subject(subject_id) for subject_id in SUBJECTS]
    extraction_seconds = time.perf_counter() - extraction_started
    results = []
    for representation_id in EXPERIMENT_PLAN:
        row = _evaluate(representation_id, subjects, output)
        row["separability"] = _separability(representation_id, subjects)
        results.append(row)
    baseline = results[0]
    candidates = results[1:]
    eligible = [
        row
        for row in candidates
        if row["pooled"]["macro_f1"] > baseline["pooled"]["macro_f1"]
        and row["worst_subject"]["safe_recall"] >= 0.20
        and row["worst_subject"]["macro_f1"] >= 0.20
    ]
    selected = max(
        eligible,
        key=lambda row: (row["worst_subject"]["safe_recall"], row["pooled"]["macro_f1"]),
        default=None,
    )
    sources = [source for row in subjects for source in row["sources"]]
    return {
        "schema_version": 1,
        "experiment_family": "V11_CAUSAL_MOTION_BEHAVIOR_ROI",
        "evidence_level": "LEVEL_2_REPEATED_DEVELOPMENT_ABLATION",
        "git_head_before_experiment": _git_head(),
        "hypothesis": "Past-only multi-scale motion and fixed behavior regions improve action discrimination without catastrophic worst-subject SAFE collapse.",
        "pre_registered_plan": EXPERIMENT_PLAN,
        "selection_rule": "Pooled Macro-F1 must exceed V11-A; worst-subject SAFE recall and Macro-F1 must each be at least 20%. Rank eligible candidates by worst-subject SAFE recall, then pooled Macro-F1.",
        "subjects": list(SUBJECTS),
        "protected_holdouts": sorted(PROTECTED_HOLDOUTS),
        "protected_holdouts_accessed": False,
        "dataset_role": "REPEATEDLY_USED_DEVELOPMENT_ONLY",
        "rights_accepted": False,
        "training_scope": "BALANCED_LINEAR_HEAD_ONLY_ON_FROZEN_OR_DETERMINISTIC_FEATURES",
        "dataset_sources": sources,
        "dataset_hash": _canonical_hash(sources),
        "class_order": list(CLASSES),
        "controlled_variables": {
            "split": "OUTER_LOSO_THREE_SUBJECTS",
            "head": "STANDARD_SCALER_PLUS_BALANCED_LOGISTIC_REGRESSION",
            "C": 1.0,
            "solver": "liblinear",
            "max_iter": 500,
            "seed": SEED,
        },
        "causal_contract": {
            "online_features_use_future_frames": False,
            "cache_sample_stride_frames": 15,
            "cache_sample_stride_seconds": 0.504,
            "multiscale_lags_samples": [1, 3, 7],
        },
        "roi_provenance_example": subjects[0]["roi_provenance"],
        "representations": results,
        "selected_candidate": selected["representation_id"] if selected else None,
        "decision": "KEEP_DEVELOPMENT_HYPOTHESIS" if selected else "REJECT_ALL_CANDIDATES",
        "promotion_decision": "REJECT",
        "runtime": {
            "feature_extraction_seconds": round(extraction_seconds, 3),
            "total_seconds": round(time.perf_counter() - started, 3),
            "environment": platform.platform(),
            "compute": "CPU_ONLY",
        },
        "limitations": [
            "Only three repeatedly used development subjects are available.",
            "The 64x64 cache cannot validate landmark-aligned hand or face crops.",
            "Fixed behavior zones are camera-normalized heuristics with UNKNOWN evidence state.",
            "Clip metrics do not establish Event F1.",
            "No independent holdout or accepted canonical dataset is available.",
        ],
        "production_approved": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(args.output)
    path = args.output / "representation_experiments.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"result": str(path), "decision": result["decision"]}, indent=2))


if __name__ == "__main__":
    main()
