import csv
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from PIL import Image

from backend.ai.auxiliary import PoseEvidence, phone_context
from backend.ai.detector import NormalizedDetection
from backend.ai.fusion import FusionFeatures, LogisticFusion
from backend.ai.sequence import TemporalFeatureBuffer
from training import download_base_weights
from training import v2_portable_runner as portable_runner
from training.apply_uncertain_decisions import apply as apply_uncertain
from training.audit_specialist_dataset import audit
from training.audit_v2_readiness import audit as audit_readiness
from training.build_adt_review_queue import build as build_adt_review_queue
from training.build_seatbelt_classifier import build as build_classifier
from training.build_seatbelt_v2 import build as build_seatbelt
from training.common import perceptual_hash
from training.convert_uncertain_review_decisions import convert as convert_uncertain_decisions
from training.evaluate_v2 import classification_report
from training.pretrain_review_v2 import review_item as pretrain_review_item
from training.train_fusion import train as train_fusion
from training.train_multimodel_v2 import (
    _build_recovery_archive,
    _restore_resume_backup,
    _write_resume_backup,
    train_component,
)
from training.v2_portable_runner import (
    _validate_yolo_label,
    enforce_readiness,
    runtime_config,
    select_profile,
    write_runtime_config,
)


def test_phone_context_uses_hand_or_face_geometry():
    phone = NormalizedDetection(0, "phone", 0.9, (40, 40, 60, 70), 1)
    handheld = PoseEvidence(0.9, ((50, 35),), ((55, 60),))
    mounted = PoseEvidence(0.9, ((5, 5),), ((5, 90),))

    assert phone_context(phone, handheld)[0] == "HANDHELD"
    assert phone_context(phone, mounted)[0] == "MOUNTED_OR_STATIC"
    assert phone_context(phone, PoseEvidence(0.0, (), ()))[0] == "UNKNOWN"


def test_proposal_reviewer_records_independent_phone_agreement_without_human_claim():
    item = {
        "sample_id": "phone-1",
        "source_group_id": "video-1",
        "annotations": [
            {
                "class_id": 0,
                "occupant_role": "PENDING",
                "yolo": [0.5, 0.5, 0.2, 0.2],
            }
        ],
    }
    quality = {
        "width": 100,
        "height": 100,
        "brightness": 120,
        "contrast": 45,
        "laplacian_variance": 100,
        "suggested_conditions": ["daylight_or_well_lit"],
    }
    result = pretrain_review_item(
        item,
        "phone",
        quality,
        [{"confidence": 0.9, "xyxy_normalized": [0.4, 0.4, 0.6, 0.6]}],
    )

    assert result["tier"] == "HIGH_CONFIDENCE"
    assert result["training_eligible"] is True
    assert result["governance_eligible"] is False
    assert result["reviewer_type"] == "AUTOMATED"


def test_proposal_reviewer_escalates_phone_found_in_proposed_negative():
    quality = {
        "width": 100,
        "height": 100,
        "brightness": 120,
        "contrast": 45,
        "laplacian_variance": 100,
        "suggested_conditions": ["daylight_or_well_lit"],
    }
    result = pretrain_review_item(
        {"sample_id": "negative-1", "source_group_id": "video-1", "annotations": []},
        "phone",
        quality,
        [{"confidence": 0.8, "xyxy_normalized": [0.4, 0.4, 0.6, 0.6]}],
    )

    assert result["decision"] == "UNCERTAIN"
    assert result["training_eligible"] is False
    assert result["evidence_level"] == "POSSIBLE_MISSING_PHONE_LABEL"


def test_temporal_feature_buffer_tracks_duration_and_smoothing():
    buffer = TemporalFeatureBuffer(window_seconds=2, alpha=0.5, positive_score=0.5)
    first = buffer.add(("vehicle", 1), 0.0, 0.8)
    second = buffer.add(("vehicle", 1), 1.0, 0.4)

    assert first.observations == 1
    assert second.duration_seconds == 1.0
    assert second.ratio == 0.5
    assert second.smoothed_score == pytest.approx(0.6)


def test_fusion_fallback_penalizes_fastened_classifier_contradiction():
    fusion = LogisticFusion(None, threshold=0.5)
    agreement = fusion.predict(
        FusionFeatures(
            detector_confidence=0.9,
            classifier_unfastened=0.9,
            classifier_fastened=0.1,
            vehicle_context=1.0,
        )
    )
    contradiction = fusion.predict(
        FusionFeatures(
            detector_confidence=0.9,
            classifier_unfastened=0.1,
            classifier_fastened=0.9,
            vehicle_context=1.0,
        )
    )

    assert agreement.decision is True
    assert contradiction.decision is False
    assert agreement.score > contradiction.score


def _source_dataset(root: Path) -> Path:
    records = []
    for split_index, split in enumerate(("train", "val", "test")):
        for source_class in (1, 2):
            sample_id = f"{split}_{source_class}"
            image_dir = root / "images" / split
            label_dir = root / "labels" / split
            image_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            image_path = image_dir / f"{sample_id}.jpg"
            Image.new("RGB", (64, 64), (80 + source_class * 30, 70 + split_index * 30, 100)).save(
                image_path
            )
            (label_dir / f"{sample_id}.txt").write_text(
                f"{source_class} 0.5 0.5 0.5 0.6\n", encoding="utf-8"
            )
            digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
            records.append(
                {
                    "sample_id": sample_id,
                    "split": split,
                    "source_id": "roboflow_seatbelttraining_v4",
                    "source_group_id": f"group:{split}:{source_class}",
                    "effective_group_id": f"effective:{split}:{source_class}",
                    "near_cluster_id": f"near:{split}:{source_class}",
                    "sha256": digest,
                    "phash": perceptual_hash(image_path),
                    "class_counts": {str(source_class): 1},
                    "human_review_status": "PENDING",
                    "usage": "PROPOSAL_MODEL_ONLY",
                }
            )
    (root / "manifest.jsonl").write_text(
        "\n".join(json.dumps(item) for item in records) + "\n", encoding="utf-8"
    )
    return root


def test_v2_seatbelt_builder_uses_independent_images_without_duplication(tmp_path):
    source = _source_dataset(tmp_path / "source")
    output = tmp_path / "seatbelt_v2"

    report = build_seatbelt(source, output)
    audited = audit(output)

    assert report["images"] == {"train": 2, "val": 2, "test": 2}
    assert report["positive_images"]["train"] == {"occupant_upper_body": 2}
    assert report["augmentation"] == "NONE_NO_DUPLICATE_OVERSAMPLING"
    data = yaml.safe_load((output / "data.yaml").read_text(encoding="utf-8"))
    assert data["nc"] == len(data["names"]) == 1
    assert audited["status"] == "PASS"
    assert not list(output.rglob("*_v2_*"))


def test_classifier_builder_inherits_source_splits(tmp_path):
    source = _source_dataset(tmp_path / "source")
    specialist = tmp_path / "seatbelt_v2"
    build_seatbelt(source, specialist)

    report = build_classifier(specialist, tmp_path / "classifier")

    assert report["counts"]["train"] == {
        "seatbelt_fastened": 1,
        "seatbelt_unfastened": 1,
        "uncertain_or_occluded": 0,
    }
    assert report["split_inheritance"] == "SOURCE_SPLITS_PRESERVED"
    assert report["ready_for_training"] is False


def test_classifier_requires_reviewed_uncertain_examples_in_every_split(tmp_path):
    source = _source_dataset(tmp_path / "source")
    specialist = tmp_path / "seatbelt_v2"
    build_seatbelt(source, specialist)
    uncertain = tmp_path / "uncertain.jsonl"
    records = []
    for split in ("train", "val", "test"):
        records.append(
            {
                "sample_id": f"uncertain_{split}",
                "split": split,
                "image_path": str((source / "images" / split / f"{split}_1.jpg").resolve()),
                "yolo": [0.5, 0.5, 0.5, 0.6],
                "source_group_id": f"uncertain-group:{split}",
                "effective_group_id": f"uncertain-effective:{split}",
            }
        )
    uncertain.write_text("\n".join(json.dumps(item) for item in records) + "\n", encoding="utf-8")

    report = build_classifier(specialist, tmp_path / "classifier", uncertain_manifest=uncertain)

    assert report["ready_for_training"] is True
    assert report["counts"]["val"]["uncertain_or_occluded"] == 1


def test_fusion_training_uses_validation_only_for_threshold(tmp_path):
    path = tmp_path / "fusion.csv"
    fieldnames = ["split", "label", *FusionFeatures().as_dict()]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for split, count in (("train", 40), ("val", 20), ("test", 10)):
            for index in range(count):
                label = index % 2
                score = 0.85 if label else 0.15
                writer.writerow(
                    {
                        "split": split,
                        "label": label,
                        "detector_confidence": score,
                        "classifier_unfastened": score,
                        "classifier_fastened": 1 - score,
                        "phone_hand_proximity": 0,
                        "phone_face_proximity": 0,
                        "pose_confidence": 0,
                        "temporal_ratio": score,
                        "driver_role": 1,
                        "vehicle_context": 1,
                    }
                )
    output = tmp_path / "fusion.json"

    artifact = train_fusion(path, output, "NO_SEATBELT", epochs=300)
    runtime = LogisticFusion(output)
    result = runtime.predict(
        FusionFeatures(
            detector_confidence=0.9,
            classifier_unfastened=0.9,
            classifier_fastened=0.1,
            temporal_ratio=0.9,
            driver_role=1,
            vehicle_context=1,
        )
    )

    assert artifact["test_rows_used"] == 0
    assert artifact["validation"]["f1"] == 1.0
    assert runtime.available is True
    assert result.decision is True


def test_uncertain_decision_keeps_source_box_identity(tmp_path):
    queue = tmp_path / "queue.json"
    queue.write_text(
        json.dumps(
            {
                "selected": [
                    {
                        "candidate_id": "frame-1:box:2",
                        "sample_id": "frame-1",
                        "box_index": 2,
                        "split": "train",
                        "image_path": "frame.jpg",
                        "sha256": "a" * 64,
                        "yolo": [0.5, 0.5, 0.4, 0.5],
                        "source_group_id": "video-1",
                        "effective_group_id": "near-1",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    decisions = tmp_path / "decisions.jsonl"
    decisions.write_text(
        json.dumps(
            {
                "candidate_id": "frame-1:box:2",
                "decision": "UNCERTAIN_OR_OCCLUDED",
                "reviewer": "reviewer@example.org",
                "reviewed_at": "2026-09-01T00:00:00Z",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = apply_uncertain(queue, decisions, tmp_path / "uncertain.jsonl")
    record = json.loads((tmp_path / "uncertain.jsonl").read_text(encoding="utf-8"))

    assert result["uncertain_accepted"] == 1
    assert record["source_sample_id"] == "frame-1"
    assert record["source_box_index"] == 2


def test_v2_classification_report_exposes_uncertain_recall():
    report = classification_report(
        [
            ("seatbelt_fastened", "seatbelt_fastened"),
            ("seatbelt_unfastened", "uncertain_or_occluded"),
            ("uncertain_or_occluded", "uncertain_or_occluded"),
        ],
        ["seatbelt_fastened", "seatbelt_unfastened", "uncertain_or_occluded"],
    )

    assert report["classes"]["uncertain_or_occluded"]["recall"] == 1.0
    assert report["classes"]["seatbelt_unfastened"]["recall"] == 0.0


def test_v2_gpu_profiles_are_deterministic_and_smoke_isolated(tmp_path):
    profiles = yaml.safe_load(
        Path("experiments/MULTIMODEL_V2/profiles.yaml").read_text(encoding="utf-8")
    )
    assert select_profile(profiles, "auto", 7.95) == "rtx5060ti_8gb"
    assert select_profile(profiles, "auto", 15.9) == "rtx5060ti_16gb"
    with pytest.raises(ValueError, match="needs at least"):
        select_profile(profiles, "rtx5060ti_16gb", 7.95)

    base = {
        "experiment_id": "MULTIMODEL_V2",
        "components": {
            "phone_detector": {"epochs": 100, "batch": 8, "multi_scale": True},
            "seatbelt_detector": {"epochs": 160, "batch": 6, "multi_scale": True},
        },
    }
    runtime = runtime_config(
        base,
        profiles,
        "rtx5060ti_8gb",
        {"seatbelt_detector"},
        smoke=True,
    )
    assert set(runtime["components"]) == {"seatbelt_detector"}
    assert runtime["components"]["seatbelt_detector"] == {
        "epochs": 1,
        "batch": 1,
        "multi_scale": False,
        "workers": 2,
        "patience": 1,
        "save_period": 1,
        "cache": False,
    }
    assert runtime["runtime_policy"]["test_split_used"] is False

    working = tmp_path / "runtime"
    path = write_runtime_config(working, runtime)
    assert write_runtime_config(working, runtime) == path
    changed = json.loads(json.dumps(runtime))
    changed["components"]["seatbelt_detector"]["batch"] = 2
    with pytest.raises(ValueError, match="different runtime config"):
        write_runtime_config(working, changed)


def test_v2_base_config_uses_vram_safe_defaults():
    config = yaml.safe_load(
        Path("experiments/MULTIMODEL_V2/config.yaml").read_text(encoding="utf-8")
    )
    phone = config["components"]["phone_detector"]
    seatbelt = config["components"]["seatbelt_detector"]
    assert phone["batch"] == 2
    assert phone["multi_scale"] is False
    assert seatbelt["batch"] == 1
    assert seatbelt["multi_scale"] is False


def test_v2_governed_readiness_is_default_gate():
    report = {
        "readiness": {
            "governance_status": {
                "proposal_detector_training_ready": True,
                "governed_training_ready": False,
            },
            "governance_blocker_codes": ["SEMANTIC_REVIEW_PENDING"],
        }
    }
    enforce_readiness(report, "proposal")
    with pytest.raises(ValueError, match="SEMANTIC_REVIEW_PENDING"):
        enforce_readiness(report, "governed")


def test_v2_preflight_only_still_enforces_governed_gate(monkeypatch):
    report = {
        "readiness": {
            "governance_status": {
                "proposal_detector_training_ready": True,
                "governed_training_ready": False,
            },
            "governance_blocker_codes": ["SEMANTIC_REVIEW_PENDING"],
        }
    }
    monkeypatch.setattr(portable_runner, "preflight", lambda _bundle: report)
    monkeypatch.setattr(sys, "argv", ["runner", "--preflight-only"])
    with pytest.raises(ValueError, match="SEMANTIC_REVIEW_PENDING"):
        portable_runner.main()


def test_adt_review_builder_groups_variants_and_never_auto_admits(tmp_path):
    source = tmp_path / "adt"
    images = source / "train" / "images"
    labels = source / "train" / "labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    (source / "data.yaml").write_text(
        "names: ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'mobile', "
        "'x8', 'person-noseatbelt', 'person-seatbelt', 'seatbelt']\n",
        encoding="utf-8",
    )
    for suffix in ("a" * 32, "b" * 32):
        stem = f"frame_jpg.rf.{suffix}"
        Image.new("RGB", (32, 32), "gray").save(images / f"{stem}.jpg")
        (labels / f"{stem}.txt").write_text("10 0.5 0.95 0.8 0.2\n", encoding="utf-8")
    queue = tmp_path / "queue.json"
    ingest = tmp_path / "ingest.jsonl"
    report_path = tmp_path / "report.json"

    report = build_adt_review_queue(source, queue, ingest, report_path)
    items = json.loads(queue.read_text(encoding="utf-8"))

    assert report["source_images"] == 2
    assert report["review_representatives"] == 1
    assert report["automatic_training_admission"] is False
    assert report["images_with_source_label_errors"] == 1
    assert items[0]["annotations"][0]["source_box_clipped"] is True


def test_uncertain_reviewer_conversion_preserves_audit_identity(tmp_path):
    queue = tmp_path / "queue.json"
    decisions = tmp_path / "decisions.jsonl"
    output = tmp_path / "converted.jsonl"
    queue.write_text(
        json.dumps([{"sample_id": "ui-1", "candidate_id": "source:box:0"}]),
        encoding="utf-8",
    )
    decisions.write_text(
        json.dumps(
            {
                "sample_id": "ui-1",
                "status": "UNCERTAIN",
                "reviewed_at_utc": "2026-09-01T00:00:00+00:00",
                "annotations": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = convert_uncertain_decisions(queue, decisions, output, "reviewer@example.org")
    row = json.loads(output.read_text(encoding="utf-8"))

    assert report["converted_decisions"] == 1
    assert row["candidate_id"] == "source:box:0"
    assert row["decision"] == "UNCERTAIN_OR_OCCLUDED"
    assert row["reviewer"] == "reviewer@example.org"


def test_current_v2_readiness_distinguishes_proposal_from_governed_training():
    report = audit_readiness()
    assert report["status"] == {
        "proposal_detector_training_ready": True,
        "governed_training_ready": False,
        "production_activation_ready": False,
    }
    assert report["datasets"]["phone"]["sources"] == {"kaggle_habbas_dms_v1": 9728}
    assert report["datasets"]["phone"]["zero_box_samples"] == 7513
    assert report["datasets"]["seatbelt"]["zero_box_samples"] == 0
    assert (
        report["candidate_external_sources"]["mendeley_driver_risk_behavior_v1"][
            "selected_phone_behavior_frames"
        ]
        == 334
    )
    assert "SEATBELT_CLASSIFIER_NOT_READY" in report["blocker_codes"]


def test_v2_recovery_archive_is_idempotent_and_leaves_no_staging_dir(tmp_path):
    working = tmp_path / "working"
    working.mkdir()
    config = tmp_path / "config.yaml"
    manifest = working / "EXP_MANIFEST.json"
    config.write_text("experiment_id: EXP\n", encoding="utf-8")
    manifest.write_text('{"experiment_id": "EXP"}\n', encoding="utf-8")
    run = working / "EXP_detector"
    (run / "weights").mkdir(parents=True)
    for path, content in (
        (run / "weights" / "best.pt", b"best"),
        (run / "weights" / "last.pt", b"last"),
    ):
        path.write_bytes(content)
    for path in (run / "results.csv", run / "args.yaml", run / "v2_component_report.json"):
        path.write_text("artifact\n", encoding="utf-8")
    plan = working / "EXP_detector_PLAN.json"
    plan.write_text("{}\n", encoding="utf-8")
    report = {
        "best_weights": str(run / "weights" / "best.pt"),
        "run": str(run),
        "plan": str(plan),
    }

    first = _build_recovery_archive(working, "EXP", config, manifest, {"detector": report})
    second = _build_recovery_archive(working, "EXP", config, manifest, {"detector": report})

    assert first == second
    assert first.is_file()
    assert not (working / "EXP_RECOVERY").exists()
    assert not list(working.glob("v2-recovery-*"))
    with zipfile.ZipFile(first) as archive:
        assert "detector/best.pt" in archive.namelist()
        assert "detector/last.pt" in archive.namelist()


def test_v2_external_resume_backup_can_restore_a_lost_working_run(tmp_path):
    run = tmp_path / "working" / "EXP_detector"
    (run / "weights").mkdir(parents=True)
    (run / "weights" / "last.pt").write_bytes(b"optimizer-checkpoint")
    (run / "weights" / "best.pt").write_bytes(b"best-checkpoint")
    (run / "results.csv").write_text("epoch,metric\n0,0.5\n", encoding="utf-8")
    (run / "args.yaml").write_text("epochs: 3\n", encoding="utf-8")
    plan = tmp_path / "working" / "EXP_detector_PLAN.json"
    plan.write_text('{"fingerprint": "plan-1"}\n', encoding="utf-8")
    backup = tmp_path / "second-disk"

    target = _write_resume_backup(run, backup, "EXP_detector", plan, "plan-1", 1)
    shutil.rmtree(run)
    restored = _restore_resume_backup(backup, run, "EXP_detector", "plan-1")

    assert restored is True
    assert (run / "weights" / "last.pt").read_bytes() == b"optimizer-checkpoint"
    assert (run / "weights" / "best.pt").read_bytes() == b"best-checkpoint"
    assert (
        json.loads((target / "resume_metadata.json").read_text(encoding="utf-8"))[
            "completed_epochs"
        ]
        == 1
    )

    shutil.rmtree(run)
    with pytest.raises(ValueError, match="plan mismatch"):
        _restore_resume_backup(backup, run, "EXP_detector", "different-plan")


def test_v2_component_resume_requires_matching_plan(tmp_path, monkeypatch):
    class FakeYOLO:
        fail_once = True
        calls: list[dict] = []

        def __init__(self, source):
            self.source = Path(source)
            self.ckpt = {"epoch": 0} if self.source.name == "last.pt" else {}

        def train(self, **kwargs):
            self.calls.append(kwargs)
            if kwargs.get("resume"):
                run = self.source.parents[1]
            else:
                run = Path(kwargs["project"]) / kwargs["name"]
            (run / "weights").mkdir(parents=True, exist_ok=True)
            (run / "weights" / "best.pt").write_bytes(b"best")
            (run / "weights" / "last.pt").write_bytes(b"last")
            (run / "results.csv").write_text("epoch,metric\n0,0.5\n", encoding="utf-8")
            if type(self).fail_once:
                type(self).fail_once = False
                raise RuntimeError("simulated interruption")

        def add_callback(self, _name, _callback):
            return None

        def val(self, **_kwargs):
            return SimpleNamespace(results_dict={"metrics/mAP50(B)": 0.5})

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeYOLO))
    data = tmp_path / "dataset" / "data.yaml"
    data.parent.mkdir()
    data.write_text("path: .\ntrain: images/train\nval: images/val\nnames: {0: x}\n")
    working = tmp_path / "working"
    working.mkdir()
    config = {
        "task": "detect",
        "model": "models/base/yolo11s.pt",
        "data": str(data),
        "epochs": 3,
        "imgsz": 64,
        "device": 0,
    }

    with pytest.raises(RuntimeError, match="simulated interruption"):
        train_component("detector", config, "EXP", 42, working)
    report = train_component("detector", config, "EXP", 42, working, resume=True)
    assert report["resumed"] is True
    assert report["test_split_used"] is False
    assert FakeYOLO.calls[-1] == {"resume": True}

    reused = train_component("detector", config, "EXP", 42, working, resume=True)
    assert reused["reused_completed"] is True
    changed = {**config, "epochs": 4}
    with pytest.raises(ValueError, match="resume plan mismatch"):
        train_component("detector", changed, "EXP", 42, working, resume=True)


def test_v2_portable_preflight_rejects_nonfinite_labels(tmp_path):
    label = tmp_path / "bad.txt"
    label.write_text("0 nan 0.5 0.2 0.2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite box"):
        _validate_yolo_label(label, 1)


def test_base_weight_downloader_verifies_existing_artifact(tmp_path, monkeypatch):
    payload = b"governed-weight"
    path = tmp_path / "yolo11s.pt"
    path.write_bytes(payload)
    monkeypatch.setattr(download_base_weights, "EXPECTED_BYTES", len(payload))
    monkeypatch.setattr(
        download_base_weights,
        "EXPECTED_SHA256",
        hashlib.sha256(payload).hexdigest(),
    )
    assert download_base_weights.download(path)["bytes"] == len(payload)
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="does not match provenance"):
        download_base_weights.download(path)
