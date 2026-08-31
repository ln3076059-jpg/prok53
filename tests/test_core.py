import hashlib
import json
import zipfile
from pathlib import Path

import pytest
import yaml
from PIL import Image
from unittest.mock import MagicMock, patch

from backend.ai.association import associate_occupant, intersection_over_box
from backend.ai.detector import NormalizedDetection
from backend.ai.events import Observation, TemporalEventEngine
from training.common import hamming_distance, parse_yolo_line
from training.apply_review_decisions import apply as apply_review_decisions
from training.build_review_queue import map_suggestions, map_suggestions_from_label
from training.ingest_dataset import read_source_labels, source_group_id, source_label_path
from training.import_phone_bootstrap_recovery import EXPECTED_SPLITS, install_recovery
from training.kaggle_runner import preflight
from training.merge_phone_proposals import merge
from training.propose_phone_boxes import crop_xyxy_to_full_yolo, expanded_crop
from training.build_phone_bootstrap import split_for_group
from training.build_mc_bootstrap import BKTree, clip_yolo_box, cluster, multi_index_near_pairs
from training.mc_bootstrap_kaggle_runner import inspect_resume_archive, restore_resume_archive
from training.split_dataset import assign_groups, validate_isolation
from training.download_source import run_download


def test_yolo_label_parser_accepts_canonical_box():
    label = parse_yolo_line("0 0.5 0.5 0.2 0.3")
    assert label.class_id == 0


@pytest.mark.parametrize("line", ["3 0.5 0.5 0.2 0.2", "1 1.1 0.5 0.2 0.2", "2 0.5 0.5 0 0.2", "0 0.5"])
def test_yolo_label_parser_rejects_invalid_lines(line):
    with pytest.raises(ValueError):
        parse_yolo_line(line)


def test_near_duplicate_distance():
    assert hamming_distance("0000000000000000", "0000000000000001") == 1


def test_mendeley_sequence_frames_share_a_group():
    first = Path("ce8291e3-2691-44cd-ac9f-e0558d819eba_173.jpg")
    second = Path("ce8291e3-2691-44cd-ac9f-e0558d819eba_174.jpg")
    assert source_group_id(first, "risk") == source_group_id(second, "risk")
    assert source_group_id(first, "risk") == "risk:ce8291e3-2691-44cd-ac9f-e0558d819eba"


def test_phone_crop_box_maps_back_to_full_image():
    assert expanded_crop((100, 50, 300, 250), 1000, 500, 0.1) == (80, 30, 320, 270)
    assert crop_xyxy_to_full_yolo((80, 30, 320, 270), (20, 20, 120, 120), 1000, 500) == [
        0.15,
        0.2,
        0.1,
        0.2,
    ]


def test_phone_bootstrap_group_split_is_stable():
    assert split_for_group("clip-42") == split_for_group("clip-42")
    assert split_for_group("clip-42") in {"train", "val", "test"}


def test_bk_tree_finds_near_perceptual_hashes():
    tree = BKTree()
    tree.add(int("0000000000000000", 16), 0)
    tree.add(int("ffffffffffffffff", 16), 1)

    assert tree.query(int("0000000000000003", 16), 2) == [0]


def test_multi_index_hashing_finds_all_near_pairs_without_far_pairs():
    values = [
        int("0000000000000000", 16),
        int("0000000000000003", 16),
        int("ffffffffffffffff", 16),
    ]

    assert multi_index_near_pairs(values, threshold=2) == [(0, 1)]


def test_multi_index_hashing_matches_bruteforce():
    values = [
        0x0000000000000000,
        0x000000000000000F,
        0x000000000000001F,
        0x1111111111111111,
        0x1111111111111110,
        0xFFFFFFFFFFFFFFFF,
    ]
    expected = [
        (left, right)
        for right in range(len(values))
        for left in range(right)
        if (values[left] ^ values[right]).bit_count() <= 4
    ]

    assert sorted(multi_index_near_pairs(values, threshold=4)) == sorted(expected)


def test_mc_bootstrap_clips_source_box_to_image_bounds():
    clipped, changed = clip_yolo_box([0.95, 0.5, 0.2, 0.4])

    assert changed is True
    assert clipped == pytest.approx([0.9249999995, 0.5, 0.149999999, 0.4])
    parse_yolo_line("0 " + " ".join(str(value) for value in clipped))


def test_mc_bootstrap_clustering_merges_groups_and_near_hashes():
    records = [
        {"sample_id": "a", "base_group_id": "g1", "sha256": "a", "phash": "0000000000000000"},
        {"sample_id": "b", "base_group_id": "g1", "sha256": "b", "phash": "ffffffffffffffff"},
        {"sample_id": "c", "base_group_id": "g2", "sha256": "c", "phash": "0000000000000001"},
    ]

    component_ids, metadata = cluster(records, near_threshold=1)

    assert len(set(component_ids.values())) == 1
    assert metadata["components"] == 1


def _mc_resume_archive(tmp_path: Path, epoch_completed: int = 17) -> tuple[Path, dict]:
    last = b"resumable-last-checkpoint"
    best = b"best-checkpoint"
    preflight_report = {
        "config_sha256": "c" * 64,
        "manifest_sha256": "m" * 64,
        "audit_manifest_sha256": "a" * 64,
        "expected_hashes_sha256": "e" * 64,
    }
    metadata = {
        "schema_version": 1,
        "experiment_id": "MC_BOOTSTRAP_001",
        "usage": "PROPOSAL_MODEL_ONLY",
        "status": "PROPOSAL_MODEL_ONLY_NOT_GOVERNED_FINAL_DATA",
        "epoch_completed": epoch_completed,
        "total_epochs": 150,
        "last_weights_sha256": hashlib.sha256(last).hexdigest(),
        "best_weights_sha256": hashlib.sha256(best).hexdigest(),
        **preflight_report,
    }
    archive_path = tmp_path / "MC_BOOTSTRAP_001_LATEST_RESUME.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("weights/last.pt", last)
        archive.writestr("weights/best.pt", best)
        archive.writestr("results.csv", "epoch,metric\n0,0.1\n")
        archive.writestr("args.yaml", "epochs: 150\n")
        archive.writestr("resume_metadata.json", json.dumps(metadata))
    return archive_path, preflight_report


def test_mc_resume_archive_is_verified_and_restored(tmp_path):
    archive, preflight_report = _mc_resume_archive(tmp_path)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "config.yaml").write_text("epochs: 150\n", encoding="utf-8")

    metadata = inspect_resume_archive(archive, preflight_report, total_epochs=150)
    checkpoint, restored = restore_resume_archive(
        [archive], bundle, tmp_path / "working", preflight_report
    )

    assert metadata["epoch_completed"] == 17
    assert restored["epoch_completed"] == 17
    assert checkpoint.read_bytes() == b"resumable-last-checkpoint"
    assert (tmp_path / "working" / "MC_BOOTSTRAP_001" / "results.csv").is_file()


def test_mc_resume_archive_rejects_unsafe_member(tmp_path):
    archive, preflight_report = _mc_resume_archive(tmp_path)
    with zipfile.ZipFile(archive, "a") as handle:
        handle.writestr("../escape.pt", b"unsafe")

    with pytest.raises(ValueError, match="unsafe resume archive path"):
        inspect_resume_archive(archive, preflight_report, total_epochs=150)


def test_mc_resume_accepts_kaggle_extracted_layout(tmp_path):
    archive, preflight_report = _mc_resume_archive(tmp_path)
    extracted = tmp_path / "kaggle_input_resume"
    with zipfile.ZipFile(archive) as handle:
        handle.extractall(extracted)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "config.yaml").write_text("epochs: 150\n", encoding="utf-8")

    checkpoint, metadata = restore_resume_archive(
        [extracted], bundle, tmp_path / "working", preflight_report
    )

    assert metadata["epoch_completed"] == 17
    assert checkpoint.read_bytes() == b"resumable-last-checkpoint"


def _phone_recovery_archive(tmp_path: Path, weight: bytes = b"phone-model") -> Path:
    digest = hashlib.sha256(weight).hexdigest()
    metrics = {"metrics/mAP50(B)": 0.75}
    metadata = {
        "experiment_id": "PHONE_BOOTSTRAP_001",
        "usage": "PROPOSAL_MODEL_ONLY",
        "best_weights_sha256": digest,
        "preflight": {"manifest_samples": 9728, "splits": EXPECTED_SPLITS},
        "test_metrics": metrics,
    }
    archive = tmp_path / "PHONE_BOOTSTRAP_001_RECOVERY.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("weights/best.pt", weight)
        handle.writestr("run_metadata.json", json.dumps(metadata))
        handle.writestr("test_metrics.json", json.dumps(metrics))
    return archive


def test_phone_recovery_is_verified_and_installed_as_proposal_only(tmp_path):
    archive = _phone_recovery_archive(tmp_path)
    result = install_recovery(archive, tmp_path / "models")

    assert result["usage"] == "PROPOSAL_MODEL_ONLY"
    assert (tmp_path / "models" / "best.pt").read_bytes() == b"phone-model"


def test_phone_recovery_rejects_path_traversal(tmp_path):
    archive = _phone_recovery_archive(tmp_path)
    with zipfile.ZipFile(archive, "a") as handle:
        handle.writestr("../escape.txt", "unsafe")

    with pytest.raises(ValueError, match="unsafe archive path"):
        install_recovery(archive, tmp_path / "models")


def test_phone_proposals_merge_into_behavior_subset_without_auto_approval():
    queue = [
        {"sample_id": "phone-frame", "source_asset_id": "a.jpg", "annotations": [], "review_tasks": []},
        {"sample_id": "neutral-frame", "source_asset_id": "b.jpg", "annotations": [], "review_tasks": []},
    ]
    report = {
        "generator": {"model": "models/proposals/phone_bootstrap_001/best.pt"},
        "samples": [
            {
                "sample_id": "phone-frame",
                "source_asset_id": "a.jpg",
                "proposals": [
                    {"class_id": 0, "class_name": "phone", "confidence": 0.8, "yolo": [0.5, 0.5, 0.2, 0.2]}
                ],
            }
        ],
    }

    result = merge(queue, report)

    assert [item["sample_id"] for item in result] == ["phone-frame"]
    assert result[0]["annotations"][0]["requires_review"] is True
    assert result[0]["annotations"][0]["review_status"] == "PENDING"
    assert result[0]["annotations"][0]["occupant_role"] == "PENDING"
    assert result[0]["occupant_role"] == "PENDING"


def test_review_materialization_preserves_box_roles(tmp_path):
    image = tmp_path / "incoming.jpg"
    Image.new("RGB", (16, 16), "gray").save(image)
    record = {
        "sample_id": "sample-1",
        "status": "PENDING",
        "ingested_path": str(image),
        "source_group_id": "source:clip-1",
    }
    decision = {
        "sample_id": "sample-1",
        "status": "APPROVED",
        "vehicle_context_id": "vehicle-1",
        "occupant_role": "driver",
        "annotations": [
            {"class_id": 0, "yolo": [0.5, 0.5, 0.2, 0.2], "occupant_role": "front_passenger"}
        ],
    }

    result = apply_review_decisions([record], {"sample-1": decision}, tmp_path / "reviewed")

    assert result[0]["occupant_roles"] == ["front_passenger"]
    assert result[0]["reviewed_annotations"][0]["occupant_role"] == "front_passenger"


def test_review_materialization_rejects_unresolved_box_role(tmp_path):
    record = {
        "sample_id": "sample-1",
        "status": "PENDING",
        "ingested_path": str(tmp_path / "unused.jpg"),
        "source_group_id": "source:clip-1",
    }
    decision = {
        "status": "APPROVED",
        "vehicle_context_id": "vehicle-1",
        "occupant_role": "driver",
        "annotations": [{"class_id": 0, "yolo": [0.5, 0.5, 0.2, 0.2], "occupant_role": "PENDING"}],
    }

    with pytest.raises(ValueError, match="unresolved annotation roles"):
        apply_review_decisions([record], {"sample-1": decision}, tmp_path / "reviewed")


def test_roboflow_augmentation_pair_shares_a_group():
    base = Path("00013_jpg.rf.aaaaaaaa.jpg")
    augmented = Path("00013_a_jpg.rf.bbbbbbbb.jpg")
    assert source_group_id(base, "dms") == source_group_id(augmented, "dms")
    assert source_group_id(base, "dms") == "dms:GROUP_REVIEW_REQUIRED:00013"


def test_invalid_source_box_becomes_rejected_review_item(tmp_path):
    label = tmp_path / "invalid.txt"
    label.write_text("3 0.1 0.1 0 0\n", encoding="utf-8")
    annotations, rejected = map_suggestions_from_label(
        label, {3: "Cellphone Use"}, {"cellphone_use": {"canonical": None, "reason": "behavior"}}
    )
    assert annotations == []
    assert rejected[0]["source_class_name"] == "INVALID_SOURCE_LABEL"


def test_custom_synthetic_grouping_keeps_scene_variants_together():
    pattern = r"^(?P<clip>.+)_\d+$"
    assert source_group_id(Path("B_L_1_0_0.png"), "synthetic", pattern) == "synthetic:B_L_1_0"
    assert source_group_id(Path("B_L_1_0_9.png"), "synthetic", pattern) == "synthetic:B_L_1_0"


def test_group_assignment_never_splits_group():
    records = [{"sample_id": str(i), "review_status": "APPROVED", "source_group_id": f"g{i // 3}"} for i in range(30)]
    assignments = assign_groups(records)
    assert set(assignments) == {f"g{i}" for i in range(10)}


def test_isolation_reports_overlap():
    report = validate_isolation([
        {"split": "train", "sha256": "same", "effective_group_id": "a", "source_group_id": "a"},
        {"split": "test", "sha256": "same", "effective_group_id": "b", "source_group_id": "b"},
    ])
    assert report["status"] == "FAIL"


def test_driver_intersection():
    assert intersection_over_box((0, 0, 100, 100), (0, 0, 50, 100)) == 0.5


def test_occupant_association_selects_best_region():
    detection = NormalizedDetection(0, "phone", 0.9, (70, 20, 90, 80), 1)
    regions = [
        {"role": "driver", "normalized_xyxy": [0, 0, 0.5, 1], "min_overlap": 0.5},
        {"role": "front_passenger", "normalized_xyxy": [0.5, 0, 1, 1], "min_overlap": 0.5},
    ]
    assert associate_occupant(detection, 100, 100, regions) == "front_passenger"


def test_phone_requires_driver_association_and_temporal_persistence():
    engine = TemporalEventEngine(window_seconds=2, min_positive_seconds=1, min_observations=3, cooldown_seconds=5)
    assert engine.add(Observation(0, "phone", 0.8, 7, "front_passenger", "vehicle-1")) == []
    assert engine.add(Observation(0, "phone", 0.8, 7, "driver", "vehicle-1")) == []
    assert engine.add(Observation(0.5, "phone", 0.8, 7, "driver", "vehicle-1")) == []
    events = engine.add(Observation(1.1, "phone", 0.9, 7, "driver", "vehicle-1"))
    assert events[0].event_type == "PHONE"
    assert events[0].review_status == "NEEDS_REVIEW"


def test_handheld_phone_context_can_be_pending_but_mounted_phone_is_ignored():
    engine = TemporalEventEngine(window_seconds=2, min_positive_seconds=1, min_observations=2, cooldown_seconds=5)
    assert engine.add(Observation(0, "phone", 0.8, 8, "driver", "vehicle-1", "MOUNTED_OR_STATIC")) == []
    engine.add(Observation(0, "phone", 0.8, 9, "driver", "vehicle-1", "HANDHELD"))
    events = engine.add(Observation(1.1, "phone", 0.9, 9, "driver", "vehicle-1", "HANDHELD"))
    assert events[0].review_status == "PENDING"


def test_conflicting_seatbelt_state_needs_review():
    engine = TemporalEventEngine(window_seconds=2, min_positive_seconds=1, min_observations=2, cooldown_seconds=5)
    for t in (0, 1.1):
        engine.add(Observation(t, "seatbelt_fastened", 0.8, 2, "front_passenger", "vehicle-1"))
    engine.add(Observation(0, "seatbelt_unfastened", 0.9, 2, "front_passenger", "vehicle-1"))
    events = engine.add(Observation(1.1, "seatbelt_unfastened", 0.9, 2, "front_passenger", "vehicle-1"))
    assert events[0].review_status == "NEEDS_REVIEW"
    assert events[0].occupant_role == "front_passenger"


def test_passenger_unfastened_is_a_violation():
    engine = TemporalEventEngine(window_seconds=2, min_positive_seconds=1, min_observations=2, cooldown_seconds=5)
    engine.add(Observation(0, "seatbelt_unfastened", 0.8, 12, "front_passenger", "vehicle-1"))
    events = engine.add(Observation(1.1, "seatbelt_unfastened", 0.9, 12, "front_passenger", "vehicle-1"))
    assert events[0].event_type == "NO_SEATBELT"


def test_person_outside_vehicle_never_creates_event():
    engine = TemporalEventEngine(window_seconds=2, min_positive_seconds=1, min_observations=2, cooldown_seconds=5)
    engine.add(Observation(0, "seatbelt_unfastened", 0.9, 21, "driver", None))
    assert engine.add(Observation(1.1, "seatbelt_unfastened", 0.9, 21, "driver", None)) == []


def test_behavior_box_is_not_mapped_to_physical_phone(tmp_path):
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"not decoded by this mapping test")
    image.with_suffix(".txt").write_text("0 0.5 0.5 0.2 0.2\n")
    mappings = {"calling": {"canonical": None, "requires_review": True, "reason": "behavior"}}
    annotations, excluded = map_suggestions(image, {0: "calling"}, mappings)
    assert annotations == []
    assert excluded[0]["reason"] == "behavior"


def test_kaggle_preflight_validates_three_class_frozen_bundle(tmp_path):
    root = tmp_path / "bundle"
    dataset = root / "dataset"
    for split in ("train", "val", "test"):
        (dataset / "images" / split).mkdir(parents=True)
        (dataset / "labels" / split).mkdir(parents=True)
        for class_id in range(3):
            sample = f"{split}_{class_id}"
            Image.new("RGB", (32, 32), "gray").save(dataset / "images" / split / f"{sample}.jpg")
            (dataset / "labels" / split / f"{sample}.txt").write_text(
                f"{class_id} 0.5 0.5 0.25 0.25\n"
            )
    (dataset / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": ".",
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "nc": 3,
                "names": {
                    0: "phone",
                    1: "seatbelt_fastened",
                    2: "seatbelt_unfastened",
                },
            }
        )
    )
    (dataset / "split_metadata.json").write_text(json.dumps({"isolation": {"status": "PASS"}}))
    (root / "test_frozen.json").write_text(json.dumps({"status": "FROZEN", "test_samples": 3}))
    expected = {}
    for path in root.rglob("*"):
        if path.is_file():
            expected[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    (root / "expected_hashes.json").write_text(json.dumps(expected))
    report = preflight(root)
    assert report["splits"]["test"]["instances"]["seatbelt_unfastened"] == 1


def test_source_yolo_layout_reads_matching_label_from_sibling_labels_directory(tmp_path):
    image = tmp_path / "train" / "images" / "driver_frame.jpg"
    label = tmp_path / "train" / "labels" / "driver_frame.txt"
    image.parent.mkdir(parents=True)
    label.parent.mkdir(parents=True)
    image.write_bytes(b"image fixture")
    label.write_text("3 0.5 0.5 0.1 0.1\n")

    assert source_label_path(image, tmp_path) == label


def test_roboflow_download_overwrites_only_private_staging_directory(tmp_path, monkeypatch):
    destination = tmp_path / "staging"
    destination.mkdir()
    monkeypatch.setenv("ROBOFLOW_API_KEY", "test-key")
    version = MagicMock()
    project = MagicMock()
    project.version.return_value = version
    workspace = MagicMock()
    workspace.project.return_value = project
    client = MagicMock()
    client.workspace.return_value = workspace
    source = {
        "source_id": "seatbelt",
        "source_url": "https://example.invalid",
        "download": {
            "type": "roboflow",
            "workspace": "workspace",
            "project": "project",
            "version": 4,
            "format": "yolov11",
        },
    }

    with patch("roboflow.Roboflow", return_value=client):
        run_download(source, destination)

    version.download.assert_called_once_with(
        "yolov11", location=str(destination), overwrite=True
    )


def test_review_queue_maps_phone_from_separate_source_label_file(tmp_path):
    label = tmp_path / "labels" / "driver_frame.txt"
    label.parent.mkdir()
    label.write_text("3 0.5 0.5 0.1 0.1\n4 0.5 0.5 0.2 0.3\n")
    mappings = {
        "phone": {"canonical": "phone", "requires_review": False},
        "seatbelt": {"canonical": None, "requires_review": True, "reason": "bare belt"},
    }

    annotations, excluded = map_suggestions_from_label(label, {3: "Phone", 4: "Seatbelt"}, mappings)

    assert annotations == [{"class_id": 0, "yolo": [0.5, 0.5, 0.1, 0.1], "occupant_role": "PENDING", "source_class_id": 3, "source_class_name": "Phone", "requires_review": True}]
    assert excluded[0]["reason"] == "bare belt"


def test_source_group_id_keeps_frames_of_one_video_together():
    assert source_group_id(Path("-1070-_mp4-10_jpg.rf.abc.jpg"), "dms") == "dms:-1070-_mp4"
    assert source_group_id(Path("-1070-_mp4-99_jpg.rf.def.jpg"), "dms") == "dms:-1070-_mp4"


def test_source_group_id_marks_unparseable_file_for_review():
    assert source_group_id(Path("_117842784_lg_jpg.rf.abc.jpg"), "dms").startswith("dms:GROUP_REVIEW_REQUIRED:")


def test_source_label_reader_accepts_noncanonical_source_class_for_review(tmp_path):
    label = tmp_path / "frame.txt"
    label.write_text("4 0.5 0.5 0.2 0.3\n")

    assert read_source_labels(label) == [(4, [0.5, 0.5, 0.2, 0.3])]
