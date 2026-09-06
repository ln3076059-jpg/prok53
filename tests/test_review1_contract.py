"""Review1-only contract tests. Ephemeral bytes are not dataset or human evidence."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from training.validate_review1 import digest, payload_digest, validate_record


class Review1ContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.video = Path(self.tmp.name) / "unit-input.bin"
        self.video.write_bytes(b"isolated unit fixture, never real media")
        self.record = dict(
            record_type="rights",
            dataset_role="TEMPORAL_DEVELOPMENT",
            video_id="unit-fixture",
            video_path=str(self.video),
            video_sha256=digest(self.video),
            reviewer_type="AI",
            reviewer_origin="AI",
            review_stage="REVIEW1",
            reviewer_agent="CODEX",
            status="REVIEW1_PROPOSAL",
            requires_human_confirmation=True,
            adjudication_status="PENDING",
            human_approved=False,
            payload=dict(
                source_page_url="https://example.invalid/asset",
                license_or_terms_url="https://example.invalid/terms",
                creator=None,
                asset_id=None,
                rights_recommendation="NEEDS_HUMAN_DECISION",
                rights_reason="Isolated unit fixture; no real rights claim.",
                evidence_available=False,
                evidence=[],
            ),
        )

    def final_fixture(self):
        record = copy.deepcopy(self.record)
        record.update(
            reviewer_type="HUMAN",
            reviewer_origin="HUMAN",
            reviewer_agent=None,
            review_stage="FINAL_REVIEW",
            status="HUMAN_APPROVED",
            requires_human_confirmation=False,
            adjudication_status="FINAL",
            human_approved=True,
            reviewer_id="Nguyen-Van-An",
            reviewed_at="2026-09-06T12:00:00+07:00",
            human_decision="APPROVE",
        )
        receipt = dict(
            video_sha256=record["video_sha256"],
            payload_sha256=payload_digest(record["payload"]),
            reviewer_id=record["reviewer_id"],
            reviewed_at=record["reviewed_at"],
            decision="APPROVE",
            record_type="rights",
            review_notes="Unit fixture only.",
        )
        path = Path(self.tmp.name) / "unit-receipt.json"
        path.write_text(json.dumps(receipt), encoding="utf-8")
        record["review_evidence"] = dict(path=str(path), sha256=digest(path))
        return record

    def sequence_fixture(self):
        record = copy.deepcopy(self.record)
        record["record_type"] = "sequence"
        interval = dict(
            start_frame=0,
            end_frame=2,
            start_time=0,
            end_time=1,
            occupant_id_proposal="unit-person",
            occupant_role_proposal="unknown",
            confidence=0.5,
            visibility="partial",
            review_notes="Unit fixture.",
        )
        record["payload"] = dict(
            frame_count=2,
            fps=2,
            occupants=[
                dict(
                    occupant_id_proposal="unit-person",
                    role_proposal="unknown",
                    inside_vehicle_proposal=None,
                    evidence_basis="Isolated fixture.",
                    confidence=0.5,
                )
            ],
            phone_intervals=[dict(interval, state="UNKNOWN")],
            seatbelt_intervals=[dict(interval, state="UNCERTAIN_OR_OCCLUDED")],
            context_intervals=[
                dict(
                    interval,
                    inside_vehicle=None,
                    outside_vehicle_person=None,
                    motorcycle_flag=None,
                    conditions="unknown",
                )
            ],
        )
        return record

    def test_proposal_valid_without_human(self):
        self.assertEqual(validate_record(self.record), [])

    def test_mixed_provenance_rejected(self):
        for key, value in [
            ("human_approved", True),
            ("reviewer_type", "HUMAN"),
            ("adjudication_status", "FINAL"),
            ("reviewer_origin", "HUMAN"),
            ("requires_human_confirmation", False),
            ("status", "HUMAN_APPROVED"),
            ("dataset_role", "NEW_UNTOUCHED_HOLDOUT"),
            ("human_verified", True),
            ("reviewer_id", "Nguyen-Van-An"),
            ("human_decision", "APPROVE"),
        ]:
            with self.subTest(key=key):
                self.assertTrue(validate_record(dict(self.record, **{key: value})))

    def test_final_receipt_consistency_only(self):
        self.assertEqual(validate_record(self.final_fixture()), [])

    def test_final_placeholder_or_naive_timestamp(self):
        for key, value in [
            ("reviewer_id", "HUMAN"),
            ("reviewer_id", "CODEX"),
            ("reviewed_at", "2026-09-06T12:00:00"),
        ]:
            record = self.final_fixture()
            record[key] = value
            self.assertTrue(validate_record(record))

    def test_evidence_mutation_and_payload_mutation(self):
        record = self.final_fixture()
        record["payload"]["rights_reason"] = "Changed after receipt."
        self.assertTrue(validate_record(record))
        record = self.final_fixture()
        Path(record["review_evidence"]["path"]).write_text("changed", encoding="utf-8")
        self.assertTrue(validate_record(record))

    def test_empty_and_missing_evidence(self):
        record = self.final_fixture()
        path = Path(record["review_evidence"]["path"])
        path.write_bytes(b"")
        record["review_evidence"]["sha256"] = digest(path)
        self.assertTrue(validate_record(record))
        path.unlink()
        self.assertTrue(validate_record(record))

    def test_changed_video(self):
        self.video.write_bytes(b"changed")
        self.assertTrue(validate_record(self.record))

    def test_half_open_full_coverage(self):
        self.assertEqual(validate_record(self.sequence_fixture()), [])

    def test_interval_gaps_overlaps_wrong_time_wrong_class(self):
        for key, value in [
            ("start_frame", 1),
            ("end_frame", 3),
            ("start_time", 0.1),
            ("state", "FASTENED"),
            ("confidence", "high"),
        ]:
            record = self.sequence_fixture()
            record["payload"]["phone_intervals"][0][key] = value
            self.assertTrue(validate_record(record))
        record = self.sequence_fixture()
        record["payload"]["phone_intervals"] *= 2
        self.assertTrue(validate_record(record))

    def test_bad_payload_types_fail_closed(self):
        for key, value in [
            ("occupants", [None]),
            ("phone_intervals", [None]),
            ("phone_intervals", None),
            ("context_intervals", []),
        ]:
            record = self.sequence_fixture()
            record["payload"][key] = value
            self.assertTrue(validate_record(record))

    def test_context_inside_outside_conflict(self):
        record = self.sequence_fixture()
        record["payload"]["context_intervals"][0].update(
            inside_vehicle=True, outside_vehicle_person=True
        )
        self.assertTrue(validate_record(record))

    def lineage_fixture(self):
        return dict(
            self.record,
            record_type="physical_lineage",
            payload=dict(
                source_id_proposal=None,
                camera_id_proposal=None,
                capture_session_id_proposal=None,
                physical_vehicle_group_id_proposal=None,
                person_group_ids_proposal=None,
                lineage_status="NOT_PROVABLE",
                evidence_basis="No evidence in isolated fixture.",
            ),
        )

    def identity_fixture(self):
        return dict(
            self.record,
            record_type="identity",
            payload=dict(
                vehicle_id_proposal="unit-vehicle",
                cabin_id_proposal="unit-cabin",
                occupants=self.sequence_fixture()["payload"]["occupants"],
            ),
        )

    def test_empty_payloads_rejected_for_all_types(self):
        for kind in ("rights", "physical_lineage", "identity", "sequence"):
            self.assertTrue(validate_record(dict(self.record, record_type=kind, payload={})))

    def test_rights_recommendation_required_and_typed(self):
        for value in (None, "HUMAN_APPROVED", "APPROVE"):
            record = copy.deepcopy(self.record)
            record["payload"]["rights_recommendation"] = value
            self.assertTrue(validate_record(record))
        record = copy.deepcopy(self.record)
        del record["payload"]["rights_recommendation"]
        self.assertTrue(validate_record(record))

    def test_ai_project_use_approval_rejected(self):
        record = copy.deepcopy(self.record)
        record["payload"]["project_use_review"] = "HUMAN_APPROVED"
        self.assertTrue(validate_record(record))
        record = self.sequence_fixture()
        record["payload"]["canonical_annotation_proposal"] = {"human_approved": True}
        self.assertTrue(validate_record(record))

    def test_unproven_lineage_must_not_invent_ids(self):
        record = self.lineage_fixture()
        self.assertEqual(validate_record(record), [])
        for key, value in [
            ("camera_id_proposal", "C01"),
            ("person_group_ids_proposal", ["invented-person"]),
            ("person_group_ids_proposal", "person"),
            ("lineage_status", "HUMAN_APPROVED"),
        ]:
            changed = copy.deepcopy(record)
            changed["payload"][key] = value
            self.assertTrue(validate_record(changed))

    def test_claimed_lineage_needs_evidence(self):
        record = self.lineage_fixture()
        record["payload"].update(
            source_id_proposal="unit-source",
            camera_id_proposal="unit-camera",
            capture_session_id_proposal="unit-session",
            physical_vehicle_group_id_proposal="unit-car",
            person_group_ids_proposal=["unit-person"],
            lineage_status="PROPOSAL",
        )
        self.assertTrue(validate_record(record))
        record["payload"]["evidence"] = [{"path": str(self.video), "sha256": digest(self.video)}]
        self.assertEqual(validate_record(record), [])  # Structure/hash only, never semantic proof.
        record["payload"]["vehicle_physical_groups_proposal"] = {"other": "different-group"}
        self.assertTrue(validate_record(record))

    def test_identity_role_duplicates_and_cabin(self):
        record = self.identity_fixture()
        self.assertEqual(validate_record(record), [])
        record["payload"]["occupants"] *= 2
        self.assertTrue(validate_record(record))
        record = self.identity_fixture()
        record["payload"]["occupants"][0]["role_proposal"] = "pilot"
        self.assertTrue(validate_record(record))
        record = self.identity_fixture()
        record["payload"]["occupants"][0]["cabin_id_proposal"] = "another-cabin"
        self.assertTrue(validate_record(record))

    def test_interval_role_and_visibility_enum(self):
        for key, value in [
            ("occupant_role_proposal", "pilot"),
            ("occupant_role_proposal", "driver"),
            ("visibility", "perfect"),
        ]:
            record = self.sequence_fixture()
            record["payload"]["phone_intervals"][0][key] = value
            self.assertTrue(validate_record(record))

    def test_rights_evidence_flag_and_hash(self):
        record = copy.deepcopy(self.record)
        record["payload"]["evidence_available"] = True
        self.assertTrue(validate_record(record))
        record["payload"]["evidence"] = [{"path": str(self.video), "sha256": "0" * 64}]
        self.assertTrue(validate_record(record))

    def test_nonfinite_and_malformed_intervals_rejected(self):
        for value in (float("nan"), float("inf"), "bad"):
            record = self.sequence_fixture()
            record["payload"]["phone_intervals"][0]["confidence"] = value
            self.assertTrue(validate_record(record))
        record = self.sequence_fixture()
        record["payload"]["phone_intervals"][0]["state"] = ["UNKNOWN"]
        self.assertTrue(validate_record(record))


if __name__ == "__main__":
    unittest.main()
