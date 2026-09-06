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
            payload={"recommendation": "NEEDS_HUMAN_DECISION"},
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
            occupants=[{"occupant_id_proposal": "unit-person"}],
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
        record["payload"]["recommendation"] = "ACCEPT_CANDIDATE"
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


if __name__ == "__main__":
    unittest.main()
