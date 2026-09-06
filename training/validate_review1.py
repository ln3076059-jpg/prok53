"""Validate temporal-only Review1 proposals/final receipts; never promote or freeze."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

from jsonschema import Draft7Validator

SCHEMA = Path(__file__).resolve().parents[1] / "datasets/schemas/v2_review1.schema.json"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def payload_digest(payload):
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def validate_record(record):
    """Return errors. Evidence validity is not proof of human authorship or rights."""
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    errors = [f"schema: {e.message}" for e in Draft7Validator(schema).iter_errors(record)]
    if errors:
        return errors
    video = Path(record["video_path"])
    if not video.is_file() or digest(video) != record["video_sha256"]:
        errors.append("video missing or SHA256 mismatch")
    if record["status"] == "REVIEW1_PROPOSAL":
        if any(
            record.get(k)
            for k in ("reviewer_id", "reviewed_at", "review_evidence", "human_decision")
        ):
            errors.append("Review1 cannot carry human reviewer/time/evidence")
    else:
        identity = record.get("reviewer_id", "").strip()
        tokens = set(re.findall(r"[a-z]+", identity.lower()))
        if (
            len(identity) < 3
            or not tokens
            or tokens
            & {
                "ai",
                "assistant",
                "bot",
                "codex",
                "placeholder",
                "test",
                "dummy",
                "example",
                "unknown",
                "pending",
                "human",
                "reviewer",
                "tbd",
            }
            or "<" in identity
            or ">" in identity
        ):
            errors.append("INVALID_HUMAN_PROVENANCE: missing/placeholder reviewer")
        try:
            timestamp = datetime.fromisoformat(record.get("reviewed_at", "").replace("Z", "+00:00"))
            if timestamp.tzinfo is None or timestamp.utcoffset() is None:
                raise ValueError("timezone missing")
        except (ValueError, TypeError):
            errors.append("INVALID_HUMAN_PROVENANCE: timezone-aware timestamp required")
        evidence = record.get("review_evidence", {})
        path = Path(evidence.get("path", ""))
        if not path.is_file() or not path.stat().st_size:
            errors.append("human evidence missing/empty")
        elif digest(path) != evidence.get("sha256"):
            errors.append("human evidence SHA256 mismatch")
        else:
            # A final receipt must identify the exact reviewed payload and video.
            # It is human-supplied, never synthesized by this validator.
            try:
                receipt = json.loads(path.read_text(encoding="utf-8-sig"))
                expected = {
                    "video_sha256": record["video_sha256"],
                    "payload_sha256": payload_digest(record["payload"]),
                    "reviewer_id": record["reviewer_id"],
                    "reviewed_at": record["reviewed_at"],
                    "decision": record["human_decision"],
                    "record_type": record["record_type"],
                }
                if not isinstance(receipt, dict) or any(
                    receipt.get(k) != v for k, v in expected.items()
                ):
                    errors.append(
                        "human evidence does not bind reviewed payload/video/reviewer/decision"
                    )
                elif (
                    not isinstance(receipt.get("review_notes"), str)
                    or not receipt["review_notes"].strip()
                ):
                    errors.append("human evidence requires substantive review_notes")
            except (ValueError, OSError):
                errors.append("human evidence must be a readable JSON review receipt")

    if record["record_type"] == "sequence":
        payload = record["payload"]
        count = payload.get("frame_count")
        fps = payload.get("fps")
        if (
            type(count) is not int
            or count <= 0
            or type(fps) not in (int, float)
            or not 0 < fps < 10000
        ):
            return errors + ["sequence requires positive frame_count/fps"]
        occupants = payload.get("occupants", [])
        if not isinstance(occupants, list) or any(not isinstance(o, dict) for o in occupants):
            return errors + ["occupants must be objects"]
        ids = [o.get("occupant_id_proposal") for o in occupants]
        if (
            not ids
            or any(not isinstance(i, str) or not i for i in ids)
            or len(set(ids)) != len(ids)
        ):
            return errors + ["sequence requires unique occupant proposals"]
        phone = {
            "PHONE_USE",
            "PHONE_PRESENT_NOT_USED",
            "MOUNTED_OR_STATIC_PHONE",
            "NO_PHONE",
            "UNKNOWN",
        }
        belt = {"FASTENED", "UNFASTENED", "UNCERTAIN_OR_OCCLUDED", "NOT_APPLICABLE"}
        for key, states in [
            ("phone_intervals", phone),
            ("seatbelt_intervals", belt),
            ("context_intervals", None),
        ]:
            intervals = payload.get(key, [])
            if not isinstance(intervals, list) or any(not isinstance(i, dict) for i in intervals):
                errors.append(f"{key} must be an array of objects")
                continue
            for interval in intervals:
                a, b = interval.get("start_frame"), interval.get("end_frame")
                if (
                    type(a) is not int
                    or type(b) is not int
                    or not 0 <= a < b <= count
                    or interval.get("occupant_id_proposal") not in ids
                    or (states is not None and interval.get("state") not in states)
                ):
                    errors.append(f"invalid {key} interval")
                    continue
                for frame, time_key in [(a, "start_time"), (b, "end_time")]:
                    value = interval.get(time_key)
                    if type(value) not in (int, float) or not abs(value - frame / fps) < 1e-6:
                        errors.append(f"{key} frame/time mismatch")
                confidence = interval.get("confidence")
                if type(confidence) not in (int, float) or not 0 <= confidence <= 1:
                    errors.append(f"{key} invalid confidence")
                for field in ("visibility", "review_notes", "occupant_role_proposal"):
                    if not isinstance(interval.get(field), str) or not interval[field].strip():
                        errors.append(f"{key} missing {field}")
                if states is None:
                    for field in ("inside_vehicle", "outside_vehicle_person", "motorcycle_flag"):
                        if field not in interval or (
                            interval[field] is not None and type(interval[field]) is not bool
                        ):
                            errors.append(f"context invalid {field}")
                    if (
                        interval.get("inside_vehicle") is True
                        and interval.get("outside_vehicle_person") is True
                    ):
                        errors.append("context cannot be both inside and outside")
                    if (
                        not isinstance(interval.get("conditions"), str)
                        or not interval["conditions"].strip()
                    ):
                        errors.append("context missing conditions")
            for oid in ids:
                cursor = 0
                try:
                    spans = sorted(
                        (i["start_frame"], i["end_frame"])
                        for i in intervals
                        if i.get("occupant_id_proposal") == oid
                    )
                    for a, b in spans:
                        if a != cursor:
                            errors.append(f"{key} coverage gap/overlap for {oid}")
                        cursor = b
                    if cursor != count:
                        errors.append(f"{key} incomplete coverage for {oid}")
                except (KeyError, TypeError):
                    errors.append(f"{key} invalid coverage")
        # Human confirmation is followed by the existing canonical validator.
        # This command never creates canonical truth or grants intake eligibility.
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path)
    args = parser.parse_args()
    try:
        errors = validate_record(json.loads(args.record.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        errors = [str(exc)]
    print(
        json.dumps(
            {
                "status": "INVALID" if errors else "RECORD_CONSISTENCY_PASS",
                "errors": errors,
                "governance_promotion": False,
            },
            indent=2,
        )
    )
    raise SystemExit(bool(errors))


if __name__ == "__main__":
    main()
