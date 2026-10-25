from __future__ import annotations

from training.dmd.audit_v31_live_decisions import validate_decisions


def _pair() -> list[dict]:
    common = {
        "event_type": "PHONE",
        "event_start_seconds": 1.0,
        "vehicle_context_id": "CABIN",
        "track_id": "driver",
        "occupant_group": "driver",
    }
    return [
        {
            **common,
            "sequence": 1,
            "decision_type": "START",
            "reason": "ACTIVATION",
            "decision_time_seconds": 1.5,
            "evidence_available_through_seconds": 1.5,
            "event_end_seconds": None,
        },
        {
            **common,
            "sequence": 2,
            "decision_type": "END",
            "reason": "EVIDENCE_RELEASE_OR_EXPIRY",
            "decision_time_seconds": 3.0,
            "evidence_available_through_seconds": 3.0,
            "event_end_seconds": 3.0,
        },
    ]


def test_validate_live_decision_pair() -> None:
    result = validate_decisions(_pair())
    assert result["status"] == "PASS"
    assert result["activation_latency_seconds"]["mean"] == 0.5
    assert result["decision_duration_seconds"]["mean"] == 2.0


def test_validate_live_decisions_rejects_end_without_start() -> None:
    result = validate_decisions(_pair()[1:])
    assert result["status"] == "FAIL"
    assert any("END without START" in item for item in result["errors"])


def test_validate_live_decisions_rejects_missing_end() -> None:
    result = validate_decisions(_pair()[:1])
    assert result["status"] == "FAIL"
    assert any("lack END" in item for item in result["errors"])
