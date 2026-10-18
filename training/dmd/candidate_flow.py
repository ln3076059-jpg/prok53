"""Stage-level research ledger for phone candidate rejection and coverage analysis."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from enum import IntEnum


class CandidateStage(IntEnum):
    DETECTED = 0
    DRIVER_ASSOCIATED = 1
    HANDHELD_EVIDENCE = 2
    ACTION_SUPPORTED = 3
    FUSION_ADMITTED = 4
    TEMPORAL_ACTIVATED = 5
    EVENT_MATCHED = 6


@dataclass(frozen=True)
class CandidateDecision:
    candidate_id: str
    stage: str
    admitted: bool
    reason_code: str
    truth: str = "UNKNOWN"

    def validate(self) -> None:
        if not self.candidate_id or self.stage not in CandidateStage.__members__:
            raise ValueError("candidate ID and a known stage are required")
        if not self.reason_code:
            raise ValueError("every candidate decision requires a reason code")
        if self.truth not in {"POSITIVE", "NEGATIVE", "UNKNOWN"}:
            raise ValueError("truth must be POSITIVE, NEGATIVE, or UNKNOWN")


class CandidateFlowLedger:
    def __init__(self) -> None:
        self._rows: list[CandidateDecision] = []
        self._last_stage: dict[str, int] = {}

    def record(self, decision: CandidateDecision) -> None:
        decision.validate()
        stage_index = int(CandidateStage[decision.stage])
        previous = self._last_stage.get(decision.candidate_id, -1)
        if stage_index <= previous:
            raise ValueError("candidate stages must be recorded once in forward order")
        self._last_stage[decision.candidate_id] = stage_index
        self._rows.append(decision)

    def rows(self) -> list[dict]:
        return [asdict(row) for row in self._rows]

    def summary(self) -> dict:
        stage_counts: defaultdict[str, Counter] = defaultdict(Counter)
        reasons: Counter[str] = Counter()
        for row in self._rows:
            stage_counts[row.stage]["admitted" if row.admitted else "rejected"] += 1
            if not row.admitted:
                reasons[row.reason_code] += 1
        positives_rejected = sum(
            1 for row in self._rows if row.truth == "POSITIVE" and not row.admitted
        )
        negatives_admitted = sum(
            1 for row in self._rows if row.truth == "NEGATIVE" and row.admitted
        )
        unique = {row.candidate_id for row in self._rows}
        unknown = {row.candidate_id for row in self._rows if row.truth == "UNKNOWN"}
        return {
            "candidates": len(unique),
            "stages": {stage: dict(counts) for stage, counts in sorted(stage_counts.items())},
            "rejection_reasons": dict(sorted(reasons.items())),
            "ground_truth_positive_rejections": positives_rejected,
            "negative_candidates_admitted": negatives_admitted,
            "unknown_candidates": len(unknown),
            "unknown_coverage": len(unknown) / len(unique) if unique else 0.0,
        }
