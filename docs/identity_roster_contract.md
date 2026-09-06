# Identity Roster and Canonical Identity Contract

This document defines the governance rules, canonical ID namespaces, and cryptographic binding requirements for ground-truth identity rosters, identity manifests, and identity adjudication in Project 7.

---

## 1. Governance & Provenance Rules

Ground truth integrity requires complete provenance without synthetic or auto-fabricated review records.

### Rule 1: No Auto-Fabricated Human Review
- `create_identity_roster()` produces an **UNREVIEWED** identity roster (`roster_status = "UNREVIEWED_IDENTITY_ROSTER"`) by default:
  - `human_review_status = "PENDING"`
  - `reviewer_type = "AI"`
  - `reviewer_id = None`
  - `reviewed_at = None`
  - `adjudication_status = "PENDING"`
- The system **never** defaults or falls back to synthetic reviewer IDs (such as `"human-reviewer-1"`). Placeholder reviewer names are rejected fail-closed.

### Rule 2: Explicit Promotion & Semantic Evidence Hash
Elevating an identity roster to human approval requires explicit human action via `approve_identity_roster()`:
- `reviewer_id`: Non-empty string identifying the verified human auditor (e.g. `"lead-auditor-alice"`).
- `reviewed_at`: Valid ISO-8601 string containing an explicit timezone (e.g. `"2026-09-06T00:00:00Z"`).
- `adjudication_status`: Must be `"FINAL"`.
- `evidence_hash`: Deterministic SHA-256 computed over the canonical video/vehicle/occupant data via `canonical_identity_evidence_hash()`.
  - The evidence hash is semantically verified at creation, approval, extraction, and freezing.
  - Passing an arbitrary or stale evidence hash is rejected fail-closed.

### Rule 3: Multi-Roster Per-Source Provenance (No "Last Reviewer Wins")
When multiple rosters are combined into an identity manifest via `extract_identity_manifest_from_roster([r1, r2, ...])`:
- Each roster file is recorded in `roster_sources` with its own cryptographic SHA-256 and review record.
- If **any** source roster is unreviewed or pending, the entire manifest evaluation scope is downgraded to `"UNREVIEWED_ROSTER_SCOPE"`, and `eligible_for_frozen_event_evaluation` is set to `False`.
- Only when **all** source rosters have verified human review does the manifest qualify for `"FULL_SYSTEM_EVENT_EVALUATION"`.

### Rule 4: Cryptographic Re-Hash at Freezer Boundary
When `freeze_identity_manifest()` locks an independent identity manifest:
- The freezer re-hashes every individual source file declared in `roster_sources` directly from disk (`sha256_file(p) == item["sha256"]`).
- In addition to the file hash, the freezer recomputes `canonical_identity_evidence_hash()` from the disk content and verifies that the recorded `evidence_hash` matches.
- Any disk tampering or file modification between extraction and freezing triggers a cryptographic mismatch error.
- All individual source review records are verified again and preserved in the frozen lock file.

---

## 2. Canonical Identity Namespaces

To prevent ambiguity, Project 7 mandates strict hierarchical entity ID formatting validated by `validate_identity_contract()`.

### Hierarchy
Every occupant belongs to a specific cabin, which belongs to a specific vehicle, which belongs to a specific video:
```
video_id
 └── vehicle_id
      └── cabin_id
           └── occupant_id
```

### Namespace Formats

| Scene Type | Vehicle ID (`vehicle_id`) | Cabin ID (`cabin_id`) | Occupant ID (`occupant_id`) |
| :--- | :--- | :--- | :--- |
| **Fixed Camera / In-Cabin** | `video:<video_id>:provided-vehicle` | `video:<video_id>:provided-cabin` | `video:<video_id>:provided-cabin:occupant-track:<n>` |
| **Dynamic Multi-Vehicle / Traffic** | `video:<video_id>:vehicle-track:<track_id>` | `video:<video_id>:vehicle-track:<track_id>:cabin:<epoch>` | `video:<video_id>:vehicle-track:<track_id>:cabin:<epoch>:occupant-track:<track_id>` |

- **Fixed Camera Scenes**: When a camera observes a single fixed vehicle cabin (such as driver-monitoring dashcams), the canonical identifier `provided-vehicle` and `provided-cabin` MUST be used by both the identity roster and the runtime pipeline.
- **Dynamic Traffic Scenes**: When multiple vehicles enter and leave the frame, the vehicle track ID and cabin epoch MUST be tracked and matched.

---

## 3. Cabin Locality and Identity Adjudication

### The Adjudication Problem
Ground truth annotations identify canonical occupants (e.g. `driver` as `occupant-track:1`, `passenger` as `occupant-track:2`).
Runtime detectors observe physical humans and assign arbitrary tracking IDs (e.g. `occupant-track:88`, `occupant-track:99`).

Identity adjudication maps runtime tracks to ground truth tracks:
$$\text{runtime\_occupant} \longrightarrow \text{gt\_occupant}$$

### Locality Enforcement
`freeze_identity_adjudication()` enforces strict cabin locality:
```python
r_cabin = runtime_occ.rsplit(":occupant-track:", 1)[0]
g_cabin = gt_occ.rsplit(":occupant-track:", 1)[0]
if r_cabin != g_cabin:
    assert cabin_mappings.get(r_cabin) == g_cabin, "cross-cabin adjudication violation"
```

1. **Intra-Cabin by Default**: By default, runtime tracks can only be mapped to GT occupants within the **exact same video, vehicle, and cabin**.
2. **Governed Cross-Cabin Mapping**: If dynamic tracking IDs differ between runtime and GT (e.g. `vehicle-track:7` vs `vehicle-track:1`), cross-cabin occupant mapping is permitted **only if explicitly declared in a verified `cabin_mappings` dictionary** within the same video.
3. **Bijective Mapping**: Both `mappings` and `cabin_mappings` must be strictly 1-to-1 (injective). No duplicate targets are permitted.
4. **Namespace Alignment**: For in-cabin test sets, detectors must be configured with `provided-cabin` context so runtime tracks share the same prefix `video:<id>:provided-cabin` as the ground-truth roster.

Freezing requires `human_review_status = APPROVED`, `reviewer_type = HUMAN`, a non-empty
`reviewer_id`, a timezone-aware `reviewed_at`, and explicit `adjudication_status = FINAL`.
The frozen lock preserves `FINAL`; evaluation and integrity verification reject missing or
non-final adjudication status. Older locks without this field require a finalized source and
a new freeze.

---

## 4. Scientific Policy on Vehicle & Cabin Association

When evaluating end-to-end models on sequence data, association errors must be treated with strict scientific rigor:

### Policy A: Default Full-System Benchmark Invariant (Unassisted)
- **Principle**: In an unassisted full-system evaluation, tracking the correct vehicle and cabin entities over time is an essential capability of the detection system.
- **Scoring**: If a system detects a driver in an unmapped or incorrect vehicle/cabin track, it constitutes an association error.
- **Effect**: This error produces a **False Positive (FP)** on the hallucinated/incorrect vehicle cabin and a **False Negative (FN)** on the true vehicle cabin.
- **Application**: Mandatory for official system benchmarks and freeze certifications.
- **Evaluator gate**: `evaluate_frozen()` rejects non-empty `cabin_mappings` by default.
  Intra-cabin occupant ID alignment remains supported under the locality rules above;
  it cannot repair vehicle/cabin association errors.

### Policy B: Governed Hierarchical Adjudication (Diagnostic / Multi-Vehicle)
- **Principle**: In complex multi-vehicle scenes where ground-truth vehicle tracks were indexed independently from runtime detectors, human auditors can isolate behavior/action recognition from tracking indexing.
- **Mechanism**: Auditors sign off on a `FROZEN_IDENTITY_ADJUDICATION` artifact containing explicit `cabin_mappings`:
  ```json
  {
    "cabin_mappings": {
      "video:v1:vehicle-track:7:cabin:0": "video:v1:vehicle-track:1:cabin:0"
    },
    "mappings": {
      "video:v1:vehicle-track:7:cabin:0:occupant-track:42": "video:v1:vehicle-track:1:cabin:0:occupant-track:1"
    }
  }
  ```
- **Evaluation Alignment**: When applied during evaluation, the prediction row's `occupant_id`, `cabin_id`, and `vehicle_id` are consistently aligned to the target ground truth cabin, allowing event detection accuracy to be scored accurately without artifact mismatches.
- **Explicit diagnostic opt-in**: Pass `allow_hierarchical_adjudication_diagnostic=True`
  to `evaluate_frozen()`, or `--allow-hierarchical-adjudication-diagnostic` on the CLI.
  Whenever the lock contains non-empty `cabin_mappings`, the report status is
  `MEASURED_HIERARCHICAL_ADJUDICATION_DIAGNOSTIC` and its `scientific_claim` is
  `DIAGNOSTIC_BEHAVIOR_METRICS_AFTER_HUMAN_IDENTITY_ALIGNMENT`, even when the manifest
  scope is `FULL_SYSTEM_EVENT_EVALUATION`. These metrics cannot certify an official benchmark.
  The manifest scope continues to describe the ground-truth population, not the report claim.
- **Conditional manifests**: `--allow-conditional-evaluation` remains separately required
  for conditional ground truth; it does not authorize hierarchical adjudication. When both
  apply, both flags are required and the report retains the hierarchical diagnostic status.
- **Verification**: `--verify-existing` checks artifact hashes and rejects hierarchical
  reports relabeled with an official status or claim.
