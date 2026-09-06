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

### Rule 2: Explicit Promotion via `approve_identity_roster`
Elevating an identity roster to human approval requires explicit human action via `approve_identity_roster()`:
- `reviewer_id`: Non-empty string identifying the verified human auditor (e.g. `"lead-auditor-alice"`).
- `reviewed_at`: Valid ISO-8601 string containing an explicit timezone (e.g. `"2026-09-06T00:00:00Z"`).
- `adjudication_status`: Must be `"FINAL"`.
- `evidence_hash`: 64-character SHA-256 hash computed over the canonical video/vehicle/occupant data.

### Rule 3: Multi-Roster Per-Source Provenance (No "Last Reviewer Wins")
When multiple rosters are combined into an identity manifest via `extract_identity_manifest_from_roster([r1, r2, ...])`:
- Each roster file is recorded in `roster_sources` with its own cryptographic SHA-256 and review record.
- If **any** source roster is unreviewed or pending, the entire manifest evaluation scope is downgraded to `"UNREVIEWED_ROSTER_SCOPE"`, and `eligible_for_frozen_event_evaluation` is set to `False`.
- Only when **all** source rosters have verified human review does the manifest qualify for `"FULL_SYSTEM_EVENT_EVALUATION"`.

### Rule 4: Cryptographic Re-Hash at Freezer Boundary
When `freeze_identity_manifest()` locks an independent identity manifest:
- The freezer re-hashes every individual source file declared in `roster_sources` directly from disk (`sha256_file(p) == item["sha256"]`).
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

### Strict Locality Requirement
`freeze_identity_adjudication()` strictly enforces **cabin locality**:
```python
r_cabin = runtime_occ.rsplit(":occupant-track:", 1)[0]
g_cabin = gt_occ.rsplit(":occupant-track:", 1)[0]
assert r_cabin == g_cabin, "cross-cabin adjudication violation"
```

1. **Intra-Cabin Only**: Runtime tracks can only be mapped to GT occupants within the **exact same video, vehicle, and cabin**.
2. **Cross-Cabin Prohibition**: Mapping a track from `cabin:0` to `cabin:1`, or from `vehicle-track:1` to `vehicle-track:2`, or from dynamic `vehicle-track:7` to `provided-cabin` is strictly rejected as a **cross-cabin adjudication violation**.
3. **Bijective Mapping**: Adjudication must be strictly 1-to-1 (injective). No two runtime tracks can be mapped to the same ground-truth occupant.
4. **Namespace Alignment**: For in-cabin test sets, detectors must be configured with `provided-cabin` context so runtime tracks share the same prefix `video:<id>:provided-cabin` as the ground-truth roster.
