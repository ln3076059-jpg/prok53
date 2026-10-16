# Dataset roots, labels and migration

The local root is `dataset/`; `datasets/` does not exist. `ROADWATCH_DATA_ROOT` can explicitly select another root; ambiguous conventional roots fail with an actionable error. Explicit caller paths are preserved.

Historical deleted-path audit: {'deleted_paths_checked': 3558, 'counterparts_present': 3558, 'exact_matches': 3491, 'protected_not_read': 0}. The detailed old-Git-blob and local SHA256 comparison is in [evidence_diagnostics.json](evidence_diagnostics.json). No dataset was restored, moved, deleted or relabeled. A byte mismatch must be investigated individually; existence alone is not checksum equivalence.
The follow-up [migration difference audit](dataset_migration_differences.json) confirms all 67 byte mismatches differ only in line endings or outer whitespace. All 65 JSON/JSONL mismatches are semantically equal; the other two are CSV files with newline-only differences.

Policies, schemas, manifests, stale path references and hashes are listed in [audit.json](audit.json). This stage fixes the schema/policy consumers and readiness audit exercised by tests; historical manifests and obsolete training scripts can still contain old `datasets/` references.
DMD remains `RIGHTS_ACCEPTED=false`, `CANONICAL_ELIGIBLE=false`, `TEMPORAL_DEVELOPMENT_ELIGIBLE=true` in the intake record. There are three permitted consumed development subjects, seven BODY/FACE/HANDS videos, and 29 phone events in the evaluated protocol. gC-14 has BODY only. Protected directories were pruned before content walking.

The older seatbelt classifier dataset has 0 train UNCERTAIN crops and `ready_for_training=false`. The actual pending-approval pretrain bundle has 100 train, 25 val and 25 test UNCERTAIN crops, and its labels are explicitly model-assisted pending approval. It has no approved independent subject split.
Phone bootstrap reports 70,282 near-duplicate pairs and grouped splits; these historical clustering facts do not establish unseen subject/camera generalization. No new near-duplicate certification or new subject consent was fabricated.
