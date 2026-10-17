# V12 code and data audit

Audit HEAD: `4b19c4834b9520df2f72caca6c6a3fc9606cbdde`. The user-authored `PRODUCT.md` remained a working-tree input and was not treated as a model artifact. The V12 code adds a live fail-closed permission gate, bounded causal streaming E3, portable JSON-only linear-head inference, conservative association ambiguity handling, metric recomputation, original-frame ROI diagnostics, and event-ID replay. Candidate integration is opt-in and production defaults were not changed.

DMD permission remains `RIGHTS_ACCEPTED=false`, `CANONICAL_ELIGIBLE=false`, `TEMPORAL_DEVELOPMENT_ELIGIBLE=true`. Linear-head, calibration, fusion, boundary, backbone fitting and upload are all denied by the current live record. `gB-9`, `gC-14`, and `gZ-36` remain repeatedly used development subjects. Protected `gE-28` and `gZ-37` media/labels were not opened.

The seatbelt manifest has 500 candidates: `{'PENDING': 500}`, with 0 automatic training labels. The checkpoint inventory contains 64 files totaling 406.6 MiB; files were hashed without pickle deserialization or activation. No training, external upload, production activation, ground-truth edit, or matcher edit occurred.
