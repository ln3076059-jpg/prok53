# ROADWATCH V19 — DMD gA-1 Holdout Integrity & Exposure Audit
## 1. Provenance Audit
- **Archive Path:** `dataset/dmd-dataset-distraction-gA-1/dmd/gA/1`
- **Archive Size:** 15 files, 5.07 GB.
- **Sessions:** `s1`, `s2` (458.1s, 8 GT phone events), `s3`.
- **Streams:** BODY, FACE, HANDS.

## 2. Exposure Findings
In V18, frame at 60s of `gA-1 s2` was loaded and processed to test ear-crop phone detection.
This constitutes **frame-level diagnostic exposure**.
Per V19 protocol:
- Status: **DEVELOPMENT_EXPOSED_OR_PARTIALLY_CONTAMINATED**.
- `gA-1` remains protected from training, fine-tuning, and threshold selection.
- It cannot serve as a blind holdout for production approval.
- An independent AI85 benchmark requires a clean multi-subject panel (>=5 subjects).
