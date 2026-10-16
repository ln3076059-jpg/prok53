# Project and resource audit

Source commit: `43b87ee3151060a9dd2b1b2d02ca26c0a45c138e`. Local and remote branch matched before changes.
Branch: `roadwatch-v4-candidate`. The user's modified prompt was retained.

The read-only audit found 357 checkpoint files and 346334 files under `dataset/`, excluding protected paths. `datasets/` is absent.
RAM: 6005.7 MiB. Audit free disk: 9.90 GiB. Runtime hardware CUDA available: False. No external training mount was configured.

Implemented: explicit data-root resolution, T0–T5 visual sampling experiments, evidence-bounded confirmation, ROI-to-frame box mapping, missing-view abstention, frozen detector/seatbelt inference and artifact verification.
Production integration remains gated because no new candidate is eligible. Existing model loading and backend workflow were exercised by system smoke and full pytest.

[Audit JSON](audit.json), [test evidence](15_TEST_RESULTS.md), [registry](../../models/model_registry_ai85_plus.json).

Status: `INDEPENDENT_AI85_NOT_ESTABLISHED`. All new AI measurements use already-consumed development data. No production model was approved. Machine-readable evidence is in this directory.
