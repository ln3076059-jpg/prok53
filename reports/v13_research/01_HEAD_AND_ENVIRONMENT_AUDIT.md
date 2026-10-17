# V13 HEAD and environment audit

Audit started from `943c723ba031e89a4ca3d5de91e17d9d7a4e746f` on `roadwatch-v4-candidate`. The only pre-existing working-tree change was the user-authored `prompt.md`; V13 did not overwrite it. Python is 3.11.9 on `Windows-10-10.0.26200-SP0`. Free workspace disk at audit time was 5.28 GiB. NVIDIA tooling was unavailable, and system RAM was constrained, so no large model download or GPU experiment was attempted.

V13 modifies the ROI extraction path only to support `output_size=None`, which preserves source crop pixels. The default remains 64x64 for compatibility. It adds a high-resolution wrapper with frame-content SHA-256, exact dimensions, crop transform, phone coverage and boundary clipping. No production registry or active model was changed.
