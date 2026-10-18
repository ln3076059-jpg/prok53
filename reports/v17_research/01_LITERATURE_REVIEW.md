# 01. Literature Review: SOTA Driver Monitoring & Generalization

## 1. Overview
Roadwatch V17 addresses distracted driver detection (phone use, action recognition, seatbelt state) with a strict data-centric and model-centric approach. The following state-of-the-art methods form the scientific foundation for our proposed hypotheses:

| Model / Paper | Authors & Venue | Code & License | Dataset Evaluated | Key Findings | Hardware / Compute | Roadwatch Application | Limitations |
|---|---|---|---|---|---|---|---|
| **DualScope** | Qiu, Li, Zhang et al. (AAAI 2026) | Pending / Research | 7 DDAR benchmarks | SOTA on 7 distracted driving datasets using motion pyramid + posture decoupling | 4x V100 (est.) | Decouples static driver posture from dynamic hand/phone interaction | High complexity; requires localized ROI extraction |
| **DriveCLIP** | Hasan et al. (IEEE T-ITS 2024) | GitHub (MIT) | DMD, StateFarm, SynDD1, SAM-DD | Frozen vision-language embeddings achieve SOTA zero-shot/few-shot cross-dataset generalization | 1-2x A100 | Zero-shot action representation without requiring large labeled subject pools | Heavy backbone; requires lightweight linear probe/adapter |
| **SAHI** | Akyon et al. (ICIP 2022) | GitHub (Apache-2.0) | VisDrone, xView | Slicing-aided tiled inference increases small-object AP by +6% to +14% | Standard GPU | Source-resolution tiling for small handheld phones in cabin | Latency overhead; risk of increased false alarms without high precision |
| **ActionFormer** | Zhang, Wu, Li (ECCV 2022) | GitHub (Apache-2.0) | THUMOS14, ActivityNet | Multi-scale local Transformer head achieves 71.0% mAP@0.5 for temporal action localization | 4x V100 | Accurate temporal boundary estimation for sub-second phone use events | Requires continuous timestamped video annotations |
| **MixStyle** | Zhou et al. (ICLR 2021) | GitHub (MIT) | PACS, VLCS, DomainNet | Mixes feature statistics (mean, std) between subjects to learn domain-invariant representations | Minimal (<0.1ms) | Subject-invariant action recognition preventing driver-specific overfitting | Feature-level regularization only, not an end-to-end architecture |
| **Group-DRO** | Sagawa et al. (ICLR 2020) | Open Source | CelebA, MultiNLI, Waterbirds | Distributionally robust optimization minimizes worst-group loss, lifting worst-group accuracy by 10-40% | Moderate | Treats each driver as a group; directly targets Roadwatch worst-subject SAFE recall | Requires defined subject group IDs and careful step size tuning |
| **Multi-Task Fusion** | Zhang et al. (Sensors 2025) | Research | Multi-view cabin RGB | Multi-stream fusion of hands, face, and gaze yields robust event detection | 2x RTX 3090 | Bayesian / logistic fusion combining spatial proximity, action, and vehicle context | Requires calibrated multi-view camera sync |
| **RST-Net** | Qiu et al. (Appl. Sci. 2024) | Research | Road surveillance seatbelt | Attention mechanisms on torso crops boost seatbelt fastened/unfastened F1 | Standard GPU | Diagonal torso attention for robust seatbelt state classification | Needs separate UNCERTAIN / occluded calibration |
| **YOLOv7 Seatbelt** | Widodo et al. (Appl. Comput. Sci. 2025) | Research | Traffic surveillance RGB | High-angle surveillance detection of seatbelt presence under day/night lighting | 1x RTX 4090 | Auxiliary presence verification for traffic camera viewpoints | Fails when driver wears dark clothing matching the belt |

## 2. Theoretical Synthesis
To surpass the 85% barrier, Roadwatch must solve three coupled bottlenecks:
1. **Small-Object Detection Bottleneck:** Phones held near the ear or on laps span only 20-40 pixels in full cabin frames. Sliced inference (SAHI) with spatial edge penalties provides the necessary magnification.
2. **Action Ambiguity & Worst-Subject Collapse:** Existing E3 models collapse on outlier drivers (e.g., gZ-36 SAFE recall = 1.66%). Group-DRO and MixStyle enforce invariant feature learning across driver morphologies.
3. **Temporal Quantization & Thresholding Loss:** Events under 1.2s are rejected by naive windowing. Local temporal action heads (ActionFormer) allow sub-second event boundaries.
