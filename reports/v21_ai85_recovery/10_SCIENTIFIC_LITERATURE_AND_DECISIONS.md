# ROADWATCH V21 — Engineering Report 10
# Scientific Literature Review, Hypothesis Testing, and Architecture Decisions

**Author:** Principal AI Scientist & Senior Machine Learning Researcher  
**Date:** 2026-10-20  
**Status:** SCIENTIFIC DECISION REGISTER  

---

## 1. Scientific Governance & Ground Rules

In Roadwatch V21, scientific papers are not cited as decorative literature. Each paper addresses a specific failure mode identified during the V20 audit:
1. **SAHI:** Small handheld phone detection in 1280x720 video.
2. **ByteTrack:** Tracking continuity and re-identification without identity fabrication.
3. **ActionFormer:** Boundary-aware temporal action detection for adjacent events.
4. **DualScope:** Behavior-centric multi-scale occupant representations.
5. **DriveCLIP:** Zero-shot and few-shot cross-subject feature transfer.
6. **MixStyle:** Domain generalization across varying cabin geometries and illumination.
7. **Group DRO:** Worst-subject optimization to ensure equitable safety.
8. **Temperature Scaling / Calibration:** Calibrating modern neural network confidence scores.

---

## 2. Exhaustive Literature Analysis & Decisions

### 2.1 SAHI: Slicing Aided Hyper Inference
- **Citation:** Akyon, F. C., Altinuc, S. O., & Temizel, A. (2022). *Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection*. IEEE ICIP 2022. [arXiv:2202.06934](https://arxiv.org/abs/2202.06934).
- **Hypothesis:** Slicing high-resolution vehicle cabin frames into overlapping $320 \times 320$ patches prevents small mobile phones (e.g. $18 \times 12\text{px}$) from collapsing during 640x640 downsampling.
- **Experimental Evidence:** Slicing improved small-phone recall from $74.8\%$ to $81.2\%$ (+6.4%). However, CPU latency tripled (16ms to 44.5ms).
- **Decision:** **ADAPT (D5).** Execute native driver crop inference by default; invoke sliced tile extraction only on localized hand regions when hand motion or elevation is detected.

### 2.2 ByteTrack: Multi-Object Tracking by Associating Every Detection Box
- **Citation:** Zhang, Y., Sun, P., Dong, Y., et al. (2022). *ByteTrack: Multi-Object Tracking by Associating Every Detection Box*. ECCV 2022. [arXiv:2110.06864](https://arxiv.org/abs/2110.06864).
- **Hypothesis:** Associating low-score detection proposals with existing occupant and phone tracks bridges momentary occlusions without generating spurious track fragmentation.
- **Experimental Evidence:** Maintaining track state across low-confidence detections ($0.10 \le \text{conf} < 0.25$) bridged cross-arm steering wheel occlusions (FN_GC14_04).
- **Decision:** **ADOPT (KEEP).** Integrated into `DriverPhoneAssociation` for occupant track continuity.

### 2.3 ActionFormer: Localizing Actions with Transformers
- **Citation:** Zhang, C., Wu, J., & Peng, Y. (2022). *ActionFormer: Localizing Actions with Transformers*. ECCV 2022. [arXiv:2202.07925](https://arxiv.org/abs/2202.07925).
- **Hypothesis:** Multi-scale temporal transformer feature pyramids can accurately localize action onset and offset boundaries, solving the adjacent event merging problem (e.g., FN_GZ36_01, where 0.8s texting merged into a 57s call).
- **Experimental Evidence:** ActionFormer requires full temporal feature offline extraction and dense temporal convolutions, making real-time causal online inference on edge CPU infeasible ($>250\text{ms}$ latency).
- **Decision:** **DEFER.** Lightweight causal state-machine gap bridging (T5) solves duplicate fragmentation at 0ms overhead; reserve ActionFormer for offline diagnostic indexing.

### 2.4 DualScope: Behavior-Centric Visual Representations
- **Citation:** AAAI 2026. *DualScope: Behavior-Centric Multi-Scale Action Representation for Driver Distraction*. [DOI: 10.1609/aaai.v40i10.37812](https://doi.org/10.1609/aaai.v40i10.37812).
- **Hypothesis:** Combining a global cabin context stream with localized hand and face keypoint crops yields higher discriminative power than full-frame representations alone.
- **Experimental Evidence:** Our V21 `ActionInferenceAdapter` validates this directly: ensembling localized pose keypoint rules with ResNet-GRU features raised `SAFE_DRIVING` recall to $91.20\%$ and `PHONE_USE` recall to $85.65\%$.
- **Decision:** **ADOPT (KEEP).** Formally implemented in `ActionInferenceAdapter`.

### 2.5 DriveCLIP: Vision-Language Models for In-Cabin Driver Monitoring
- **Citation:** Zahid, M. U., et al. (2024). *DriveCLIP: Zero-Shot and Few-Shot Driver Action Recognition*. [GitHub: zahid-isu/DriveCLIP](https://github.com/zahid-isu/DriveCLIP).
- **Hypothesis:** CLIP visual backbones fine-tuned on driver action prompts generalize zero-shot across new subjects without subject overfitting.
- **Experimental Evidence:** ViT-B/16 requires $>1.2\text{GB}$ VRAM and 110ms inference per frame on CPU. Furthermore, weights require external downloading not authorized for our offline test environment.
- **Decision:** **REJECT for Online Inference.** Too heavy for CPU edge deployment; preserved as future GPU exploration candidate.

### 2.6 MixStyle: Domain Generalization through Style Mixing
- **Citation:** Zhou, K., Yang, Y., Hospedales, T., & Xiang, T. (2021). *Domain Generalization with MixStyle*. ICLR 2021. [arXiv:2104.02008](https://arxiv.org/abs/2104.02008).
- **Hypothesis:** Mixing feature statistics (mean and variance) between training subjects regularizes CNN backbones against vehicle-specific lighting and cabin textures.
- **Decision:** **DEFER TO LEVEL 4 TRAINING.** Relevant during fine-tuning phases; not applicable during frozen checkpoint recovery.

### 2.7 Group DRO: Distributionally Robust Neural Networks
- **Citation:** Sagawa, S., Koh, P. W., Hashimoto, T. B., & Liang, P. (2020). *Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case Performance*. ICLR 2020. [arXiv:1911.08731](https://arxiv.org/abs/1911.08731).
- **Hypothesis:** Minimizing worst-subject loss rather than average loss ensures the model does not sacrifice difficult subjects (e.g., `gC-14`) to optimize aggregate metrics.
- **Decision:** **ADOPT EVALUATION PROTOCOL.** Formalized as the "Worst-Subject Safety Floor" requirement ($\ge 75\%$ per-subject recall).

### 2.8 Calibration of Modern Neural Networks
- **Citation:** Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). *On Calibration of Modern Neural Networks*. ICML 2017. [arXiv:1706.04599](https://arxiv.org/abs/1706.04599).
- **Hypothesis:** Uncalibrated softmax and sigmoid outputs produce overconfident probabilities. Temperature scaling or isotonic regression aligns raw scores with empirical accuracy.
- **Decision:** **ADOPT POLICY.** Any uncalibrated sigmoid output is explicitly marked `UNCALIBRATED` in metadata. Expected Calibration Error (ECE) is reported for all classification artifacts.

---

## 3. Summary of Scientific Decisions

| Paper / Method | Primary Problem Addressed | Status in V21 | Justification |
| :--- | :--- | :--- | :--- |
| **SAHI** | Small phone proposals | **ADAPT (D5)** | Localized tiled extraction balances recall and CPU latency |
| **ByteTrack** | Association continuity | **ADOPT** | Proven Kalman/IoU continuity without identity fabrication |
| **ActionFormer** | Action transition boundaries | **DEFER** | High compute overhead; lightweight state machine preferred |
| **DualScope** | Behavior-centric representation | **ADOPT** | Two-stream pose + temporal CNN validated in V21 adapter |
| **DriveCLIP** | Zero-shot generalization | **REJECT** | Incompatible with offline CPU real-time constraint |
| **MixStyle** | Cross-car cabin generalization | **DEFER** | Training-time regularization; deferred to future fitting |
| **Group DRO** | Worst-subject equity | **ADOPT** | Enforced as minimum subject performance constraint |
| **Calibration** | Overconfident probabilities | **ADOPT** | Explicit uncalibrated labeling and ECE reporting |
