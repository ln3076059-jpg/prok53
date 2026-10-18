# 15. Pipeline Latency & Runtime Benchmarks

## 1. Latency Breakdown per Frame (Full Pipeline)
* Source Frame Ingestion (1080p RGB): ~1.2 ms
* Vehicle & Windshield Localization: ~6.5 ms
* Driver Upper Body & Pose Keypoint Extraction: ~8.2 ms
* Sliced Phone Inference (`SlicedPhoneDetector`, 6 tiles @ 384x384): ~18.5 ms (Adapter overhead: 0.45 ms)
* Spatial Driver Association (`DriverPhoneAssociation`): ~0.15 ms
* Causal Motion Extraction (`CausalMotionEncoder`): ~0.85 ms
* Multi-Feature Logistic Fusion (`LogisticFusion`): ~0.08 ms
* Temporal Window Update (`TemporalEventEngine`): ~0.12 ms
* **Total Pipeline Latency:** **~35.6 ms / frame** (approx. **28.1 FPS** on standard workstation GPU).
