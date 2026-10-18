# Verified scientific literature

The full structured matrix is `literature_matrix.json`; fields that could not be confirmed are marked `NOT VERIFIED`.

- [SAHI, ICIP 2022](https://arxiv.org/abs/2202.06934) provides the strongest direct fit: overlapping slices, original-coordinate remapping and post-processing address small phones. Its VisDrone/xView AP gains are external detection results.
- [DualScope, AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/37812) combines behavior-centric spatial distillation, fine motion and temporal pyramids. The supplied GitHub returned unavailable and the checkpoint page could not be authenticated, so license/reproducibility remain unverified.
- [DriveCLIP](https://github.com/zahid-isu/DriveCLIP) supports the frame-based framework. The README samples video frames and predicts frames, while linking separate VideoCLIP code; it is not evidence of a complete Roadwatch temporal event pipeline.
- [ActionFormer, ECCV 2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136640485.pdf) predicts categories and temporal boundaries from multi-scale features with local self-attention. It uses pre-extracted features and is too costly before a small causal boundary baseline.
- [Multi-Task Fusion, Sensors 2025](https://www.mdpi.com/1424-8220/25/21/6799) uses 3D ResNet18 plus BiLSTM across BODY/FACE/HANDS tasks. Missing views must remain missing evidence, not negative evidence.
- [MixStyle](https://arxiv.org/abs/2104.02008) and [Group DRO](https://arxiv.org/abs/1911.08731) target domain/group robustness. Group DRO is rejected now because only three repeatedly used development subjects are available and the method needs group labels, regularization and early stopping.
- [ByteTrack](https://github.com/FoundationVision/ByteTrack) can preserve low-score boxes through association, but a track ID does not establish the driver's role.
- [RST-Net seatbelt](https://www.mdpi.com/2076-3417/14/8/3346) supports torso localization, pruning and Triplet Attention. Its private/self-built data and image F1 cannot be transferred to Roadwatch. The [overhead YOLOv7 study](https://ph.pollub.pl/index.php/acs/article/view/7594) has a roadside geometry mismatch.
- The current [DMD repository](https://github.com/Vicomtech/DMD-Driver-Monitoring-Dataset) says the 2026 release is real-car RGB only and CC BY-NC-ND 4.0. OpenLABEL/VCD annotations and BODY/FACE/HANDS exist, but local authorization still denies fitting/upload.
- [DGMB-Net](https://www.mdpi.com/1424-8220/24/5/1386) uses teacher-student semi-supervision, adaptive perceptual learning and a hierarchical feature pyramid on Bayonet-Drivers/State Farm. It is detection research from different camera/data conditions.

No paper classification accuracy, AP, mAP, MOTA or image F1 is reported as Roadwatch Event F1.
