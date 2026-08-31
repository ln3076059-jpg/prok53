# Research and source discovery

Research checked on 2026-08-31. Reported source metrics are context only and are not Roadwatch results.

| Source | Architecture / classes / size | License | Strengths | Limitations and use |
| --- | --- | --- | --- | --- |
| [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/) | YOLO11 detection family; COCO reference metrics; YOLO11s has 9.4M parameters | AGPL-3.0 or enterprise | Supports train, validate, predict, and export | No formal YOLO11 paper. Project fixes YOLO11s by prior design choice, not a sweep. Deployment licensing needs review. |
| [Ultralytics tracking](https://docs.ultralytics.com/modes/track/) | Detect-model tracking with ByteTrack or BoT-SORT | Ultralytics terms | Direct `bytetrack.yaml` integration | Tracking does not turn phone objects into behavior by itself. |
| [ByteTrack](https://github.com/FoundationVision/ByteTrack) | Association of high and low confidence boxes; MOT17 reference results | MIT | Simple, established baseline | Published MOT metrics do not transfer to this project. |
| [Robust Seatbelt Detection and Usage Recognition](https://arxiv.org/abs/2203.00810) | Driver-monitoring seatbelt use recognition | Paper metadata; dataset rights not established | Documents IR, fisheye, contrast, blur, and occlusion challenges | No source is admitted until exact downloadable license and compatible ROI semantics are verified. |
| [Kaggle Sintes seatbelt dataset](https://www.kaggle.com/datasets/alexandresintes/seatbelt-detection-dataset-real-car-photos) | YOLO-oriented front-seat photographs | CC BY 4.0 | Explicit license | Small, single-person domain; annotations require upper-body semantic review. Candidate only. |
| [Roboflow seatbelt-detection v4](https://universe.roboflow.com/seatbelttraining-7yh0f/seatbelt-detection-lb1ec/dataset/4) | 3,489 images shown by provider | CC BY 4.0 | Explicit version and license | Exact classes, provenance, and thin-belt box semantics require inspection. Candidate only. |
| [Hugging Face synthetic driver monitoring](https://huggingface.co/datasets/anywaylabs/synthetic-driver-monitoring-detection) | 1.36k synthetic object-detection rows shown by provider | CC BY 4.0 | Explicit license and synthetic domain diversity | Labels appear behavior-oriented and may not box physical phones or seatbelt-state upper bodies. Candidate only. |
| [State Farm competition](https://www.kaggle.com/competitions/state-farm-distracted-driver-detection/data) | 10 posture classes; competition page describes subject-separated train/test | Competition rules | Useful hard-negative and subject-split research reference | Access terms do not establish redistribution rights; classifier labels are not physical-phone boxes. Rejected pending legal review. |

The Advance Driver Monitoring System Kaggle dataset is rejected because its displayed license is unknown. No private CCTV, social-media scraping, or access bypass is permitted.

