# Seatbelt Evaluation and Improvement

The existing real YOLO11s classifier was independently recomputed from 641 saved probabilities (source SHA-256 `22d57b98ebcfc533ebe88220e692ac21181b74e1f81bf5cf3cf36e06a0d777c5`). This is validation evidence, not a new independent holdout.

| Class | Support | Raw P | Raw R | Raw F1 | Policy P | Policy R | Policy F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| seatbelt_fastened | 350 | 83.10% | 85.71% | 84.39% | 83.10% | 85.71% | 84.39% |
| seatbelt_unfastened | 266 | 78.70% | 81.95% | 80.29% | 81.53% | 76.32% | 78.83% |
| uncertain_or_occluded | 25 | 33.33% | 4.00% | 7.14% | 3.23% | 4.00% | 3.57% |

- Raw accuracy: 80.97%; raw macro-F1: 57.28%; balanced accuracy: 57.22%.
- Post-policy macro-F1: 55.60%; balanced accuracy: 55.34%.
- False-unfastened: 36 / 350 true-fastened (10.29%).
- Output uncertain/abstention rate: 4.84%; threshold-forced rate: 4.37%.
- Top-1 10-bin ECE: 0.0846.

The uncertain class has only 25 examples and 4% recall. The reject policy reduces false-unfastened count from 49 to 36 but lowers overall three-class macro-F1; it is a safety abstention mechanism, not demonstrated class improvement. No retraining was performed because new human-verified uncertain examples were unavailable. Required next step: label torso crops for dark clothes/belts, bag straps, reflections, occlusion, viewpoints and low resolution, then evaluate on a camera-independent split. “Belt not visible” must remain uncertain, never automatically unfastened.
