# Seatbelt metrics và research gap

**V5 status: `NOT_RUN`.** Không có seatbelt temporal ground truth mới hoặc same-camera overhead dataset
hợp lệ trong scope, nên không huấn luyện RST-Net/YOLOv7 và không tuyên bố V5 improvement.

## Historical V2 validation — `MEASURED_ON_ROADWATCH`

Model SHA `e55158e3f152922e710ad260295da16488ffc2c6a145b8dfdc5c4ce323392bd4`, split validation 641 crops, raw Top-1 80.97%, macro-F1 57.27%.

| Class | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| seatbelt_fastened | 350 | 83.10% | 85.71% | 84.39% |
| seatbelt_unfastened | 266 | 78.70% | 81.95% | 80.29% |
| uncertain_or_occluded | 25 | 33.33% | 4.00% | 7.14% |

Lớp uncertain chỉ có 25 mẫu và recall 4.0%; đây là bottleneck nghiêm trọng. Reject policy fastened≥0.50,
unfastened≥0.60 giảm false-unfastened từ 49 xuống 36 nhưng post-policy uncertain F1 chỉ 3.57%. “Không
nhìn thấy dây” phải giữ `UNCERTAIN_OR_OCCLUDED`, không được biến thành `UNFASTENED`.

## Research design tiếp theo

1. Thu thập/human-review thêm uncertain/occluded theo vehicle/camera group.
2. So sánh same-split torso ROI baseline với lightweight attention/pruned ResNet; đo calibration/ECE và
   false violation rate ngoài per-class metrics.
3. Với camera ngoài đường, dùng two-stage vehicle→windshield/occupant→belt và group split theo video,
   location, day/night để tránh adjacent-frame leakage.
4. Chỉ promote khi uncertain recall và false-unfastened đạt gate trên independent camera holdout.
