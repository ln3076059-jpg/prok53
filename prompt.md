# ROADWATCH V9 — RESEARCH-DRIVEN PERFORMANCE RECOVERY

Bạn là Principal AI Research Scientist, Senior Computer Vision Engineer và MLOps Engineer.

Repository: https://github.com/ln3076059-jpg/prok53

Baseline GitHub đã xác minh: `0a730e5b02e2f41706baad254ddb952544dab945`

Hãy kiểm tra local HEAD trước khi triển khai và kế thừa mọi thay đổi mới hơn.

## MỤC TIÊU

Trực tiếp cải thiện mô hình Roadwatch để nghiên cứu khả năng đạt:

- Event Precision >= 85%
- Event Recall >= 85%
- Event F1 >= 85%
- False Alarms/min <= 0.15
- SAFE_DRIVING Recall >= 85%
- Seatbelt Macro-F1 >= 85%

Đây là mục tiêu nghiên cứu, không phải kết quả được bảo đảm.

Không sửa ground truth, event matching policy hoặc sử dụng holdout đã khóa để ép chỉ số.

## BASELINE

DMD controlled binary:
- TP = 16
- FP = 4
- FN = 13
- Precision = 80%
- Recall = 55.17%
- F1 = 65.31%

V8 temporal experiments T1–T5 chưa cải thiện F1.

Dense sampling không phát hiện được 4 sự kiện dưới 1 giây.

Seatbelt original Macro-F1 = 56.80%, UNCERTAIN Recall = 4%, với dữ liệu nhãn chưa hoàn toàn được phê duyệt.

## NHIỆM VỤ 1 — NGHIÊN CỨU KHOA HỌC

Đọc và phân tích:

1. DualScope — AAAI 2026:
https://ojs.aaai.org/index.php/AAAI/article/view/37812

2. Multi-Task Fusion — Sensors 2025:
https://doi.org/10.3390/s25216799

3. DriveCLIP:
https://github.com/zahid-isu/DriveCLIP

4. DGMB-Net — Sensors 2024:
https://doi.org/10.3390/s24051386

5. RST-Net Seatbelt:
https://doi.org/10.3390/app14083346

6. YOLOv7 Overhead Seatbelt:
https://ph.pollub.pl/index.php/acs/article/view/7594

Với mỗi bài báo, trích xuất kiến trúc, dataset, metric, phương pháp, mã nguồn, checkpoint và khả năng ứng dụng.

Không sao chép chỉ số của paper thành kết quả của Roadwatch.

## NHIỆM VỤ 2 — PHONE DETECTION RECOVERY

Điều tra 13 FN trên development data hợp lệ.

Tập trung:
- Phone ROI.
- Small-object detection.
- Hand–Phone Association.
- Phone–Face Geometry.
- Occlusion.
- Driver association.
- Confidence calibration.

Phân tích từ original video và dữ liệu có thật.

Bổ sung instrumentation khi cache thiếu phone boxes hoặc occupant IDs.

## NHIỆM VỤ 3 — VIDEO ACTION RECOGNITION

Không tiếp tục tăng sampling density đơn thuần.

Nghiên cứu:
- Behavior-centric spatial representation.
- Motion differences.
- Multi-scale temporal modeling.
- Lightweight visual encoder.
- ResNet/GRU hoặc BiLSTM.
- CLIP feature comparison.
- SAFE/PHONE/OTHER discrimination.

Thiết lập ablation, giữ nguyên baseline, đo clip Macro-F1 và Event F1 riêng biệt.

## NHIỆM VỤ 4 — FUSION

Kết hợp:
- Object evidence.
- Action probabilities.
- Pose/geometry.
- Temporal continuity.
- Camera reliability.
- Negative evidence.

Không tự tạo cảnh báo từ tín hiệu camera phụ thiếu ngữ cảnh.

Không tăng precision bằng cách bỏ sót nhiều sự kiện hơn.

Đo recovered TP, lost TP, new FP và removed FP sau mỗi thay đổi.

## NHIỆM VỤ 5 — SEATBELT

Kiểm tra nhãn UNCERTAIN và nguồn gốc dataset.

Ưu tiên human-approved labels trước retraining.

Thử torso ROI, Triplet Attention, hard negatives và calibration bằng ablation.

Không biến invisible belt thành UNFASTENED.

Đánh giá per-class F1, Macro-F1, balanced accuracy, false-unfastened rate và abstention rate.

## NHIỆM VỤ 6 — SCIENTIFIC EVALUATION

Không sử dụng gZ-37 hoặc gE-28 cho tuning.

Không tiếp tục overfit ba development subjects cũ.

Nếu có dữ liệu mới hợp lệ:
- Group-disjoint train/validation/test.
- Khóa test split.
- Freeze model và threshold.
- Đánh giá trên independent holdout.
- Báo cáo confidence interval và per-subject metrics.

Nếu không có dữ liệu mới, đánh dấu kết quả development-only.

## NHIỆM VỤ 7 — CODE, TEST VÀ DELIVERABLES

Trực tiếp sửa mã, thêm unit/regression tests, chạy benchmark khi đủ dữ liệu và compute.

Bảo toàn dữ liệu, checkpoint và báo cáo lịch sử.

Tạo `reports/v9_research/` gồm:
- Literature review.
- Data/label quality audit.
- FN/FP root causes.
- Representation experiments.
- Fusion ablations.
- Seatbelt experiments.
- Event metrics comparison.
- Runtime benchmark.
- Test results.
- Final scientific report.

Báo cáo chỉ số đã đo và cả những thí nghiệm thất bại.

Không tự phê duyệt production model.

BẮT ĐẦU TỪ PHÂN TÍCH 13 FN, CHẤT LƯỢNG ROI/FEATURE VÀ CÁC NHÃN CẦN HUMAN REVIEW. CHỈ FINE-TUNE SAU KHI XÁC MINH ĐƯỢC DỮ LIỆU VÀ GIẢ THUYẾT CẢI TIẾN.

Mục tiêu cuối cùng là một hệ thống có khả năng phát hiện đúng, ít cảnh báo sai, được đánh giá khoa học và có thể chứng minh kết quả trước hội đồng đồ án.