# ROADWATCH AI85 — NEXT STAGE: DATA-CENTRIC PERFORMANCE RECOVERY

Bạn là Principal Computer Vision Researcher, Deep Learning Scientist và Senior MLOps Engineer.

Repository:
https://github.com/ln3076059-jpg/prok53

Commit baseline:
`a0c651330400e572db4d6d50f4466d3c4d200fb0`

Workspace:
`D:\.idea\giangdoantotnghiep`

## NHIỆM VỤ

Tiếp tục cải thiện Roadwatch từ kết quả AI85 hiện tại, hướng đến Precision, Recall, Event F1 và SAFE_DRIVING Recall tối thiểu 85%, đồng thời kiểm soát False Alarms/min.

Không được tuyên bố đạt mục tiêu khi chưa có đánh giá độc lập xác nhận.

Không lặp lại các thí nghiệm vừa thất bại nếu không có giả thuyết kỹ thuật mới.

### BASELINE ĐÃ XÁC MINH

- Event Precision: 76.19%.
- Event Recall: 55.17%.
- Event F1: 64.00%.
- False Alarms/min: 0.2266.
- SAFE_DRIVING coarse Recall: 12.57%.
- Clip Macro-F1: 34.23%.
- TP: 16.
- FP: 5.
- FN: 13.

Các thí nghiệm Mean Fine-tune, Temporal Attention và Native112 chưa cải thiện Event F1.

## PHASE 1 — ROOT CAUSE OF SAFE CLASSIFICATION FAILURE

Đọc trực tiếp:

- `reports/ai85_upgrade/06_SAFE_DRIVING_CLASSIFICATION.md`
- `reports/ai85_upgrade/04_FALSE_NEGATIVE_ANALYSIS.md`
- `reports/ai85_upgrade/error_analysis.json`
- `reports/ai85_upgrade/combined_metrics.json`

Điều tra vì sao 668 SAFE clips chỉ có 84 predictions đúng.

Đối chiếu:

1. Subject-level confusion matrices.
2. Training and validation class distributions.
3. Clip labels.
4. Driver ROI quality.
5. Train/inference preprocessing.
6. Confidence distributions.
7. Embedding separability.
8. Classification-head design.
9. Subject appearance bias.
10. Label consistency across camera views.

Không kết luận nguyên nhân chỉ từ confusion matrix.

Xuất bằng chứng cụ thể về failure mode.

## PHASE 2 — INVESTIGATE ALL 13 FALSE NEGATIVES

Đọc:

`reports/ai85_upgrade/fn_frame_manifest.json`

và original ROI evidence đã xuất.

Đối với từng FN:

- Xác minh visual evidence.
- Phân tích phone confidence theo thời gian.
- Phân tích coarse/fine action scores.
- Đo temporal activation consistency.
- Kiểm tra trường hợp event ngắn.
- Xác định tình trạng missing samples.
- Kiểm tra sự phân mảnh hoặc gộp sai.
- So sánh raw prediction với final event.

Không được chỉ dựa trên xác suất PHONE lớn nhất tại một frame.

Cần đánh giá evidence liên tục trong toàn bộ event interval.

Nếu thiếu phone boxes hoặc occupant association, bổ sung instrumentation vào feature-extraction pipeline và chạy lại trên dữ liệu hợp lệ.

## PHASE 3 — IMPROVE VISUAL REPRESENTATION

Các thí nghiệm trước cho thấy thay mean pooling bằng attention đơn thuần không giúp cải thiện.

Do đó ưu tiên nghiên cứu cải tiến visual representations.

Thử có kiểm soát:

A. Driver torso + hands multi-ROI representation.

B. Hand-to-face movement representation.

C. Phone-focused high-resolution crop.

D. Spatial-temporal motion features.

E. Frozen backbone embedding comparison.

F. Fine-tuning selected upper backbone layers nếu có GPU và dữ liệu đủ lớn.

G. Behavior-centric ROI inspired by DualScope.

H. Video action representations inspired by DriveCLIP.

Không bắt buộc huấn luyện mô hình lớn nếu tài nguyên thiếu.

Đo separability của SAFE, PHONE và OTHER trước khi thêm kiến trúc phức tạp.

Không mặc định chất lượng hình ảnh tốt hơn sẽ làm Event F1 tăng.

## PHASE 4 — DATA-CENTRIC LEARNING

Kiểm tra số subjects thực sự có quyền sử dụng.

Ưu tiên:

- New approved development subjects.
- Human-confirmed hard-negative examples.
- Improved SAFE_DRIVING annotations.
- Phone-use edge cases.
- Class consistency.
- Camera/subject diversity.
- Duplicate and overlap detection.

Thiết lập group-disjoint splits trước training.

Không sử dụng consumed holdouts:

- gZ-37.
- gE-28.

Không tự sửa hoặc sinh nhãn giả để tăng metric.

Nếu không có subjects mới, phải báo cáo rõ dữ liệu là bottleneck của nghiên cứu.

## PHASE 5 — TEMPORAL AND EVENT RECOVERY

Các thí nghiệm trước giữ nguyên threshold 0.80 và hai positive samples liên tiếp.

Không tối ưu threshold trực tiếp từ outer test labels.

Phân tích:

- Action probability continuity.
- Short-event coverage.
- Missing centered clips.
- Activation delay.
- Event boundary errors.
- Event overmerge.
- Duplicate false positives.

Nếu cần thay đổi temporal policy, phải chọn policy bằng dữ liệu validation tách biệt.

Không kéo dài predicted events một cách nhân tạo để tăng matching.

## PHASE 6 — SCIENTIFIC EXPERIMENTS

Chỉ triển khai những thí nghiệm có giả thuyết rõ ràng.

Mỗi thí nghiệm phải có:

- Dataset hash.
- Model hash.
- Config.
- Training seed.
- Subject split.
- Prediction files.
- Per-class confusion matrix.
- Event TP/FP/FN.
- Precision.
- Recall.
- F1.
- FA/min.
- SAFE_DRIVING Recall.
- Macro-F1.
- Per-subject breakdown.

Giữ nguyên baseline và historical benchmark.

Nếu một thí nghiệm giảm Event F1 hoặc tăng False Alarms, lưu kết quả thất bại thay vì xóa.

## PHASE 7 — NEW INDEPENDENT VALIDATION

Không tiếp tục sử dụng ba development subjects đã được thử nghiệm nhiều lần như bằng chứng independent performance.

Nếu có dữ liệu mới, hợp lệ:

1. Kiểm tra license.
2. Kiểm tra annotation.
3. Khóa subject-disjoint split.
4. Chọn model bằng development/validation.
5. Freeze model và threshold.
6. Đánh giá trên untouched holdout một lần.
7. Công bố confusion matrix, event counts, metrics và confidence intervals.

Nếu không có independent holdout:

`AI85_INDEPENDENT_VALIDATION = NOT_ESTABLISHED`

Không promote model lên production.

## PHASE 8 — EXECUTION AND DELIVERABLES

Không chỉ viết Markdown.

Phải thực hiện tối đa các công việc khả thi:

1. Audit source code và baseline.
2. Phân tích SAFE classification.
3. Phân tích 13 FN.
4. Bổ sung diagnostics còn thiếu.
5. Sửa code nếu xác định được nguyên nhân.
6. Thực hiện thí nghiệm có kiểm soát.
7. Chạy regression tests.
8. Đo metrics thực tế.
9. Tổng hợp trade-off.
10. Báo cáo blocker còn lại.

Lưu tại:

`reports/ai85_next/`

Bao gồm:

- `ROOT_CAUSE_ANALYSIS.md`
- `SAFE_CLASSIFICATION_ANALYSIS.md`
- `FN_RECOVERY.md`
- `VISUAL_REPRESENTATION_ABLATION.md`
- `DATASET_GENERALIZATION.md`
- `METRICS_COMPARISON.md`
- `TEST_RESULTS.md`
- `FINAL_REPORT.md`

MỤC TIÊU NGHIÊN CỨU:

- Event Precision >=85%.
- Event Recall >=85%.
- Event F1 >=85%.
- SAFE_DRIVING Recall >=85%.
- False Alarms/min <=0.15.

Đây là mục tiêu, không phải kết quả bảo đảm.

**BẮT ĐẦU BẰNG VIỆC XÁC ĐỊNH TẠI SAO SAFE_DRIVING BỊ NHẦM VÀ TẠI SAO 13 SỰ KIỆN PHONE-USE CHƯA ĐƯỢC PHÁT HIỆN. KHÔNG LẶP LẠI CÁC THÍ NGHIỆM AI85 ĐÃ THẤT BẠI.**