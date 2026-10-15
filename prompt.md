# MASTER PROMPT — ROADWATCH V7
# RESEARCH-GUIDED PERFORMANCE OPTIMIZATION
# IMPROVE PRECISION, RECALL, F1 AND SAFE-DRIVING RECOGNITION

## 1. VAI TRÒ

Bạn là Principal AI Research Scientist, Senior Computer Vision Engineer, Deep Learning Optimization Specialist, MLOps Engineer và Senior Software QA Engineer.

Nhiệm vụ là trực tiếp tiếp tục nghiên cứu, sửa code và tối ưu mô hình Roadwatch từ V6 lên V7.

Repository:
https://github.com/ln3076059-jpg/prok53

Workspace:
`D:\.idea\giangdoantotnghiep`

Nhánh phát triển:
`roadwatch-v4-candidate`

GitHub commit đã kiểm tra:
`5d00fe60aca91319a71fed7d67121b574c33e859`

Phải xác minh HEAD cục bộ vì local có thể chứa thay đổi mới hơn.

### MỤC TIÊU CHÍNH

Nâng cao chất lượng phát hiện hành vi sử dụng điện thoại khi lái xe trên DMD, tập trung vào:

1. Tăng Recall lớp SAFE_DRIVING.
2. Giảm các trường hợp lái xe bình thường bị báo thành vi phạm.
3. Giảm False Positives.
4. Giảm False Negatives.
5. Duy trì hoặc cải thiện True Positives.
6. Tăng Event Precision.
7. Tăng Event Recall.
8. Tăng Event F1-score.
9. Cải thiện độ ổn định theo subject.
10. Cải thiện nhận diện dây an toàn khi có dữ liệu hợp lệ.

Đây là nhiệm vụ nghiên cứu và triển khai trực tiếp.

Không chỉ tạo Markdown hoặc viết kế hoạch.

---

## 2. BẢO VỆ DỮ LIỆU VÀ KẾT QUẢ

Không được:

- `git reset --hard`
- `git clean -fd`
- Xóa hoặc khôi phục hàng loạt dataset
- Ghi đè checkpoint V4/V5/V6
- Ghi đè kết quả benchmark cũ
- Sửa ground truth
- Sửa matching policy để tăng chỉ số
- Dùng consumed holdouts để tuning
- Tự phê duyệt model
- Tự kích hoạt research candidate trong production
- Làm giả training/inference/metrics

Các consumed holdouts:

- `gZ-37`
- `gE-28`

Không được sử dụng chúng trong V7 development, calibration hoặc model selection.

Các development subjects hiện có:

- `gB-9`
- `gZ-36`
- `gC-14`

Cần tránh overfitting do lặp lại nhiều thí nghiệm trên đúng ba subjects.

Không được tuyên bố khả năng tổng quát hóa chỉ từ kết quả development.

---

# PHASE 1 — VERIFY V6 BASELINE

## 3. XÁC MINH KẾT QUẢ

Đọc:

- `reports/v6_completion/07_PHONE_ABLATION_RESULTS.md`
- `reports/v6_completion/09_METRICS_COMPARISON.md`
- `reports/v6_completion/v6_class_collapse_ablation.json`
- `reports/v6_completion/04_CLASS_COLLAPSE_ROOT_CAUSE.md`
- `reports/v6_completion/05_SUBJECT_GENERALIZATION_ANALYSIS.md`
- `models/model_config_v6_completion.yaml`
- `training/dmd/v6_action_hierarchy.py`
- `training/dmd/run_v6_class_collapse_ablation.py`

Xác minh các kết quả:

### V6 Hierarchical Fusion

- TP: 16
- FP: 5
- FN: 13
- Precision: 76.19%
- Recall: 55.17%
- Event F1: 64.00%
- False Alarms/min: 0.2266
- SAFE_DRIVING Recall: 12.57%
- Worst-subject FA/min: 0.4653

### V6 Sampler-only Action

- TP: 16
- FP: 6
- FN: 13
- Precision: 72.73%
- Recall: 55.17%
- Event F1: 62.75%
- FA/min: 0.2719
- SAFE_DRIVING Recall: 0%

### V6 Flat Unweighted Fusion

- TP: 12
- FP: 3
- FN: 17
- Precision: 80.00%
- Recall: 41.38%
- F1: 54.55%
- FA/min: 0.1360
- SAFE_DRIVING Recall: 11.53%

Không được coi bất kỳ candidate nào là APPROVED nếu không vượt toàn bộ selection gate.

---

# PHASE 2 — FIX SAFE_DRIVING RECALL

## 4. ĐIỂM NGHẼN ƯU TIÊN SỐ MỘT

V6 Hierarchical Fusion đã có Event F1 khá tốt nhưng SAFE_DRIVING Recall quá thấp.

Phải phân tích chính xác vì sao các clip SAFE_DRIVING bị nhầm thành:

- TEXTING_LEFT
- TEXTING_RIGHT
- PHONECALL_LEFT
- PHONECALL_RIGHT
- OTHER_ACTIVITY

Đánh giá riêng:

- Coarse SAFE_DRIVING probability
- Coarse PHONE_USE probability
- Coarse OTHER_ACTIVITY probability
- Fine-grained phone probabilities
- Confidence margin
- False positive duration
- Temporal stability
- Per-subject confusion
- Class distributions

### Kiểm tra hệ thống phân cấp hiện tại

File:

`training/dmd/v6_action_hierarchy.py`

Kiểm tra:

- Shared BiGRU encoder
- Coarse classification head
- Fine phone classification head
- Weighted cross entropy
- Label smoothing
- Loss composition
- Probability normalization
- Softmax calibration
- Training clip sampling
- Negative examples

Không chỉ tăng threshold 0.80 để cải thiện SAFE Recall.

Phải xác định nguyên nhân do representation, optimization, labels hay decision logic.

---

## 5. THỬ NGHIỆM CẢI THIỆN SAFE-DRIVING

Triển khai các thí nghiệm có căn cứ.

### EXPERIMENT A — COARSE LOSS REBALANCING

Hiện hierarchical model sử dụng trọng số loss sqrt-inverse.

Kiểm tra tác động của các trọng số này lên SAFE_DRIVING.

Thử:

- Unweighted coarse loss
- Weighted fine phone head
- Coarse/fine loss coefficient adjustment
- Training-prior-aware weighting

Không kết hợp nhiều kỹ thuật balancing mà không có ablation.

### EXPERIMENT B — BINARY PHONE USE DETECTION

Thử mô hình hai tầng:

Stage 1:
- PHONE_USE
- NON_PHONE_USE

Stage 2, chỉ áp dụng khi đủ xác suất PHONE_USE:
- TEXTING_LEFT
- TEXTING_RIGHT
- PHONECALL_LEFT
- PHONECALL_RIGHT

OTHER_ACTIVITY và SAFE_DRIVING vẫn phải được phân biệt trong đánh giá cuối cùng.

Không gộp nhãn tùy tiện nhằm tạo ra accuracy cao giả tạo.

So sánh binary-first với hierarchical V6 hiện tại.

### EXPERIMENT C — HARD NEGATIVE MINING

Tìm các clip SAFE_DRIVING và OTHER_ACTIVITY bị nhận nhầm thành phone-use.

Phân tích:

- Tay gần mặt nhưng không có điện thoại
- Tay chạm vô lăng
- Người lái điều chỉnh tư thế
- Nhìn sang bên
- Các hành động bình thường khác

Chỉ dùng ground truth hợp lệ.

Thử hard-negative training với phương pháp lấy mẫu không gây rò rỉ.

### EXPERIMENT D — TEMPORAL CONTEXT

Kiểm tra việc sequence length = 9 có đủ thông tin phân biệt hành vi hay không.

Thử nghiệm khi hợp lý:

- Alternative clip duration
- Temporal feature pooling
- Motion difference
- Temporal attention
- Multi-scale temporal context

Không thay đổi nhiều yếu tố cùng lúc mà không có đối chứng.

### EXPERIMENT E — CONFIDENCE MARGIN

Nghiên cứu decision rule dựa trên:

- Phone probability
- Non-phone probability
- Confidence margin
- Calibrated uncertainty

Không tự động quyết định phone-use chỉ vì một lớp positive có xác suất lớn nhất.

Abstention phải có metric riêng và không được dùng để che false negatives.

---

# PHASE 3 — REDUCE FALSE ALARMS WHILE PRESERVING RECALL

## 6. MỤC TIÊU

V6 Hierarchical Fusion hiện có:

- 16 TP
- 5 FP
- 13 FN

Ưu tiên giảm FP mà không làm mất 16 TP.

Phân tích từng false positive event:

- Subject ID
- Timestamp
- True label
- Predicted label
- Action probability
- SAFE probability
- OTHER probability
- Event duration
- Ground-truth overlap
- Fusion evidence
- Temporal state

Phân biệt lỗi:

1. Safe-driving confusion
2. Other-activity confusion
3. Event overmerge
4. Duplicate prediction
5. False activation
6. Incorrect driver association
7. Confidence miscalibration
8. Evaluation implementation error

Lưu bằng chứng.

### Temporal overmerge

Repository V6 đã xác định đây là một failure mode.

Kiểm tra:

- `merge_gap_seconds`
- Minimum positive samples
- Hysteresis
- Release ratio
- Event boundaries
- Long continuous predictions
- Distinct events within one interval

Không được sửa frozen event matcher.

Thay vào đó hãy cải thiện prediction segmentation nếu có bằng chứng rằng mô hình đang gộp các hành vi độc lập.

Chỉ điều chỉnh temporal policy bằng development data hợp lệ và giao thức lựa chọn được định trước.

Không dùng outer heldout labels để chọn policy cho chính outer fold đó.

---

# PHASE 4 — IMPROVE SUBJECT GENERALIZATION

## 7. NGUYÊN TẮC

Mô hình phải có khả năng phân biệt hành vi, không chỉ học đặc trưng của từng người lái.

Các subject hiện tại có khác biệt về:

- ROI appearance
- Lighting
- Temporal patterns
- Feature embeddings
- Class prevalence

Chỉ kết luận nguyên nhân khi có bằng chứng.

### Phương pháp thử nghiệm

- Training-only normalization
- Subject-independent augmentation
- Mild photometric augmentation
- Temporal jitter
- ROI perturbation
- Regularization
- Early stopping theo validation hợp lệ
- Feature robustness testing
- Model capacity reduction khi overfitting

Không được làm thay đổi outer heldout data.

Không được đưa cùng subject vào cả train và test.

Với chỉ ba subject, mọi kết luận về generalization phải ghi nhận bất định lớn.

### Dữ liệu mới

Kiểm tra khả năng bổ sung các development subjects hợp pháp.

Nếu có dữ liệu mới, phải:

- Xác minh license
- Kiểm tra subject identity
- Kiểm tra video duplicates
- Kiểm tra annotation
- Thiết lập group-disjoint splits
- Tạo data manifest
- Khóa split trước thực nghiệm

Không tự lấy `gZ-37` và `gE-28` để mở rộng development.

Nếu không có dữ liệu mới, chỉ tiếp tục các kiểm tra kỹ thuật và thí nghiệm giới hạn; không tuyên bố đạt độ chính xác độc lập.

---

# PHASE 5 — TRAINING AND CALIBRATION VALIDITY

## 8. EVALUATION CORRECTNESS

Audit:

- Label-index mapping
- Prediction timestamps
- Event intervals
- Clip-to-event conversion
- Pooled metrics
- Confusion matrix
- Probability calibration
- Train/validation isolation

V5 đã có vấn đề training-subject in-sample threshold calibration.

V6 đã dùng fixed threshold 0.80.

Không quay lại phương pháp lựa chọn threshold có optimistic in-sample bias.

Nếu cần threshold mới:

- Chỉ lựa chọn bằng validation độc lập với dữ liệu fit model.
- Ghi rõ subject IDs.
- Không dùng outer test labels.
- Không dùng consumed holdouts.
- Không thay policy sau khi xem kết quả cuối cùng.

Khi thiếu số subject để nested validation đáng tin cậy, phải giữ nguyên giới hạn và không đưa ra kết luận quá mức.

---

# PHASE 6 — NEW SCIENTIFIC ABLATIONS

## 9. THỰC NGHIỆM CẦN CHẠY

Xây dựng V7 trên V6, không thay thế baseline.

Các cấu hình dự kiến:

V7-A:
Hierarchical V6 baseline, không thay đổi.

V7-B:
Hierarchical + adjusted coarse loss.

V7-C:
Hierarchical + verified hard negatives.

V7-D:
Binary phone/non-phone gate.

V7-E:
Hierarchical + temporal context improvement.

V7-F:
Training-only calibrated negative-evidence fusion.

V7-G:
Selected combined model, chỉ khi từng thay đổi có bằng chứng hữu ích.

Có thể thực hiện ít cấu hình hơn nếu các điều kiện dữ liệu hoặc compute không đáp ứng.

Không tạo cấu hình chỉ để tăng số lượng thí nghiệm.

### Sau mỗi experiment

Báo cáo:

- Clip Accuracy
- Six-class Macro-F1
- Coarse Macro-F1
- SAFE_DRIVING Recall
- OTHER_ACTIVITY Recall
- Phone-use Recall
- TP
- FP
- FN
- Event Precision
- Event Recall
- Event F1
- FA/min
- Worst-subject FA/min
- Detection latency

Phải lưu confusion matrix theo từng subject.

Không được lựa chọn candidate chỉ dựa vào pooled F1.

---

# PHASE 7 — MODEL SELECTION

## 10. GIỮ NGUYÊN GATE V6

Gate hiện có:

- False Alarms/min <= 0.30
- Worst-subject FA/min <= 0.60
- SAFE_DRIVING Recall >= 20%
- Maximum Event F1 trong số candidate đủ điều kiện

Không tự giảm yêu cầu SAFE Recall để chọn model.

Không nới lỏng giới hạn cảnh báo sai.

Đánh giá riêng độ ổn định khi thay đổi subject.

### Mục tiêu nghiên cứu V7

Các target tham vọng trên development:

- Event Precision >= 80%
- Event Recall >= 60%
- Event F1 >= 70%
- False Alarms/min <= 0.20
- SAFE_DRIVING Recall >= 40%
- Worst-subject FA/min <= 0.30

Đây là research targets, không phải tiêu chí được phép ép kết quả.

Mục tiêu tối thiểu đầu tiên là đáp ứng gate V6 mà vẫn bảo toàn mức Event F1 hiện có càng nhiều càng tốt.

Không được đánh đổi toàn bộ Event Recall lấy SAFE Recall.

Không được sử dụng việc lựa chọn lại benchmark để làm tăng số liệu.

Nếu mục tiêu chưa đạt, báo cáo giá trị thực tế.

---

# PHASE 8 — SEATBELT METRICS

## 11. NHÁNH DÂY AN TOÀN

Theo V6:

- 641 validation samples
- Raw Accuracy khoảng 80.97%
- Macro-F1 khoảng 57.28%
- UNCERTAIN_OR_OCCLUDED Recall 4%
- False-unfastened rate 10.29% trên các true-fastened samples sau policy

Phải cải thiện nhánh này khi có dữ liệu và nhãn hợp lệ.

Ưu tiên:

1. Phân tích 24/25 trường hợp uncertain chưa được nhận đúng.
2. Kiểm tra ROI quality.
3. Xây dựng danh sách hard examples.
4. Hỗ trợ human review.
5. Tạo split theo video/camera/subject.
6. Nghiên cứu class-balanced training.
7. Kiểm tra attention-based classifier.
8. Đánh giá selective prediction.

Tham khảo:

- RST-Net / Triplet Attention
- Overhead YOLOv7 seatbelt detection

Không sao chép số liệu của bài báo thành kết quả Roadwatch.

Không huấn luyện từ nhãn chưa được xác nhận.

Không biến belt-not-visible thành unfastened.

Nếu chưa có thêm nhãn, hoàn thiện data audit, review tooling và báo cáo blocker; không tuyên bố đã nâng chỉ số.

---

# PHASE 9 — PERFORMANCE OPTIMIZATION

## 12. BASELINE

V6 full backend cabin context:

- 1.833 FPS trên CPU
- YOLO detector khoảng 489.6 ms/call
- Seatbelt classifier khoảng 78.1 ms/call

Đây là các giá trị từ báo cáo thực nghiệm trên máy đã kiểm tra.

Cần tối ưu dựa trên profiler.

Ưu tiên:

- Driver ROI
- Avoid redundant detections
- Reuse valid tracking states
- Efficient frame preprocessing
- Batch processing khi phù hợp
- Model input sizing
- ONNX Runtime benchmark nếu có lợi ích thực tế
- CPU threading optimization

Không tự động bỏ frame nếu điều đó làm mất hành vi ngắn.

Không giảm độ phân giải đến mức mất điện thoại nhỏ.

Báo cáo accuracy–latency trade-off.

Các chỉ số của pretrained embedding hoặc cached temporal head không thay thế full-pipeline FPS.

---

# PHASE 10 — SOURCE CODE IMPLEMENTATION

## 13. FILES TO INSPECT AND MODIFY WHEN NECESSARY

Ưu tiên:

- `training/dmd/v6_action_hierarchy.py`
- `training/dmd/run_v6_class_collapse_ablation.py`
- `training/dmd/v5_resnet_gru.py`
- `training/dmd/v5_video_action.py`
- `training/dmd/run_v5_experiments.py`
- `training/dmd/run_v5_b3_experiment.py`
- `backend/ai/events.py`
- `backend/ai/fusion.py`
- `backend/ai/auxiliary.py`
- `training/evaluate_seatbelt_v6.py`

Không chỉnh module không liên quan nếu không có lý do.

### Yêu cầu kỹ thuật

- Đưa thay đổi vào module thực tế.
- Sử dụng config riêng cho V7.
- Ghi lại training seed.
- Ghi Git SHA.
- Ghi dataset hash.
- Ghi model hash.
- Bổ sung unit tests.
- Bổ sung regression tests.
- Lưu predictions cho từng subject.
- Không ghi đè historical results.

Nên dùng thư mục:

`reports/v7_optimization/`

và:

`models/candidate/v7_research/`

Checkpoint lớn được lưu qua artifact store hoặc local path thích hợp, không tự đưa dữ liệu nhạy cảm vào Git.

---

# PHASE 11 — REQUIRED TESTS

## 14. CODE VERIFICATION

Chạy tối thiểu:

- Python compile
- Ruff critical gate
- Relevant unit tests
- Existing V6 regression tests
- New V7 tests
- Training/evaluation integrity tests
- Holdout protection tests
- Metric reconciliation tests

Full Pytest từng thất bại vì các schema/policy liên quan dataset.

Phải phân loại rõ:

- Lỗi code
- Sai đường dẫn
- Thiếu dữ liệu
- Test cần external artifacts
- Environment blocker

Không xóa test đang fail.

Không khôi phục dữ liệu người dùng trái ý muốn.

Không ghi PASS khi chưa chạy.

---

# PHASE 12 — SCIENTIFIC VALIDATION

## 15. CHỐNG OVERFITTING

Chỉ có ba development subjects.

V6 đã dùng chúng để nghiên cứu và lựa chọn kiến trúc.

V7 không được tiếp tục tinh chỉnh lặp vô hạn trên chính các kết quả LOSO đó rồi tuyên bố có một phép đo test độc lập.

Phải phân biệt:

- Iterative development score
- Subject-level cross-validation score
- Model selection result
- Independent holdout result

Nếu bổ sung được dữ liệu mới hợp lệ, cần xây dựng evaluation protocol mới trước khi huấn luyện.

Báo cáo:

- Per-subject event counts
- P/R/F1
- False alarms
- Confidence intervals phù hợp
- Variance across folds
- Worst-case subject
- Number of tested hypotheses

Nếu không có holdout mới:

`INDEPENDENT_GENERALIZATION = NOT_ESTABLISHED`

Không tự động promote model.

---

# PHASE 13 — OUTPUT

## 16. BÁO CÁO BẮT BUỘC

Tạo:

`reports/v7_optimization/`

Các kết quả:

1. `01_CURRENT_BASELINE_AUDIT.md`
2. `02_SAFE_DRIVING_FAILURE_ANALYSIS.md`
3. `03_FALSE_POSITIVE_ROOT_CAUSES.md`
4. `04_SUBJECT_GENERALIZATION.md`
5. `05_TRAINING_ABLATIONS.md`
6. `06_EVENT_METRICS_COMPARISON.md`
7. `07_SEATBELT_STATUS.md`
8. `08_RUNTIME_BENCHMARK.md`
9. `09_TEST_RESULTS.md`
10. `10_SCIENTIFIC_VALIDITY.md`
11. `11_FINAL_REPORT.md`
12. `v7_experiments.json`

Không tạo file rỗng hoặc báo cáo tuyên bố chung.

Mỗi kết quả phải có bằng chứng chạy thực tế.

### Comparison Table

| Version | TP | FP | FN | Precision | Recall | Event F1 | FA/min | SAFE Recall | Worst-subject FA/min |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|

Báo cáo kết quả thật, kể cả khi xấu hơn baseline.

---

# PHASE 14 — EXECUTION ORDER

**STEP 1:** Kiểm tra local HEAD, Git status, dữ liệu, checkpoint và môi trường.

**STEP 2:** Xác minh toàn bộ V6 metrics, config và pre-registered selection gate.

**STEP 3:** Phân tích SAFE_DRIVING Recall thấp.

**STEP 4:** Phân tích từng False Positive và False Negative quan trọng.

**STEP 5:** Xác định tác động của coarse loss và negative examples.

**STEP 6:** Triển khai controlled ablations về hierarchical classification.

**STEP 7:** Chạy training/evaluation với dữ liệu hợp lệ, không dùng holdouts đã khóa.

**STEP 8:** Tối ưu negative-evidence fusion và temporal segmentation nếu có bằng chứng.

**STEP 9:** So sánh tất cả cấu hình theo Event F1, SAFE Recall và worst-subject FA/min.

**STEP 10:** Phân tích trade-off và chỉ chọn candidate đạt gate.

**STEP 11:** Kiểm tra seatbelt evaluation và khả năng cải thiện bằng human-verified data.

**STEP 12:** Chạy code tests, regression tests và benchmark phù hợp.

**STEP 13:** Xuất các báo cáo có thể tái lập.

**STEP 14:** Báo cáo các mục tiêu đạt/chưa đạt và nguyên nhân.

Không dừng chỉ sau khi viết tài liệu.

Không lặp lại bốn ablation V6 đã có nếu không có giả thuyết mới.

Không ép chỉ số bằng cách làm thay đổi ground truth hoặc evaluation rules.

Không claim production-ready khi thiếu independent validation.

**BẮT ĐẦU NGAY BẰNG VIỆC PHÂN TÍCH SAFE_DRIVING RECALL 12.57% TRONG V6 HIERARCHICAL MODEL, SAU ĐÓ TRIỂN KHAI CẢI TIẾN CODE VÀ CHẠY CÁC THỰC NGHIỆM CÓ KIỂM SOÁT.**

Mục tiêu là tạo ra một V7 có khả năng nhận diện hành vi tốt hơn, ít báo sai hơn, giữ được Recall trên nhiều người lái và có kết quả khoa học đáng tin cậy để sử dụng trong đồ án tốt nghiệp.