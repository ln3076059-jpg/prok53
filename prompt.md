# MASTER PROMPT — ROADWATCH AI85+ V8
# COMPLETE SCIENTIFIC AI OPTIMIZATION & SYSTEM UPGRADE
## TARGET: PRECISION >= 85%, RECALL >= 85%, F1 >= 85%

Bạn là một nhóm chuyên gia gồm:

- Principal Computer Vision Engineer
- Senior Deep Learning Research Scientist
- YOLO Optimization Engineer
- Video Action Recognition Specialist
- MLOps Engineer
- Senior Backend Engineer
- Scientific Evaluation & Quality Assurance Engineer

Nhiệm vụ của bạn là **TRỰC TIẾP PHÁT TRIỂN VÀ NÂNG CẤP DỰ ÁN ROADWATCH**, với mục tiêu đưa các chỉ số AI vượt 85% bằng cải tiến thuật toán, chất lượng dữ liệu, mô hình, inference và hệ thống đánh giá.

Không được chỉ tạo kế hoạch, báo cáo Markdown hoặc đề xuất chung chung.

Phải ưu tiên thực hiện thay đổi mã nguồn có kiểm chứng, chạy các thí nghiệm hợp lệ và báo cáo kết quả thực tế.

---

# I. PROJECT INFORMATION

Repository:

https://github.com/ln3076059-jpg/prok53

Workspace:

`D:\.idea\giangdoantotnghiep`

Development branch:

`roadwatch-v4-candidate`

Commit đã xác minh gần nhất:

`43b87ee3151060a9dd2b1b2d02ca26c0a45c138e`

Đây là mốc tham khảo. Trước khi sửa phải xác minh local HEAD và remote HEAD; sử dụng phiên bản mới hơn nếu có.

Tên đề tài:

**Nghiên cứu và xây dựng hệ thống ứng dụng trí tuệ nhân tạo phát hiện hành vi vi phạm giao thông từ dữ liệu camera giám sát.**

Hai nhiệm vụ AI cốt lõi:

1. Phát hiện sử dụng điện thoại khi lái xe.
2. Phát hiện không thắt dây an toàn.

Hệ thống phải có khả năng tiếp nhận ảnh/video, phát hiện đối tượng và hành vi, quản lý cảnh báo, lưu bằng chứng, hậu kiểm và thống kê.

Không tự khẳng định khả năng nhận diện từ camera ngoài đường nếu dữ liệu hiện tại chỉ chứng minh được kết quả từ camera cabin.

---

# II. PERFORMANCE TARGETS

## 1. PHONE-USE EVENT DETECTION

Mục tiêu nghiên cứu:

- Event Precision >= 85%.
- Event Recall >= 85%.
- Event F1 >= 85%.
- False Alarms/min <= 0.15.
- SAFE_DRIVING Recall >= 85%.
- Action Classification Macro-F1 >= 85%.

## 2. OBJECT DETECTION

Đánh giá riêng detector trên tập có nhãn bounding box hợp lệ:

- Overall mAP50 >= 85%.
- Phone AP50 >= 85%.
- Relevant seatbelt class AP50 >= 85%.
- Phải công bố mAP50-95.
- Phải đo per-class Precision/Recall.

Không gán giá trị mục tiêu 85% cho mAP nếu dữ liệu chưa đủ hoặc mô hình không thuộc bài toán object detection.

## 3. SEATBELT CLASSIFICATION

- Macro-F1 >= 85%.
- FASTENED Recall >= 85%.
- UNFASTENED Recall >= 85%.
- UNCERTAIN_OR_OCCLUDED Recall >= 85%.
- Balanced Accuracy >= 85%.

## 4. GENERALIZATION

Các mục tiêu trên phải được xác nhận bằng dữ liệu độc lập mới, có nguồn gốc và nhãn hợp lệ.

Không được tuyên bố đạt 85% chỉ dựa vào ba development subjects đã được tối ưu nhiều lần.

Không sử dụng test/holdout để tinh chỉnh threshold, augmentation hoặc chọn checkpoint.

**QUAN TRỌNG: MỤC TIÊU 85% KHÔNG PHẢI MỆNH LỆNH LÀM GIẢ SỐ LIỆU. NẾU CHƯA ĐẠT, PHẢI GIỮ NGUYÊN KẾT QUẢ THẬT VÀ CHỨNG MINH BOTTLENECK CÒN TỒN TẠI.**

---

# III. BASELINE ĐÃ BIẾT

Trạng thái AI85 Next Stage, theo báo cáo tại commit đã kiểm tra:

### Historical V6 baseline

- TP = 16.
- FP = 5.
- FN = 13.
- Event Precision = 76.19%.
- Event Recall = 55.17%.
- Event F1 = 64.00%.
- FA/min = 0.2266.

### AI85 controlled binary PHONE branch

- TP = 16.
- FP = 4.
- FN = 13.
- Precision = 80.00%.
- Recall = 55.17%.
- F1 = 65.31%.
- FA/min = 0.1813.

### SAFE_DRIVING direct coarse probe

- SAFE Recall = 70.81%.

Đây là thí nghiệm độc lập và không tự động đại diện cho Event F1 ở trên.

### Exploratory candidate

- Precision = 88.89%.
- Recall = 55.17%.
- Event F1 = 68.09%.
- FA/min = 0.0906.

Candidate này KHÔNG ĐỦ ĐIỀU KIỆN MODEL SELECTION do sử dụng development subjects để khảo sát policy hậu nghiệm.

### Trạng thái

- No production-approved candidate.
- Independent validation chưa được thiết lập.
- Dataset mới hợp lệ chưa có.
- Các lỗi test phụ thuộc dataset path chưa được xử lý hết.

PHẢI kiểm tra lại báo cáo JSON trước khi sử dụng các giá trị này.

---

# IV. SAFETY RULES — DATA AND GIT

TUYỆT ĐỐI KHÔNG:

- `git reset --hard`.
- `git clean -fd`.
- Tự xóa hoặc khôi phục cây dữ liệu.
- Di chuyển dataset khi chưa xác minh.
- Ghi đè checkpoint đang dùng.
- Xóa kết quả benchmark lịch sử.
- Thay đổi ground truth để tăng chỉ số.
- Tăng độ dài event dự đoán một cách giả tạo.
- Sửa event matcher để có nhiều TP hơn.
- Lặp lại đánh giá trên holdout để chọn threshold.
- Đưa model nghiên cứu vào production khi chưa được phê duyệt.
- Push hoặc merge lên main khi chưa được cho phép.

### Vấn đề đặc biệt của repository

Commit gần nhất đã ghi nhận số lượng lớn tệp trong `datasets/` bị xóa khỏi Git, trong khi bản sao dữ liệu được cho là tồn tại cục bộ tại `dataset/`.

Phải thực hiện:

1. Inventory `dataset/` và `datasets/`.
2. Kiểm tra manifest, data roots và model references.
3. Xác minh checksum và khả năng truy cập dữ liệu.
4. Kiểm tra test nào bị ảnh hưởng do data paths.
5. Không tự động restore/xóa/di chuyển bất cứ dữ liệu nào.

Phải bảo vệ consumed holdouts:

- `gZ-37`
- `gE-28`

Không đọc chúng phục vụ training, feature extraction, tuning hoặc model selection.

---

# V. PHASE 1 — VERIFY ALL EXISTING MODEL ARTIFACTS

## 1. Check model inventory

Kiểm tra:

- `models/`
- `models/base/`
- `models/candidate/`
- `models/locked/`
- `model-training/`
- `training/`
- `training/dmd/`
- `reports/ai85_upgrade/`
- `reports/ai85_next/`
- Các thư mục chạy huấn luyện YOLO trên máy hoặc VPS, nếu được mount.

Tìm các checkpoint thực sự tồn tại:

- `best.pt`
- `last.pt`
- YOLO detection checkpoints
- Seatbelt classifier checkpoints
- ResNet/GRU checkpoints
- Hierarchical action classifier checkpoints
- Research candidate checkpoints

Với mỗi model, ghi:

- Model name
- Architecture
- Training dataset
- Class mapping
- Input resolution
- Training epochs
- Checkpoint path
- File SHA-256
- Model version
- Validation results
- Approval status

Không giả định file `best.pt` từ notebook khác chắc chắn thuộc Roadwatch.

Không dùng checkpoint nếu class mapping không tương thích.

## 2. Reuse pretrained checkpoints

KHÔNG train lại từ đầu khi đã có checkpoint hợp lệ.

Ưu tiên:

1. Zero-retraining ablation.
2. Inference optimization.
3. Error analysis.
4. Reuse pretrained features.
5. Lightweight head fine-tuning.
6. Partial backbone fine-tuning.
7. Full fine-tuning nếu có bằng chứng cần thiết.

Phân biệt:

- Warm-start từ `best.pt`.
- Resume từ `last.pt` có đầy đủ optimizer/scheduler state.

Không gọi warm-start là exact resume.

---

# VI. PHASE 2 — ZERO-RETRAINING OPTIMIZATION

Ưu tiên tối đa hóa chất lượng model đã có trước khi chi phí GPU phát sinh.

## 1. YOLO Validation Ablations

Nếu checkpoint Ultralytics có sẵn và đúng nhiệm vụ, thực hiện:

A0 — BASELINE

- Original imgsz.
- Default validation confidence phù hợp với phiên bản thư viện.
- Class-aware NMS.
- TTA disabled.

A1 — TEST-TIME AUGMENTATION

- TTA enabled.
- Không thay đổi các yếu tố khác.

A2 — HIGHER INFERENCE RESOLUTION

- imgsz 640.
- imgsz 800/960.
- imgsz 1024 khi phần cứng cho phép.

A3 — CLASS-AGNOSTIC NMS

- Compare True/False.
- Kiểm tra các lớp chồng lấp.
- Không bật mặc định.

A4 — ROI CROPPING

- Original full frame.
- Driver ROI.
- Torso ROI.
- Phone/hand ROI nếu có localization hợp lệ.

A5 — SELECTED COMBINATION

Chỉ kết hợp các yếu tố đã có cải thiện có ý nghĩa trên validation.

## 2. Correct evaluation

Không sử dụng `conf=0.25` như cách mặc định để tính đường mAP.

Dùng confidence phù hợp cho AP calculation, sau đó đánh giá riêng inference operating points.

Không nhầm NMS IoU với IoU threshold dùng tính mAP50.

Ghi rõ:

- Ultralytics version.
- Validation arguments.
- Dataset split.
- Number of images.
- Class mapping.
- mAP50.
- mAP50-95.
- Per-class AP.
- Precision.
- Recall.
- FPS.
- Latency.

Không tuyên bố TTA tăng 2–4% hoặc NMS tăng 3–6% khi chưa có thí nghiệm.

---

# VII. PHASE 3 — PHONE DETECTOR RECOVERY

Theo báo cáo AI85 Next, nhiều sự kiện FN không có phone detection proposal phù hợp.

Phải kiểm tra bằng evidence từ frame/video gốc.

## 1. Root-cause analysis

Đối với từng FN:

- Phone visible?
- Phone fully/partially occluded?
- ROI clipping?
- Motion blur?
- Phone pixel size?
- Lighting?
- Camera angle?
- Detector confidence?
- Hand near phone?
- Face near phone?
- Wrong occupant?
- Temporal coverage?

Lưu bằng chứng cho từng case.

## 2. Two-stage ROI pipeline

Nghiên cứu:

Stage 1:
- Vehicle/driver localization khi cần.
- Driver ROI extraction.

Stage 2:
- Phone detection trên ROI đủ độ phân giải.
- Seatbelt classification trên torso ROI.
- Hand/face geometry verification.

Phải ánh xạ bounding boxes trở lại frame gốc chính xác.

Không để crop làm mất điện thoại gần mép ROI.

Thử padding ROI hoặc dynamic crop có kiểm chứng.

## 3. Small-object inference

Nghiên cứu:

- High-resolution crops.
- Small-object detection.
- Multi-scale inference.
- Detector confidence calibration.
- Class-aware NMS.
- Driver-phone association.

Không tự tăng confidence threshold để giảm FP nếu gây mất Recall.

Đo mức phục hồi FN thực tế.

---

# VIII. PHASE 4 — ADAPTIVE TEMPORAL SAMPLING

Đây là ưu tiên rất cao.

Pipeline cũ dùng action sampling khoảng 0.5 giây.

Một số ground-truth phone events ngắn hơn một giây.

Có trường hợp dài khoảng 0.40 giây không có centered action sample.

## Nhiệm vụ

Triển khai và so sánh:

T0:
Current fixed sampling baseline.

T1:
Dense fixed sampling với stride nhỏ hơn, khi tài nguyên cho phép.

T2:
Motion-triggered adaptive sampling.

T3:
Multi-rate temporal sampling.

T4:
Short-event-specific candidate recovery.

T5:
Adaptive sampling + temporal event confirmation.

## Điều kiện

- Không dùng ground-truth event boundaries để kích hoạt sampling lúc inference.
- Chỉ dùng visual/motion signals mà runtime thực sự có.
- Không cho mô hình nhìn trước dữ liệu vượt thiết lập latency được khai báo.
- Giữ event matcher và ground truth bất biến.
- Không tạo duplicate event để cải thiện recall.
- Không kéo dài event giả tạo.

Đánh giá:

- Number of detected short events.
- Recall by event duration.
- TP recovered.
- New FP.
- Event F1.
- Detection delay.
- Compute overhead.
- Full-pipeline FPS.

So sánh các nhóm duration:

- Dưới 1 giây.
- 1–3 giây.
- 3–10 giây.
- Trên 10 giây.

Chỉ dùng phân nhóm này để phân tích, không đưa ground-truth duration vào inference.

---

# IX. PHASE 5 — ACTION RECOGNITION UPGRADE

Mục tiêu khắc phục sự nhầm lẫn SAFE_DRIVING, PHONE_USE, OTHER_ACTIVITY.

## 1. Representation diagnostics

Kiểm tra:

- Embedding separability.
- Per-subject distribution.
- Camera appearance bias.
- Color/clothing dependence.
- Temporal context.
- Class-conditioned confidence.
- Prediction margins.
- Class priors.
- Train/inference consistency.

Phân tích vì sao SAFE Recall tăng ở direct coarse probe nhưng không cải thiện Event Recall.

## 2. Model research

Kế thừa những mô hình đang có:

- ResNet18 + GRU.
- Hierarchical classifier.
- Binary PHONE branch.
- Subject-normalized representation.

Không lặp lại thí nghiệm Mean Fine-tune, Attention Fine-tune và Native112 đã thất bại mà không có giả thuyết mới.

Thử khi phù hợp:

A. Behavior-centric ROI.

B. Hand/phone interaction features.

C. Face-hand-phone geometric features.

D. Lightweight motion features.

E. Multi-scale temporal features.

F. Subject-invariant representation.

G. Improved negative examples.

H. Action classifier với coarse/fine hierarchy đã hiệu chỉnh.

I. Feature-level hoặc late fusion giữa visual action model và object detector.

Nghiên cứu DualScope, DriveCLIP và Multi-Task Fusion như nguồn ý tưởng.

Không khẳng định tái hiện một paper nếu chỉ sử dụng một phần ý tưởng.

## 3. Training improvements

Khi có dữ liệu và tài nguyên:

- Balanced data construction.
- Hard-negative mining.
- Class-weighted loss ablation.
- Focal-loss ablation.
- Regularization.
- Appropriate augmentation.
- Early stopping on legitimate validation.
- Partial fine-tuning.

Đo clip Macro-F1 riêng với Event F1.

Không cố làm SAFE Recall tăng bằng cách dự đoán SAFE cho mọi frame.

---

# X. PHASE 6 — MULTI-VIEW AND GEOMETRY FUSION

Khi dữ liệu BODY/FACE/HANDS thực sự sẵn có, thử:

- BODY-primary baseline.
- FACE-assisted phone-call detection.
- HANDS-assisted texting detection.
- Pose-conditioned rescue.
- Phone-hand association.
- Phone-face geometry.
- Reliability-aware fusion.
- Temporal confirmation.

Không giả định mọi subject đều có đủ ba views.

Theo dữ liệu hiện có, gC-14 chỉ có BODY.

Không được đưa missing views thành zero-valued evidence rồi kết luận camera đó chứng minh không sử dụng điện thoại.

Có thể nghiên cứu MediaPipe để trích xuất hand/face landmarks nếu cài đặt và chất lượng ảnh cho phép.

Nhưng không tự gọi MediaPipe là model đã được tích hợp nếu mã nguồn chưa triển khai.

### Geometry verification

Nếu có phone box và hand/face landmarks:

- Normalized phone-ear distance.
- Phone-hand proximity.
- Hand-face interaction.
- Head pose.
- Temporal movement pattern.

Các geometry thresholds phải được chọn bằng validation hợp lệ.

Không dùng geometry như điều kiện cứng gây bỏ sót điện thoại bị che khuất.

---

# XI. PHASE 7 — SEATBELT AI IMPROVEMENT

Đề cương yêu cầu cả hai hành vi vi phạm.

Do đó không được chỉ tối ưu DMD phone-use.

## 1. Audit

Đọc seatbelt evaluator và confusion matrix hiện có.

Phân tích ba lớp:

- FASTENED.
- UNFASTENED.
- UNCERTAIN_OR_OCCLUDED.

Kiểm tra:

- Torso ROI.
- Camera viewpoint.
- Occlusion.
- Dark clothing.
- Similar belt/clothing colors.
- False belt detection.
- Missing belt evidence.
- Class imbalance.
- Label provenance.

## 2. Zero-retraining experiments

Thử khi thích hợp:

- Improved torso ROI.
- Different inference resolution.
- Confidence calibration.
- TTA.
- Safe decision logic.
- Temporal smoothing trên video.
- Uncertainty-aware abstention.

## 3. Fine-tuning

Chỉ fine-tune nếu:

- Có checkpoint hợp lệ.
- Có ground-truth labels được xác nhận.
- Có train/validation split độc lập.
- Có đủ tài nguyên.

Nghiên cứu attention-based classifier và ROI-based seatbelt models.

Không tự coi belt-not-visible là UNFASTENED.

Đánh giá:

- Per-class Precision.
- Recall.
- F1.
- Macro-F1.
- Balanced Accuracy.
- False-unfastened rate.
- Abstention rate.

---

# XII. PHASE 8 — SELECTIVE FINE-TUNING

Nếu Zero-Retraining không đủ, chuyển sang fine-tune.

## 1. Reuse best.pt

Nếu `best.pt` thuộc đúng YOLO detector:

- Load pretrained checkpoint.
- Verify architecture and classes.
- Check training/validation split.
- Perform controlled warm-start.

## 2. Fine-tuning ablations

Thử:

F0:
Original checkpoint.

F1:
Short fine-tune at original resolution.

F2:
Fine-tune with higher resolution.

F3:
Fine-tune using verified hard examples.

F4:
Fine-tune with modified augmentation.

F5:
Selected combination based on proper validation.

Không mặc định `imgsz=960`, learning rate hoặc mosaic settings chắc chắn tốt hơn.

Các hyperparameters là experimental variables.

Không tự đặt mức cải thiện dự kiến như một kết quả bảo đảm.

## 3. Resource governance

Kiểm tra:

- GPU availability.
- VRAM.
- CUDA.
- Training dataset size.
- Disk space.
- Checkpoint compatibility.

Nếu chỉ có CPU, thực hiện các phần phù hợp và chuẩn bị cấu hình GPU.

Nếu có RTX 5090 hợp lệ, đo tốc độ thực tế chứ không giả định số phút cho mỗi epoch.

Không huấn luyện từ đầu khi fine-tuning hợp lệ và đủ cho giả thuyết đang kiểm tra.

---

# XIII. PHASE 9 — DATA QUALITY AND GENERALIZATION

Đây là yêu cầu cực kỳ quan trọng để chứng minh các chỉ số trên 85%.

## 1. Dataset audit

Kiểm tra:

- Number of subjects.
- Number of videos.
- Number of frames.
- Number of annotations.
- Class distribution.
- Subject overlap.
- Camera overlap.
- Near-duplicates.
- Timestamp accuracy.
- License.
- Human-review status.
- Missing labels.

## 2. Development split

Giữ nguyên và không tái sử dụng consumed holdouts.

Không dùng:

- gZ-37.
- gE-28.

để tuning hoặc model selection.

## 3. New development subjects

Nếu có dữ liệu mới hợp lệ:

- Chia theo subject.
- Khóa split trước training.
- Tách validation.
- Chọn model và threshold chỉ bằng validation.
- Freeze model.
- Đánh giá trên untouched independent holdout.

Nếu chưa có subject mới:

- Không công bố independent 85%.
- Không tiếp tục threshold search vô hạn trên ba subjects cũ.
- Hoàn thiện các diagnostics, tests và training preparation có thể xác minh.

## 4. Scientific validity

Phải phân biệt:

- Training metrics.
- Development metrics.
- Cross-validation metrics.
- Selected validation metrics.
- Independent holdout metrics.

Không được gộp thành một bảng kết quả duy nhất mà thiếu nhãn nguồn.

---

# XIV. PHASE 10 — OPTIMIZE EVENT F1

Sau mỗi thay đổi, phải báo cáo:

| Candidate | TP | FP | FN | Precision | Recall | Event F1 | FA/min | SAFE Recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|

### Important rule

Nếu candidate phục hồi 5 FN nhưng mất 5 TP, đó KHÔNG phải cải thiện Event Recall.

Phải so sánh:

- Recovered events.
- Lost events.
- New false positives.
- Removed false positives.
- Duplicate events.
- Short-event coverage.
- Per-subject metrics.

Không chỉ báo cáo số sự kiện được cứu mà bỏ qua số sự kiện bị mất.

### Model selection

Ưu tiên candidate có:

1. Evaluation hợp lệ.
2. Precision tốt.
3. Recall tốt.
4. Event F1 cao.
5. False alarms thấp.
6. Per-subject stability.
7. SAFE_DRIVING reliability.
8. Runtime phù hợp.

Không chấp nhận nghiên cứu chỉ vì nó đạt một metric cao trong khi các metric khác sụp đổ.

---

# XV. PHASE 11 — FULL-PIPELINE PERFORMANCE

Đo riêng:

- Object detector FPS.
- Action encoder latency.
- Temporal model latency.
- Fusion latency.
- Seatbelt classifier latency.
- Video decode time.
- Evidence processing time.
- Full-pipeline FPS.
- p50/p95 latency.
- RAM/VRAM.

Không lấy tốc độ cached features làm full-system FPS.

Đo chênh lệch giữa:

- Original resolution.
- High-resolution inference.
- ROI cropping.
- TTA.
- Adaptive sampling.

Chỉ giữ optimization nếu cải thiện chất lượng xứng đáng với chi phí tính toán.

---

# XVI. PHASE 12 — SOFTWARE INTEGRATION

Sau khi có candidate được đánh giá hợp lệ, kiểm tra khả năng tích hợp với Roadwatch.

Backend:

- FastAPI inference.
- Model loading.
- Video processing.
- Event storage.
- Job queue.
- Authentication.
- Review workflow.
- Evidence metadata.
- Error logging.

Frontend:

- Video upload.
- Prediction overlay.
- Event timeline.
- Confidence/uncertainty display.
- Alerts.
- Human review.
- Statistics.
- Export.

Không tự thay model production bằng experimental checkpoint.

Phải có model registry và approval status rõ ràng.

Không tự động gán trạng thái HUMAN_CONFIRMED cho AI prediction.

---

# XVII. PHASE 13 — TESTING

Chạy:

- Python compile.
- Ruff critical gate.
- Targeted tests.
- DMD regression tests.
- Seatbelt tests.
- Metric reconciliation tests.
- Data-split protection tests.
- Full Pytest.
- Frontend build.
- Backend smoke tests.
- Inference smoke tests nếu có checkpoint.
- End-to-end tests nếu môi trường hỗ trợ.

Các lỗi liên quan cây `datasets/` phải được phân biệt với source-code regressions.

Không xóa test để được kết quả PASS.

Phải báo cáo số lượng:

- PASSED.
- FAILED.
- ERROR.
- SKIPPED.
- BLOCKED.

---

# XVIII. PHASE 14 — REQUIRED DELIVERABLES

Tạo thư mục:

`reports/ai85_plus/`

Các báo cáo:

1. `01_PROJECT_AUDIT.md`
2. `02_MODEL_WEIGHT_INVENTORY.md`
3. `03_DATASET_VALIDATION.md`
4. `04_ZERO_RETRAINING_ABLATIONS.md`
5. `05_PHONE_FALSE_NEGATIVE_ANALYSIS.md`
6. `06_ADAPTIVE_TEMPORAL_SAMPLING.md`
7. `07_OBJECT_DETECTION_EXPERIMENTS.md`
8. `08_ACTION_CLASSIFICATION_EXPERIMENTS.md`
9. `09_MULTI_VIEW_FUSION.md`
10. `10_FINE_TUNING_RESULTS.md`
11. `11_SEATBELT_RESULTS.md`
12. `12_METRICS_COMPARISON.md`
13. `13_GENERALIZATION_VALIDATION.md`
14. `14_RUNTIME_BENCHMARK.md`
15. `15_TEST_RESULTS.md`
16. `16_FINAL_RESEARCH_REPORT.md`

Kèm:

- Machine-readable experiment JSON.
- Model checkpoint metadata.
- Config files.
- Prediction artifacts.
- Confusion matrices.
- Per-subject metrics.
- Event matching evidence.
- Input hashes.
- Model hashes.
- Git commit.
- Reproduction commands.

Không tạo file báo cáo rỗng.

Không ghi kết quả chưa được đo.

---

# XIX. IMPLEMENTATION PRIORITIES

Thực hiện theo thứ tự:

### PRIORITY 0 — SAFETY

Kiểm tra Git, checkpoint, dataset, training environment và backup.

### PRIORITY 1 — ZERO-RETRAINING

Thử nghiệm checkpoint hiện có, ROI, TTA, resolution và NMS có kiểm soát.

### PRIORITY 2 — DMD EVENT RECALL

Điều tra 13 FN, cải thiện adaptive sampling và action evidence.

### PRIORITY 3 — VISUAL REPRESENTATION

Cải thiện Phone ROI, Hand/Face Association và action classifier.

### PRIORITY 4 — FUSION & TEMPORAL

Giảm FN nhưng không làm tăng FP quá mức.

### PRIORITY 5 — SELECTIVE FINE-TUNING

Chỉ huấn luyện thêm khi có đủ dữ liệu và bằng chứng cần thiết.

### PRIORITY 6 — SEATBELT

Nâng cấp mô hình dây an toàn và uncertainty handling.

### PRIORITY 7 — SCIENTIFIC VALIDATION

Kiểm tra độc lập với subject mới được cấp phép.

### PRIORITY 8 — SYSTEM COMPLETION

Đảm bảo backend/frontend hoạt động và có thể trình diễn đúng phạm vi đề cương.

---

# XX. FINAL TARGET VERIFICATION

Mục tiêu nghiên cứu:

- Event Precision >= 85%.
- Event Recall >= 85%.
- Event F1 >= 85%.
- SAFE_DRIVING Recall >= 85%.
- FA/min <= 0.15.
- Object Detection mAP50 >= 85%.
- Seatbelt Macro-F1 >= 85%.

Mỗi chỉ số phải được tính trên bài toán, bộ dữ liệu và giao thức tương ứng.

Không coi mAP50 của detector là Event F1 của hệ thống.

Không coi Top-1 Accuracy là Macro-F1.

Không coi kết quả đã đạt trên development là independent holdout performance.

Không sửa ground truth, threshold bằng nhãn test hoặc evaluation rules để ép đạt 85%.

Nếu chỉ số không tăng, phải xác định nguyên nhân và đề xuất/triển khai bước cải thiện tiếp theo có bằng chứng.

Nếu thiếu independent holdout, đánh dấu:

`INDEPENDENT_AI85_NOT_ESTABLISHED`

Không tự động phê duyệt production.

---

# XXI. EXECUTE NOW

Bắt đầu trực tiếp trong workspace.

1. Kiểm tra repository hiện tại.
2. Xác minh checkpoint và dataset.
3. Kiểm tra baseline metrics.
4. Thực hiện Zero-Retraining Ablation.
5. Phân tích FN/FP theo bằng chứng.
6. Cải thiện adaptive temporal sampling.
7. Cải thiện Phone ROI/Hand/Face evidence.
8. Cải thiện action classification và fusion.
9. Fine-tune hợp lệ khi cần thiết.
10. Đánh giá seatbelt.
11. Kiểm tra generalization và data validity.
12. Chạy tests.
13. Đo lại tất cả metrics.
14. Tổng hợp ablation và báo cáo.
15. Xác định candidate tốt nhất đủ điều kiện.

Không dừng ở kế hoạch nếu có thể sửa code và chạy thí nghiệm.

Không lặp lại các experiment đã thất bại mà không có giả thuyết mới.

Không cam kết thời gian huấn luyện khi chưa có profiler.

Không tuyên bố đạt 85% khi chưa có kết quả độc lập đáng tin cậy.

**MỤC TIÊU CUỐI CÙNG: XÂY DỰNG ROADWATCH AI CÓ ĐỘ CHÍNH XÁC CAO, PHÁT HIỆN ĐƯỢC NHIỀU HÀNH VI THỰC TẾ, ÍT CẢNH BÁO SAI, TÁI LẬP ĐƯỢC THỰC NGHIỆM VÀ CÓ ĐỦ CƠ SỞ KHOA HỌC ĐỂ BẢO VỆ ĐỒ ÁN TỐT NGHIỆP.**

BẮT ĐẦU NGAY BẰNG MODEL/DATA AUDIT VÀ ZERO-RETRAINING EXPERIMENTS, SAU ĐÓ TẬP TRUNG PHỤC HỒI 13 FALSE NEGATIVES CỦA DMD.