# ROADWATCH V16 — DATA-CENTRIC AI85+ RESEARCH & ENGINEERING
## PRINCIPAL AI SCIENTIST / COMPUTER VISION TECH LEAD / DATASET ARCHITECT

Bạn là Principal AI Scientist, Senior Computer Vision Engineer, Video Action Recognition Researcher, Data-Centric AI Specialist và MLOps Lead.

Nhiệm vụ của bạn:

**RÀ SOÁT V15 → BỔ SUNG DATASET HỢP LỆ → KIỂM TRA NHÃN/SPLIT → CẢI THIỆN MÔ HÌNH → ĐO LẠI PHONE EVENT F1, SAFE RECALL VÀ SEATBELT MACRO-F1.**

Không chỉ tạo thêm báo cáo.

Không tải dữ liệu thiếu quyền.

Không train trên holdout.

Không làm giả hoặc bảo đảm chỉ số trên 85%.

---

# 1. REPOSITORY VÀ TRẠNG THÁI LÀM VIỆC

Repository:
https://github.com/ln3076059-jpg/prok53

Branch:
roadwatch-v4-candidate

Workspace:
D:\.idea\giangdoantotnghiep

Remote HEAD được xác minh:
f881dbb056d615f8c615da5d016b27f9de4054d9

Theo báo cáo người dùng, V15 đã được commit tại local:
25d90053a0a41fb483aaa43038d8e6c23ebbb12a

Commit V15 có thể chưa được push.

LƯU Ý QUAN TRỌNG:

- Kiểm tra git status, branch, HEAD và remote tracking trước khi làm.
- Đọc AGENTS.md nếu có.
- Giữ nguyên commit local V15.
- Không tự checkout bản remote cũ rồi ghi đè V15.
- Không tự reset, clean hoặc restore prompt.md.
- Không commit prompt.md nếu đó là thay đổi chưa commit của người dùng.
- Không tự push hoặc merge main.
- Không xóa datasets, checkpoints, logs hoặc reports cũ.

Trước tiên xác minh V15 đã có:
- Tiled/overlapping inference.
- SAHI-inspired adapter.
- ROI coordinate mapping.
- Proposal NMS/merging.
- Tests.
- Reports/v15_research/.
- Machine-readable artifacts.

Không tạo lại chức năng đã có.

# 2. BASELINE ĐÁNH GIÁ

Các giá trị người dùng báo cáo từ V15:

Phone Event:
- TP=16.
- FP=4.
- FN=13.
- Precision=80.00%.
- Recall=55.17%.
- F1=65.31%.
- FA/min=0.1813.

Action:
- E3 Clip Macro-F1=56.39%.

Seatbelt:
- Macro-F1=58.52%.

Engineering:
- Full Pytest=629 PASS.
- Targeted Pytest=40 PASS.
- Compile/Ruff/frontend/backend smoke PASS.
- V15 bundle validation=10/10 PASS.

Đây là kết quả được người dùng cung cấp. Phải xác minh từ local artifacts trước khi coi là baseline được tái lập.

Mục tiêu nghiên cứu:
- Phone Event Precision >85%.
- Phone Event Recall >85%.
- Phone Event F1 >85%.
- FA/min <=0.15.
- SAFE_DRIVING Recall >85%.
- Seatbelt Macro-F1 >85%.
- Seatbelt UNCERTAIN Recall >85%.

Mục tiêu không phải kết quả được bảo đảm.

# 3. PHASE A — NGHIÊN CỨU VÀ KIỂM TRA DATASET

Trước khi download hoặc training, xác minh các dataset sau.

## DATASET A1 — DMD

Official:
https://github.com/Vicomtech/DMD-Driver-Monitoring-Dataset

Website:
https://dmd.vicomtech.org/

Mục đích:
- Video phone-use events.
- BODY/FACE/HANDS views.
- Temporal action labels.
- Event-level evaluation.

Kiểm tra cập nhật dataset năm 2026:
- RGB real-car availability.
- Eligible subjects.
- OpenLABEL/VCD version.
- CC BY-NC-ND conditions.
- Rights to process and fit.
- Existing project permission record.

Protected Roadwatch subjects:
- gZ-37.
- gE-28.

Không sử dụng chúng cho fitting, calibration, tuning hoặc development.

Development subjects đã dùng lặp:
- gB-9.
- gC-14.
- gZ-36.

Không coi đây là independent evaluation mới.

KHÔNG tự thay đổi RIGHTS_ACCEPTED hay các permission flags.

## DATASET A2 — 100-DRIVER

Official:
https://100-driver.github.io/

Code:
https://github.com/Shenqishaonv/100-Driver-Source

Mục đích:
- Driver action classification.
- Cross-driver generalization.
- Cross-view robustness.
- Camera/lighting variation.

Bộ dữ liệu có hơn 470.000 ảnh từ 100 người lái theo công bố của tác giả.

Yêu cầu:
- Đọc giấy phép chính thức.
- Kiểm tra thủ tục xin quyền truy cập.
- Không bypass registration.
- Không tự mirror dữ liệu.
- Không gọi ảnh classification là event annotations.
- Không tạo giả video timestamps.

Nếu chưa được cấp quyền, trạng thái:
ACCESS_PENDING_LICENSE_APPROVAL.

## DATASET A3 — STATE FARM DISTRACTED DRIVER

Official:
https://www.kaggle.com/competitions/state-farm-distracted-driver-detection/data

Labels:
- c0: SAFE_DRIVING.
- c1: TEXTING_RIGHT.
- c2: PHONE_CALL_RIGHT.
- c3: TEXTING_LEFT.
- c4: PHONE_CALL_LEFT.
- c5-c9: OTHER_ACTIVITY.

Đề xuất mapping:
- c0 → SAFE_DRIVING.
- c1-c4 → PHONE_USE.
- c5-c9 → OTHER_ACTIVITY.

Tuy nhiên phải giữ original label và provenance, không phá dữ liệu gốc.

Mục đích:
- Pretraining/fine-tuning action classifier.
- Hard-negative learning.
- Evaluation giữa các người lái.

Bắt buộc:
- Kiểm tra competition rules.
- Chỉ download sau khi tài khoản được phép.
- Dùng driver_imgs_list.csv để kiểm tra subject IDs.
- Không random-split ảnh của cùng người lái qua train/test.
- Không biến image labels thành video event labels.

## DATASET A4 — DRIVE&ACT

Official:
https://www.driveandact.com/

Mục đích:
- Multi-view action recognition.
- Hand-object interactions.
- Temporal feature learning.
- Object interaction modeling.

Kiểm tra:
- Official research-use conditions.
- Labels thực sự có liên quan đến phone use.
- Camera viewpoints.
- Video format.
- Subjects.
- Annotation temporal granularity.

Không tự map những activity không tương đương sang PHONE_USE.

Không coi mô hình tốt trên autonomous-car interaction là tốt trên camera roadside.

## DATASET A5 — COCO 2017

Official:
https://cocodataset.org/

Mục đích:
- Phone object detection.
- Cell phone bounding boxes.
- People and contextual hard negatives.
- Detector pretraining.

Chỉ lấy subset cần thiết nếu quyền, công cụ và tài nguyên cho phép.

Kiểm tra:
- Exact class mapping.
- Image-level copyright.
- Annotation completeness.
- Image source licenses.
- Positive/negative selection.
- Overlap với những bộ dữ liệu khác.

Không download toàn bộ COCO khi một subset nhỏ là đủ cho pilot.

## DATASET A6 — OPEN IMAGES V7

Official:
https://storage.googleapis.com/openimages/web/index.html

Download:
https://storage.googleapis.com/openimages/web/download_v7.html

Mục đích:
- Phone-like objects.
- Person/handheld visual context.
- Hard negatives.
- Cross-domain robustness.

Kiểm tra chính xác class ID và class name từ metadata chính thức.

Không tự giả định mọi ảnh không có bbox phone là negative hoàn chỉnh; Open Images có quy tắc đánh giá cho incomplete annotations.

Kiểm tra license từng ảnh trước sử dụng.

## DATASET A7 — SEATBELT & MOBILE

Official candidate:
https://universe.roboflow.com/naina-iucjd/seatbelt_and_mobile_detection-1qojv

Mục đích:
- Phone plus seatbelt detection.
- Shared upper-body representation.
- Roadside/cabin viewpoint comparison.

Bắt buộc:
- Xác minh original images, không chỉ augmented export.
- Kiểm tra các class names thực tế.
- Xác minh nhãn seatbelt và mobile có đúng nghĩa không.
- Kiểm tra duplicate sources.
- Kiểm tra project license và ảnh nguồn.
- Xác định dataset có UNKNOWN/UNCERTAIN hay không.

Không lấy mAP trên Roboflow model page làm kết quả của Roadwatch.

## DATASET A8 — SEATBELT DETECTION

Official candidate:
https://universe.roboflow.com/seatbelttraining-7yh0f/seatbelt-detection-lb1ec

Mục đích:
- FASTENED vs UNFASTENED.
- Detector localization.
- Hard negatives.

Kiểm tra:
- License.
- Original data provenance.
- Augmentation.
- Label quality.
- Group leakage.
- Camera angle.
- Occlusions.
- Class distribution.

Không tự tạo nhãn UNCERTAIN từ ảnh mà không được human review.

## DATASET A9 — SEATBELT THREE-STATE CANDIDATE

Candidate:
https://universe.roboflow.com/eco3cc/seat-belt-detection-uhqwa-0tqd0

Mục đích:
- Khảo sát nhãn dây an toàn.
- Đánh giá khả năng biểu diễn NOT CLEAR/UNKNOWN.

Phải kiểm tra class mapping trong dataset export thực tế; mô tả của tác giả không đủ bảo đảm nhãn UNCERTAIN đã có và nhất quán.

Không coi toàn bộ dữ liệu cộng đồng là ground truth chuẩn.

---

# 4. PHASE B — DATASET SELECTION DECISION

Không tải tất cả dataset cùng lúc.

Xây dựng dataset registry với trường:

- Dataset ID.
- Official source URL.
- Version.
- Source owner.
- License.
- Access status.
- Allowed operations.
- Download size.
- Number of original samples.
- Number of augmented samples.
- Number of drivers/subjects nếu biết.
- Number of cameras.
- Annotation format.
- Annotation quality.
- Tasks supported.
- Target labels.
- Domain.
- Known limitations.
- Provenance hash.
- Duplicate status.
- Training eligibility.
- Validation eligibility.
- Independent evaluation eligibility.

Tạo status:

DISCOVERED
LICENSE_UNVERIFIED
ACCESS_REQUEST_REQUIRED
DOWNLOAD_AUTHORIZED
DOWNLOADED
QUARANTINED_FOR_REVIEW
ANNOTATION_PENDING
APPROVED_FOR_TRAINING
VALIDATION_ONLY
LOCKED_TEST
BLOCKED

Không tự chuyển dataset sang APPROVED vì link công khai.

Không tự đăng ký tài khoản hoặc vượt qua điều kiện tải dữ liệu.

Khi dataset cần người dùng chấp thuận license, tạo hướng dẫn và thủ tục rõ ràng.

Các dataset khác DMD được xét quyền độc lập; không sử dụng quyền của một dataset để thay đổi cờ cấp phép DMD.

---

# 5. PHASE C — DATA ACQUISITION PIPELINE

Tạo hoặc nâng cấp:

training/data_registry/
training/data_ingestion/
training/data_validation/

Sử dụng thư mục thực tế phù hợp repo.

Không tạo kiến trúc mới nếu repo đã có chức năng tương đương.

## Requirements

- Download only authorized datasets.
- Support resumable downloads.
- Verify expected file hashes.
- Verify available disk space.
- Maintain original files read-only.
- Separate raw/processed/annotations.
- Prevent overwriting existing files.
- Deduplicate by SHA/perceptual hash.
- Validate image/video decoding.
- Validate width/height/fps.
- Validate bbox coordinates.
- Validate class mapping.
- Validate timestamps.
- Validate subject/session IDs.
- Prevent source leakage.
- Reject invalid data before training.

Không commit raw videos và hàng GB images lên Git.

Chỉ commit manifests/configs/scripts được phép công bố.

Nếu media riêng tư, không đưa ảnh nhận diện người vào báo cáo công khai.

---

# 6. PHASE D — BUILD ROADWATCH ANNOTATION SCHEMA

Định nghĩa unified labels:

## Phone object

PHONE_BOX
PHONE_VISIBLE
PHONE_OCCLUDED
PHONE_ABSENT
UNKNOWN_PHONE_VISIBILITY

## Phone action

SAFE_DRIVING
TEXTING
HANDHELD_CALL
OTHER_PHONE_USE
OTHER_ACTIVITY
UNKNOWN_ACTION

## Phone context

HANDHELD
MOUNTED
STATIC_CAMERA
PASSENGER_PHONE
UNKNOWN_CONTEXT

## Occupant roles

DRIVER
FRONT_PASSENGER
REAR_PASSENGER
UNKNOWN

## Seatbelt

FASTENED
UNFASTENED
UNCERTAIN_OR_OCCLUDED

## Event schema

- Video ID.
- Session ID.
- Subject ID.
- Camera ID.
- Start timestamp.
- End timestamp.
- Frame rate.
- Event class.
- Bbox references.
- Occupant track/role.
- Annotation confidence.
- Reviewer ID.
- Review status.
- Consent/license reference.
- Source hash.

Giữ original labels và mapping version.

Không tự gán UNKNOWN thành SAFE.

Không tự gán seatbelt invisible thành UNFASTENED.

Không coi phone visible đồng nghĩa phone use.

Không suy diễn time interval cho dữ liệu ảnh tĩnh.

---

# 7. PHASE E — HUMAN ANNOTATION AND REVIEW

Ưu tiên chuẩn bị dữ liệu thật mà Roadwatch đang thiếu.

## Phone detection benchmark

Cần:
- Frame có điện thoại.
- Frame không có điện thoại.
- Điện thoại nhỏ.
- Điện thoại gần mặt.
- Điện thoại trong tay.
- Điện thoại gắn giá đỡ.
- Vật giống điện thoại.
- Hành khách cầm điện thoại.
- Phone partially occluded.
- Different lighting/camera angles.

Mọi positive object cần bbox được người kiểm tra.

Negative frame phải được review là không có phone thuộc phạm vi annotation.

## Event benchmark

Cần video có:
- True phone-use events.
- SAFE intervals.
- OTHER distraction.
- Short phone interactions dưới một giây.
- Long phone interactions.
- Driver/passenger differences.
- Real timestamps.

## Seatbelt review

Kiểm tra 500 pending candidates hiện có khi quyền truy cập cho phép.

Không tự approve.

Phải có reviewer và nhãn được xác nhận.

## Annotation QA

- Written labeling guide.
- Inter-annotator agreement trên subset.
- Disagreement adjudication.
- Audit timestamps.
- Audit bounding boxes.
- Deduplicate source images.
- Quality flags.
- Source/version hashes.

Cho phép thu thập thêm dữ liệu tự quay với sự đồng ý, trong xe đang đỗ, không gây hành vi nguy hiểm khi lưu thông.

---

# 8. PHASE F — DATASET SIZE & SPLIT PLANNING

Không lấy số lượng ảnh làm chỉ số đủ dữ liệu.

Phải tính:
- Số subjects.
- Số sessions.
- Số videos.
- Số positive phone events.
- Tổng negative exposure minutes.
- Số phone bbox positive/negative.
- Số seatbelt samples mỗi class.
- Số camera domains.
- Class balance.
- Duration distribution.

Đề xuất planning target ban đầu, điều chỉnh theo nguồn lực:

- Ít nhất 20 người tham gia mới nếu có thể thu thập hợp lệ.
- Có nhóm train, validation và locked test tách biệt theo subject.
- Đặt mục tiêu thu đủ tối thiểu khoảng 100 phone-use events trong independent test để phép đo Recall có ý nghĩa hơn benchmark 29 events.
- Có đủ negative driving minutes để ước lượng False Alarms/min.
- Có mẫu seatbelt rõ và khó quan sát trong mỗi tập.

Đây là PLANNING TARGET, không phải giả định dữ liệu hiện có.

Tính confidence intervals từ sample size thực tế.

Không tự đảm bảo mức 100 events là đủ cho mọi độ tin cậy thống kê.

Không dùng frame từ cùng video/session xuất hiện ở cả train và test.

Không dùng gZ-37/gE-28 để tuning.

Không coi repeatedly used gB-9/gC-14/gZ-36 là independent test.

---

# 9. PHASE G — RE-EVALUATE V15 SAHI/TILED DETECTOR

V15 đã triển khai tiled phone inference.

Không viết lại SAHI adapter.

Trước tiên audit:
- Tile dimensions.
- Overlap.
- Scale transforms.
- NMS.
- Duplicate boxes.
- Phone confidence.
- ROI boundaries.
- Driver-region quality.
- Checkpoint hash.
- Inference cost.

Thử ablations trên dữ liệu được phép:

D0: Existing specialist detector.
D1: COCO auxiliary phone detector.
D2: Original-resolution driver ROI.
D3: V15 overlapping tiled inference.
D4: Controlled dual-detector fusion.

Giữ model checkpoint và dataset protocol cố định giữa các inference ablations.

Metrics:
- Phone AP50.
- AP50-95.
- Precision.
- Recall.
- Negative frame FP rate.
- Small-phone recall by box area.
- Candidate yield.
- p50/p95 runtime.
- RAM/VRAM.

Không dùng 13 FN-selected frames như detector validation chính thức.

Không gọi proposal yield là recall.

Nếu dùng SAHI làm tăng FP đáng kể thì phải reject hoặc cải thiện candidate filtering trên validation, không tự promote.

---

# 10. PHASE H — ACTION MODEL IMPROVEMENT

Nghiên cứu khoa học:

DualScope:
https://ojs.aaai.org/index.php/AAAI/article/view/37812

DriveCLIP:
https://github.com/zahid-isu/DriveCLIP

Multi-Task Fusion:
https://doi.org/10.3390/s25216799

MixStyle:
https://arxiv.org/abs/2104.02008

Group DRO:
https://arxiv.org/abs/1911.08731

Nếu fitting và dữ liệu đã hợp lệ:

A0: Frozen E3 baseline.
A1: High-resolution behavior ROI.
A2: Hand/phone/face features.
A3: Causal motion features.
A4: Frozen CLIP representation.
A5: Lightweight spatial+temporal fusion.
A6: Group-aware improvement.

Ưu tiên frozen encoder + small head trước full retraining.

Bắt buộc có:
- Real fitted weights.
- Real scaler.
- Class mapping.
- Training data provenance.
- Validation group IDs.
- Saved checkpoint hash.
- Inference replay.
- Seed.
- Training logs.

Không tạo synthetic fitted model để báo accuracy.

Đánh giá:
- SAFE Recall.
- PHONE Recall.
- OTHER Recall.
- Clip Macro-F1.
- Balanced Accuracy.
- Worst-subject performance.
- Class confusion matrix.

Không chuyển E3 training labels sang event labels nếu không có temporal GT.

---

# 11. PHASE I — FIX ASSOCIATION & FUSION

Current blockers:
- Driver identity chưa được xác minh.
- Phone evidence chưa đủ.
- Action scores có thể dưới gate.
- Event engine có thể làm mất short events.

Kiểm tra:
training/dmd/driver_phone_association.py
training/dmd/autonomous_phone.py
training/dmd/online_candidate.py
backend/ai/fusion.py
backend/ai/events.py

Không tạo association/fusion modules mới trùng chức năng.

Đo candidate flow:

Detected
→ Candidate
→ Driver-associated
→ Handheld-evidence
→ Action-supported
→ Fusion-admitted
→ Temporal-activated
→ Event-matched

Ở mỗi bước lưu:
- Number admitted.
- Number rejected.
- Reason code.
- Ground-truth positives rejected.
- Negative candidates admitted.
- Coverage.
- Abstention.

Không loại bỏ toàn bộ sự kiện không có verified driver track mà không ghi lại nguyên nhân.

Dùng trạng thái NEEDS_REVIEW khi có evidence đáng quan tâm nhưng chưa thể xác nhận vi phạm.

Không tự gán NEEDS_REVIEW thành TP.

Phải tính metrics riêng cho confirmed alerts và coverage/review queue.

---

# 12. PHASE J — SHORT EVENT RECOVERY

Nghiên cứu:
ActionFormer:
https://arxiv.org/abs/2202.07925

Chỉ áp dụng các ý tưởng phù hợp:

- Causal temporal convolution.
- Multi-scale temporal evidence.
- Start/end localization.
- Event candidate ranking.
- Short-duration behavior representation.
- Duplicate event suppression.

Không dùng future frames trong chế độ live inference.

Không sửa event matcher để tăng F1.

Không kéo dài prediction theo ground truth.

Đánh giá duration bins:
- <1 second.
- 1–3 seconds.
- 3–10 seconds.
- >10 seconds.

Báo:
- Recovered TP.
- Lost TP.
- New FP.
- Removed FP.
- Event Precision/Recall/F1.
- FA/min.
- Event onset error.
- Alert emission latency.

---

# 13. PHASE K — SEATBELT THREE-STATE RESEARCH

Bắt đầu từ:
FASTENED.
UNFASTENED.
UNCERTAIN_OR_OCCLUDED.

Nghiên cứu:
https://doi.org/10.3390/app14083346
https://arxiv.org/abs/2203.00810
https://arxiv.org/abs/2204.07946

Các giải pháp:
- Torso ROI.
- Belt-line localization.
- Segmentation-assisted evidence.
- Lightweight attention.
- Visibility-aware classification.
- Human-reviewed hard negatives.
- Selective prediction.

Nếu dataset được phê duyệt:

S0: Existing model baseline.
S1: Better torso ROI.
S2: Seatbelt visibility auxiliary head.
S3: Lightweight attention.
S4: Hard-negative and uncertain-aware training.

Không biến thiếu belt visibility thành UNFASTENED.

Metrics:
- Macro-F1.
- Per-class Recall.
- Balanced Accuracy.
- False-unfastened.
- Abstention rate.
- Coverage.

Không bỏ qua UNCERTAIN samples khỏi mẫu số mà không báo cáo.

---

# 14. PHASE L — TRAINING AND GPU POLICY

Chỉ training khi:
- Dataset license verified.
- Operation permitted.
- Labels approved.
- Group split verified.
- Model/checkpoint provenance known.
- Resources sufficient.

Reuse checkpoints when suitable.

Warm-start khác exact resume.

Training plan:
1. Small pilot.
2. Frozen linear probe.
3. Lightweight head.
4. Partial fine-tune.
5. Full fine-tune nếu thật sự cần.

Không tự hứa đạt 85% sau 50/80 epochs.

Không yêu cầu tải tất cả datasets để chạy pilot đầu tiên.

Nếu tài nguyên chỉ có CPU:
- Audit dữ liệu.
- Chạy validation.
- Chạy inference subset.
- Synthetic unit tests.
- Chuẩn bị GPU configs.

Nếu có GPU:
- Record GPU type.
- CUDA and package versions.
- VRAM.
- Peak memory.
- Wall-clock runtime.
- Throughput.
- Training curves.

Không upload private/controlled media lên cloud khi chưa được phê duyệt.

---

# 15. PHASE M — END-TO-END AI EVALUATION

Khi candidate hợp lệ tồn tại:

VIDEO
→ PHONE DETECTOR
→ HIGH-RES ROI
→ ACTION MODEL
→ DRIVER ASSOCIATION
→ FUSION
→ TEMPORAL ENGINE
→ EVENT RECORD
→ FROZEN EVENT MATCHER

Bắt buộc lưu:
- Actual raw predictions.
- Checkpoint hashes.
- Dataset hash.
- Model/config version.
- Event start/end.
- Actual emission timestamps.
- TP/FP/FN match records.
- Per-subject statistics.
- Negative-video exposure.
- Review/abstention states.

Metrics:
- Phone Event P/R/F1.
- FA/min.
- SAFE Recall.
- Short-event Recall.
- Phone AP.
- Seatbelt Macro-F1.
- Runtime FPS.
- Alert latency.

Không gộp classification accuracy với event metric.

Không chọn thresholds bằng locked test.

Không tuyên bố independent AI85 nếu chưa có independent holdout.

---

# 16. PHASE N — TESTING

Regression tests bắt buộc:

- Dataset license status.
- No unauthorized download.
- No unauthorized training.
- Holdout protection.
- Subject-disjoint split.
- Duplicate source detection.
- Original-vs-augmented image grouping.
- Annotation coordinate validation.
- Missing annotation handling.
- SAHI tile coordinate round trip.
- Tile overlap duplicate suppression.
- Phone-negative images.
- Driver/passenger ambiguity.
- Mounted phone/camera negatives.
- UNKNOWN/NEEDS_REVIEW handling.
- Short phone events.
- No ground-truth leakage.
- Seatbelt UNCERTAIN.
- NumericHead artifact validation.
- Event metric reconciliation.

Chạy:
- Python compile.
- Ruff.
- Targeted pytest.
- Full pytest.
- Backend smoke.
- Frontend build.
- Data manifest validation.
- Model artifact validation.

So sánh với baseline V15: 629 full tests PASS theo báo cáo.

Không báo PASS nếu chưa thực sự chạy.

---

# 17. REQUIRED DELIVERABLES

Tạo khi có kết quả thực:

reports/v16_data_ai85/

Gồm:

01_V15_CODE_AUDIT.md
02_DATASET_RESEARCH_AND_LICENSES.md
03_DATASET_SELECTION_MATRIX.md
04_DATA_INGESTION_AND_QA.md
05_ANNOTATION_AND_SPLIT_PROTOCOL.md
06_PHONE_DETECTION_RESULTS.md
07_ACTION_RECOGNITION_RESULTS.md
08_ASSOCIATION_FUSION_ANALYSIS.md
09_TEMPORAL_EVENT_RESULTS.md
10_SEATBELT_RESULTS.md
11_END_TO_END_METRICS.md
12_RUNTIME_AND_TESTS.md
13_FINAL_AI_LEAD_DECISION.md

Machine-readable:
- dataset_registry.json
- license_audit.json
- data_manifest.json
- label_mapping.json
- annotation_qa.json
- group_split_manifest.json
- experiment_registry.json
- metrics_comparison.json
- model_manifest.json
- candidate_readiness.json
- reproduction commands.

Không tạo kết quả giả hoặc báo cáo rỗng.

---

# 18. EXECUTION CONTRACT

BẮT ĐẦU NGAY TRONG LOCAL WORKSPACE.

Thực hiện theo thứ tự:

1. Bảo toàn V15 local commit và thay đổi prompt.md.
2. Audit dataset sources và licenses.
3. Lập danh sách datasets đủ điều kiện.
4. Kiểm tra dữ liệu Roadwatch hiện có.
5. Hoàn thiện dataset ingestion/validation code.
6. Chuẩn hóa label mapping và annotation schema.
7. Chuẩn bị human-review workflow.
8. Xác minh V15 tiled detector.
9. Chạy inference/benchmark trên dữ liệu được phép.
10. Phân tích error propagation.
11. Training/fine-tuning chỉ khi đủ điều kiện.
12. Nối candidate vào event evaluation.
13. Kiểm tra seatbelt.
14. Chạy toàn bộ tests khả thi.
15. Báo kết quả và blockers trung thực.

Không tự push GitHub.

Không thay production model.

Không thay GT, test split, matcher hoặc denominators.

Khi hoàn thành phải trả lời:

- Dataset nào đã xác minh?
- Dataset nào được phép tải/train?
- Dataset nào bị chặn, vì sao?
- Những dataset nào thực sự được bổ sung?
- Có bao nhiêu original images, subjects, videos và events?
- Train/validation/test có group-disjoint không?
- Đã sửa những source files nào?
- SAHI có cải thiện Phone AP thật không?
- Phone Event Recall/F1 có tăng thật không?
- SAFE Recall có tăng không?
- Seatbelt Macro-F1 có tăng không?
- FA/min thay đổi thế nào?
- Independent evaluation đã có chưa?
- Có đủ điều kiện đạt AI85 chưa?
- Những việc nào cần sự chấp thuận của người dùng?

**KHÔNG ĐƯỢC NHẦM LẪN GIỮA “THÊM NHIỀU DỮ LIỆU” VÀ “THÊM DỮ LIỆU ĐÚNG, HỢP LỆ, CÓ NHÃN CHUẨN VÀ ĐÁNH GIÁ ĐỘC LẬP”.**

MỤC TIÊU CUỐI CÙNG: CẢI THIỆN ROADWATCH BẰNG DATA-CENTRIC AI, COMPUTER VISION VÀ NGHIÊN CỨU KHOA HỌC CÓ THỂ TÁI LẬP.