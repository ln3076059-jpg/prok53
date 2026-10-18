# ROADWATCH V19 — MASTER EXECUTION PROMPT
## SENIOR AI ENGINEERING · DEEP LEARNING · SCIENTIFIC RESEARCH · REAL METRIC IMPROVEMENT

### MỤC TIÊU: TỪ CHECKPOINT ĐÃ TRAIN ĐẾN EVENT F1 ĐƯỢC ĐO THẬT

Bạn là hội đồng kỹ thuật gồm:

- Principal AI Research Scientist.
- Senior Computer Vision Engineer.
- Senior Machine Learning / Deep Learning Researcher.
- Senior Video Action Recognition Specialist.
- Senior Python & Backend Developer.
- Senior MLOps / Model Evaluation Engineer.
- Data Engineering and Annotation Specialist.
- Scientific Research Reviewer.
- Senior Software Quality and Performance Engineer.

**NHIỆM VỤ CHÍNH:**

Đọc toàn bộ code liên quan của Roadwatch. Kiểm tra các mô hình đã huấn luyện, dữ liệu thực tế, pipeline inference, lỗi đã sửa ở V18 và nghiên cứu khoa học hiện có. Sau đó trực tiếp triển khai, kiểm thử và đo lường các cải tiến kỹ thuật có thể thực hiện ngay.

**KHÔNG CHỈ VIẾT BÁO CÁO. KHÔNG TRAIN LẠI TỪ ĐẦU NẾU CHƯA CHỨNG MINH CẦN THIẾT. KHÔNG DÙNG gA-1 ĐỂ TỐI ƯU MÔ HÌNH.**

---

# 0. PROJECT CONTEXT — ĐỌC TRƯỚC KHI LÀM

GitHub:
https://github.com/ln3076059-jpg/prok53

Branch:
roadwatch-v4-candidate

Workspace:
D:\.idea\giangdoantotnghiep

Last verified HEAD:
ba97f876ce3d4b6fc976455d02c6abc3df507447

Last commit:
research: add V18 code audit, gA-1 holdout protection, and verified bug fixes

Trước khi chạy phải xác minh HEAD mới nhất thay vì mặc định commit trên còn là HEAD hiện tại.

## Existing training history

Người dùng đã huấn luyện:

- YOLO phone detector.
- YOLO seatbelt detector.
- Seatbelt classifier.
- V5 ResNet18-GRU action models.
- V6 hierarchical/flat action models.
- V7 and AI85 research checkpoints.
- Other feature-based candidate models.

V10 từng kiểm kê 64 checkpoint trên local filesystem, bao gồm nhiều bản sao trùng hash.

V18 báo cáo 59 model files trong `models/`.

Các con số này có scope khác nhau. Không coi đó là mâu thuẫn nếu chưa so sánh inventory root, thời điểm và byte-identical files.

## Current historical metrics

Phone Event:
- TP: 16.
- FP: 4.
- FN: 13.
- Precision: 80.00%.
- Recall: 55.17%.
- F1: 65.31%.
- FA/min: 0.1813.

Action E3 development:
- Clip Macro-F1: 56.39%.
- SAFE Recall: 60.38%.
- PHONE Recall: 66.78%.
- OTHER Recall: 43.26%.

Seatbelt:
- Development Macro-F1: 58.52%.
- UNCERTAIN Recall: 4%.

Historical trained-detector validation metrics reported in V18:
- Phone Detector mAP50: 94.48%.
- Phone Detector mAP50-95: 70.03%.
- Phone Detector Precision: 97.13%.
- Phone Detector Recall: 82.73%.
- Seatbelt Detector mAP50: 94.73%.
- Seatbelt Classifier Top-1: 80.97%.

**BẮT BUỘC phân biệt detector-level AP, clip-level Macro-F1, event-level F1 và test-set domain.**

Dữ liệu DMD `gA-1` đã được người dùng tải.

V18 báo cáo đường dẫn:
dataset/dmd-dataset-distraction-gA-1/dmd/gA/1

Không giả định file còn ở đúng đường dẫn nếu chưa xác minh trên máy.

## Current research restrictions

- `gA-1`: reserved research holdout, nhưng đã có ít nhất một frame/label được V18 kiểm tra.
- `gZ-37`, `gE-28`: previously consumed/protected.
- `gB-9`, `gC-14`, `gZ-36`: repeatedly used development subjects.
- No confirmed independent multi-subject AI85.
- No compatible, verified fitted OnlineE3 head/scaler artifact.
- Production: not approved.

---

# 1. NON-NEGOTIABLE ENGINEERING RULES

1. Đọc `AGENTS.md` và project instructions nếu tồn tại.
2. Kiểm tra `git status`, `git log`, local/remote HEAD.
3. Giữ nguyên mọi uncommitted changes.
4. Không reset, clean, force checkout, xóa checkpoint hoặc dataset.
5. Không sửa `prompt.md` nếu không cần.
6. Không tự push, merge hoặc force-push.
7. Không bật production.
8. Không commit raw DMD video, private annotations, signed download URL hoặc credentials.
9. Không thay đổi ground truth, evaluator hoặc event matcher để tăng điểm.
10. Không tạo số liệu giả trong Markdown/JSON.
11. Không báo `PASS`, `READY` hoặc `IMPROVED` nếu chưa có phép thử đúng phạm vi.
12. Không dùng synthetic NumericHead làm bằng chứng model accuracy.
13. Không sử dụng gA-1 để train hoặc điều chỉnh threshold.
14. Không tự đánh dấu data rights approved.
15. Không chọn model từ independent test results.
16. Không viết lại module cũ khi có thể tích hợp và sửa tận gốc.

---

# 2. PHASE A — AUDIT TOÀN BỘ SOURCE CODE

Đây là yêu cầu bắt buộc.

Không chỉ đọc các báo cáo V17/V18. Phải mở source code thực tế.

Kiểm tra toàn bộ first-party code liên quan trong:

backend/
backend/ai/
training/
training/dmd/
dataset/
models/
tests/
scripts/
frontend/

Đọc các entry point FastAPI, inference worker, camera ingestion, video processing, tracking, alerts, database storage, frontend rendering và test pipeline.

Sử dụng repository search để phát hiện:

- Active model loaders.
- `YOLO(...)`.
- `torch.load(...)`.
- Training entry points.
- Model selection.
- Event matcher.
- Thresholds.
- Temporal configurations.
- Feature preprocessing.
- Ground-truth adapters.
- Data split code.
- License/holdout guards.

## Reconstruct actual call graph

VIDEO INPUT
→ DECODING
→ VEHICLE / CABIN DETECTION
→ OCCUPANT / DRIVER ROI
→ PHONE DETECTOR
→ SOURCE-RESOLUTION / SAHI
→ ACTION FEATURE ENCODER
→ TRAINED ACTION HEAD
→ DRIVER-PHONE ASSOCIATION
→ FUSION
→ TEMPORAL EVENT ENGINE
→ ALERT
→ EVENT EVALUATION

Với mỗi bước:

- Source file and function.
- Called by which module.
- Called in production or research.
- Input/output schemas.
- Source model checkpoint.
- Preprocessing.
- Threshold source.
- Missing evidence semantics.
- Runtime cost.
- Existing tests.
- Failure modes.

## Output

Tạo machine-readable module graph:

reports/v19_metric_recovery/pipeline_graph.json

Và báo cáo source-backed:

reports/v19_metric_recovery/01_FULL_CODE_AUDIT.md

Không tuyên bố “đã đọc toàn bộ code” nếu mới đọc 7–12 files.

Ghi rõ số file thực sự đã inspect và các phần chưa inspect.

---

# 3. PHASE B — VERIFY TRAINED CHECKPOINTS

Kiểm tra trực tiếp:

models/active/
models/candidate/
models/locked/
models/recovery_extracted/
kaggle/
runs/

Đặc biệt:

models/active/v2/phone_detector.pt
models/active/v2/seatbelt_detector.pt
models/active/v2/seatbelt_classifier.pt
models/auxiliary/yolo11n.pt
models/auxiliary/yolo11n-pose.pt

V5/V6/V7/AI85 LOSO checkpoints.

Tìm:

best.pt
last.pt
results.csv
args.yaml
data.yaml
metadata.json
dataset_lock.json
model_lock.json
*.npz
*.pt

## Checkpoint contract

Với từng mô hình:

- File exists?
- SHA-256?
- Byte-identical duplicates?
- Architecture?
- Number of classes?
- Class names and order?
- Preprocessing?
- Input resolution?
- Training dataset?
- Train/validation subjects?
- Best epoch?
- Last epoch?
- Source metrics?
- Inference runnable?
- Is the checkpoint compatible with current code?

Không deserialize untrusted pickle checkpoint trong môi trường chính.

Đối chiếu old training manifest với raw training log khi có.

Báo rõ:

RECOVERED_TRAINING_HISTORY
REPRODUCED_VALIDATION_METRICS
PROVENANCE_INCOMPLETE
INFERENCE_COMPATIBLE
NOT_FOUND_LOCALLY
UNSAFE_TO_LOAD

## Essential distinction

`0 newly approved training rows` không có nghĩa là `0 trained checkpoints`.

`No fitted OnlineE3 numeric head` không có nghĩa là `No trained V5/V6 action model`.

Không tự xóa lịch sử huấn luyện chỉ vì adapter E3 mới chưa hỗ trợ các trọng số cũ.

Nếu action checkpoint không tương thích OnlineE3, ưu tiên tìm hoặc tạo một adapter đọc artifact phù hợp mà không tự fit lại.

---

# 4. PHASE C — REVIEW V17 AND V18 REPORT RELIABILITY

**Đây là P0.**

Kiểm tra:

training/dmd/generate_v17_research_reports.py
training/dmd/generate_v18_research_reports.py

reports/v17_research/
reports/v17_ai85_recovery/
reports/v18_research/

V17 có script tạo văn bản với một số con số root-cause và runtime được ghi trực tiếp trong chuỗi Markdown.

Vì vậy, phải đối chiếu từng claim với dữ liệu gốc.

## Verify:

1. 13 FN có per-event prediction records không?
2. Nhóm “5 short events” có nguồn timestamp độc lập không?
3. Nhóm “11 detector misses” có raw detections và frame hashes không?
4. 4 FP có frame-level evidence không?
5. `35.6ms/frame` hoặc `28.1 FPS` có log benchmark thật không?
6. Có nhầm detector crop với SAHI inference không?
7. Có nhầm GT-aware diagnostic với autonomous inference không?
8. Có mâu thuẫn giữa README/report và active configuration không?
9. Có lời khẳng định “resolved” trong khi chưa có event replay sau sửa lỗi không?
10. Có kết quả derived từ một frame duy nhất bị gọi là model improvement không?

Phân loại mọi claim:

VERIFIED_FROM_RAW_OUTPUT
VERIFIED_FROM_CODE
SUPPORTED_BY_TEST
REPORTED_NOT_REPRODUCED
HYPOTHESIS
UNSUPPORTED
CONTRADICTED

Không xóa báo cáo cũ; tạo bản audit làm rõ sự khác biệt.

Không sửa những con số lịch sử mà không có raw evidence.

---

# 5. PHASE D — DMD gA-1 HOLDOUT INTEGRITY

DMD File 1 đã được tải và giải nén theo báo cáo V18.

Reported dataset:
- Subject: gA-1.
- Sessions: s1, s2, s3.
- BODY, FACE, HANDS video streams.
- OpenLABEL/VCD annotations.

## Important concern

V18 đã nghiên cứu một frame tại giây 60 thuộc `gA-1 s2`, dùng nhãn `phonecall_right` để xác định bối cảnh.

Một báo cáo khác ghi video `s2` dài 458.1s với 8 GT phone events.

Do đó, gA-1 đã có exposure ở cấp frame/label.

Không được tiếp tục tuyên bố `gA-1` hoàn toàn unseen cho mọi quyết định mô hình nếu thông tin đó đã ảnh hưởng đến thiết kế.

## Mandatory provenance audit

Xác minh:

- File/archive SHA-256.
- Exact dataset path.
- Source revision.
- Permissions.
- Original videos.
- Annotation files.
- Video metadata.
- Checkpoint training subject overlap.
- Feature caches.
- Evaluation logs.
- Reports using gA-1.
- Git commits made after viewing gA-1.

Ghi rõ những dữ kiện nào đã được dùng để:

- Debug software.
- Select detector.
- Select crop.
- Change threshold.
- Design fusion.
- Choose model.
- Report final accuracy.

## Holdout decision

Nếu gA-1 chỉ được dùng như structural smoke test với code/protocol đã khóa và không ảnh hưởng model selection, ghi rõ giới hạn đó.

Nếu đã dùng GT hoặc prediction outcome để chọn phương pháp:

mark as:

DEVELOPMENT_EXPOSED_OR_PARTIALLY_CONTAMINATED

Không tự động chuyển nó trở lại `PRISTINE_INDEPENDENT`.

Giữ gA-1 bị chặn khỏi fitting và calibration.

Nếu cần một final independent test hoàn toàn sạch, phải chọn subject mới khác được phép truy cập và khóa từ trước.

Không tự coi nhiều session của một người là nhiều independent subjects.

---

# 6. PHASE E — REPRODUCE REAL BASELINES

Phải xây dựng runner cho:

1. Detector-only validation.
2. Action-only validation.
3. Frozen event replay.
4. End-to-end inference.
5. Seatbelt-only validation.

Dùng dữ liệu development/validation được phép theo provenance thực tế.

Nếu chỉ có cached predictions hợp lệ, chạy evaluation từ cached predictions và ghi đúng scope.

## Phone detector

Reproduce metrics:
- P.
- R.
- AP50.
- AP50-95.
- Per-size recall.
- Negative-frame false positives.

## Action

Reproduce:
- Clip Macro-F1.
- Per-class recall.
- Subject-level metrics.
- Confusion matrices.
- SAFE/PHONE class collapse.

## Event

Reproduce:
- TP 16.
- FP 4.
- FN 13.
- F1 65.31%.
- FA/min 0.1813.

Nếu không thể reproduce:

Ghi NOT_REPRODUCED và nguyên nhân.

Không tự kế thừa metric từ báo cáo để gọi là “new evaluation”.

---

# 7. PHASE F — ROOT-CAUSE ANALYSIS

Mục tiêu là trả lời:

**VÌ SAO PHONE DETECTOR CÓ mAP50 CAO NHƯNG PHONE EVENT RECALL CHỈ 55.17%?**

Phải kiểm tra thực nghiệm từng stage.

## Detection stage

- Small phone pixel resolution.
- Full-frame resize.
- Driver ROI source resolution.
- Specialist YOLO.
- Auxiliary COCO YOLO.
- SAHI/tiled detector.
- Confidence filtering.
- NMS.
- Crop boundary.
- Missing driver ROI.

## Action stage

- Correct trained checkpoint loaded?
- Correct architecture?
- Correct class map?
- Correct preprocessing?
- Training-vs-inference resolution shift?
- Causal vs bidirectional architecture?
- Temporal sampling gaps?
- SAFE/PHONE confusion?
- Domain shift between subjects?

## Association stage

- Is driver identity actually verified?
- Is occupant track stable?
- Passenger spillover?
- Hand-phone interaction?
- Mounted-phone context?
- UNKNOWN coverage?

## Fusion stage

- Is action score calibrated?
- Is `min(phone_score, action_score)` blocking valid evidence?
- Strict HANDHELD state?
- Contradictory evidence?
- Missing view behavior?
- Why did each candidate get rejected?

## Temporal stage

- Active config loaded from where?
- Is minimum duration 0.5s or 1.2s in each experiment?
- Is minimum observations 2 or 3?
- Actual sample cadence?
- Stream reset due to dropped frames?
- Does UNKNOWN interrupt evidence accumulation?
- Event hysteresis?
- Cooldown?
- Short event fragmentation?

## Event matcher

- Timestamp alignment.
- Ground-truth boundaries.
- Predicted boundaries.
- IoU tolerance.
- Onset tolerance.
- Duplicate matching.
- Exposure denominator.

## FN/FP output

Với 13 FN + 4 FP cũ, lưu per-case diagnostic records nếu có nguồn dữ liệu hợp lệ.

Một FN được phép có nhiều contributing causes.

Phải báo first blocking stage tách khỏi secondary causes.

Không dùng các nhóm nguyên nhân chồng lấp như partition độc lập.

---

# 8. PHASE G — MEASURE THE IMPACT OF V18 BUG FIXES

Các lỗi V18 đã báo sửa:

E1 — HANDHELD/HANDHELD_USE.

E2 — Temporal engine reset on unadmitted frame.

E4 — Subject split false PASS.

E5 — SAFE_DRIVING with visible non-handheld phone.

**Những lỗi này đã có regression tests, nhưng đó chưa phải event-level accuracy evidence.**

## Required controlled replays

R0 — Pre-V18 code and frozen baseline.

R1 — Only E1 fix enabled.

R2 — Only E2 fix enabled.

R3 — E1 + E2 enabled.

R4 — E1 + E2 + existing original-resolution phone proposals.

R5 — Selected engineering candidate.

Cố định:
- Same video cohort.
- Same annotations.
- Same checkpoints.
- Same model preprocessing.
- Same event matcher.
- Same scoring definitions.

Không đưa gA-1 vào vòng lựa chọn R0–R5.

## Metrics

TP, FP, FN, P, R, F1, FA/min.

Recovered TP IDs.
Lost TP IDs.
New FP IDs.
Removed FP IDs.

Short-event recall.
Fragmentation.
Emission delay.
Abstention coverage.

Không gọi R5 là tốt hơn nếu chỉ có candidate proposals mà chưa tạo matched events.

Nếu không có replay inputs thật:

IMPLEMENT_REPLAY_RUNNER_AND_TESTS

và ghi:

REAL_EVENT_DELTA_NOT_MEASURED

Không tạo fake success records.

---

# 9. PHASE H — SCIENTIFIC RESEARCH BEFORE NEW MODEL CHANGES

Đọc trực tiếp paper và official code.

## Priority 1 — SAHI

Paper:
https://doi.org/10.1109/ICIP46576.2022.9897990

Code:
https://github.com/obss/sahi

Ứng dụng:
Small phone detection bằng tiled inference.

Cần kiểm tra:
- ROI tile size.
- Overlap.
- Box merging.
- Small object recall.
- Negative FP rate.
- Latency.

Ưu tiên inference ablation với checkpoint cũ.

## Priority 2 — DualScope

Paper:
https://doi.org/10.1609/aaai.v40i10.37812

Code candidate:
https://github.com/Qqqzzjsupr3me/DualScope

Ứng dụng:
Behavior-centric spatial and temporal representation.

## Priority 3 — DriveCLIP

Paper:
https://doi.org/10.1109/TITS.2024.3381175

Code:
https://github.com/zahid-isu/DriveCLIP

Ứng dụng:
Compare action representation against existing V5/V6/E3.

## Priority 4 — ActionFormer

Paper:
https://arxiv.org/abs/2202.07925

Official implementation:
https://github.com/happyharrycn/actionformer_release

Ứng dụng:
Short-event temporal boundary reasoning.

Không triển khai full Transformer nếu lightweight causal state-machine correction đã giải quyết bottleneck.

## Priority 5 — MixStyle

https://arxiv.org/abs/2104.02008

Ứng dụng:
Subject/cabin generalization.

## Priority 6 — Group DRO

https://arxiv.org/abs/1911.08731

Ứng dụng:
Worst-subject performance.

## Priority 7 — ByteTrack

https://arxiv.org/abs/2110.06864

Ứng dụng:
Track continuity, không phải driver identity ground truth.

## Priority 8 — DMD publications

https://dmd.vicomtech.org/

Đọc original DMD papers và annotation specifications.

## Priority 9 — Multi-Task Fusion

https://doi.org/10.3390/s25216799

Ứng dụng:
Hands, face, body and gaze evidence fusion.

## Priority 10 — Seatbelt Attention

https://doi.org/10.3390/app14083346

Ứng dụng:
Torso crop, visibility and seatbelt classification.

## Scientific evaluation requirement

Mỗi paper cần:

- Verified DOI.
- Year/venue.
- Official source.
- Code repository.
- License.
- Dataset.
- Camera domain.
- Train/validation/test split.
- Actual metric definition.
- Published result.
- Hardware.
- Inference latency if genuinely measured.
- Applicability.
- Implementation cost.
- Evidence gaps.

Nếu source/code không xác minh được, ghi UNVERIFIED.

Không đưa kết quả paper vào cột Roadwatch candidate accuracy.

---

# 10. PHASE I — MODEL IMPROVEMENT STRATEGY

Chỉ chọn phương pháp sau root-cause analysis.

## P0 — Inference fixes

- Reuse trained checkpoint.
- Fix invalid class/context mappings.
- Fix temporal state handling.
- Fix crop transforms.
- Fix source resolution.
- Improve negative-context filtering.
- Preserve evidence provenance.

## P1 — Feature improvements

Nếu action performance mới là bottleneck:

- High-resolution hand/phone features.
- Torso/face interaction.
- Causal frame differences.
- Multi-region appearance.
- Frozen feature extractor.
- Lightweight action head.

## P2 — Fine-tuning

Chỉ khi cần thay weights và có:

- Authorized training data.
- Approved labels.
- Subject-disjoint splits.
- Frozen validation protocol.
- Compatible checkpoint.
- Sufficient hardware.

Warm-start từ best previous checkpoint.

Không full train từ scratch nếu chưa chứng minh warm-start không phù hợp.

Không dùng gA-1 hoặc protected subjects.

Không tự download/checkpoint pretrained model có license không phù hợp.

---

# 11. PHASE J — SEATBELT MODEL RECOVERY

Không bỏ qua seatbelt chỉ vì đang tập trung phone.

Kiểm kê trained seatbelt detector/classifier.

Kiểm tra:
- Existing Top-1 80.97% context.
- Three-state class mapping.
- FASTENED/UNFASTENED confusion.
- UNCERTAIN recall.
- Torso ROI quality.
- False-unfastened rate.
- Model activation state.

500 seatbelt review candidates vẫn cần human validation.

Không tự approve.

Khi có approved labels:
- Warm-start existing model.
- Test visibility-aware classification.
- Try lightweight attention.
- Preserve UNCERTAIN state.
- Evaluate Macro-F1/per-class Recall.

Nếu chưa có quyền:
- Finish engineering and review tooling.
- Keep training blocked.

---

# 12. PHASE K — EVALUATE gA-1 CORRECTLY

Trước khi chạy bất kỳ new independent evaluation:

- Audit prior gA-1 exposure.
- Determine whether gA-1 influenced method selection.
- Freeze checkpoint and code.
- Freeze all thresholds.
- Freeze preprocessing.
- Freeze event matcher.
- Freeze labels mapping.
- Freeze metric definitions.

Nếu gA-1 đã được dùng để chọn ROI/detector, không báo một lần test lại trên chính gA-1 là pristine independent validation.

Có thể dùng gA-1 như development-exposed diagnostic nếu quyền cho phép; khi đó báo đúng status và không tuning tiếp nếu vẫn muốn giữ nó làm holdout hạn chế.

Nếu gA-1 còn đủ điều kiện test theo audit:
- Perform controlled evaluation only with frozen candidate.
- Produce exact TP/FP/FN.
- Report event exposure.
- Report limitations of single-subject panel.

Nếu không:
- Seek a NEW, licensed, untouched independent subject.
- Lock it before touching outcome labels.

Không tuyên bố AI85 chỉ vì kết quả tốt trên một người lái.

---

# 13. PHASE L — SCIENTIFIC METRICS

Các mục tiêu:

Phone Event Precision >85%.
Phone Event Recall >85%.
Phone Event F1 >85%.
False Alarms/min <=0.15.

SAFE Recall >85%.
PHONE Recall >85%.
Action Macro-F1 >85%.

Seatbelt Macro-F1 >85%.
UNCERTAIN Recall >85%.

Detector AP50 >85% trên một benchmark bbox hợp lệ.

## Mathematical integrity

Event Precision = TP/(TP+FP).
Event Recall = TP/(TP+FN).
Event F1 = 2PR/(P+R).

FA/min = FP / evaluated negative exposure minutes theo protocol đã khóa; không âm thầm thay exposure definition.

Baseline 29 events:

Để đạt Recall >85%, cần ít nhất 25 TP trên 29 GT events.

Nếu FP vẫn bằng 4, FA/min vẫn khoảng 0.1813 trên 22.06 phút, nên chưa đạt mục tiêu ≤0.15.

Với exposure không đổi, tối đa 3 FP mới đạt ≤0.15.

Đây chỉ là mục tiêu phát triển trên cohort cũ, không phải independent certification.

## Required output

Baseline vs Candidate:
- TP.
- FP.
- FN.
- P.
- R.
- F1.
- FA/min.
- Short Event Recall.
- SAFE Recall.
- Worst-subject SAFE Recall.
- Seatbelt Macro-F1.
- Runtime.
- Uncertainty/confidence intervals where possible.

Nếu phép đo không tồn tại:
NOT MEASURED.

Không điền giá trị 0 để thay thế unavailable metric.

---

# 14. PHASE M — RUNTIME PERFORMANCE

Báo cáo trước từng ghi khoảng 28 FPS, nhưng V18 ghi CPU inference khoảng 1 FPS.

Phải phân biệt môi trường, hardware và scope.

Chạy benchmark thực tế nếu có resources.

Báo:
- CPU/GPU.
- CUDA.
- Input video size.
- Number of cameras.
- Decode latency.
- Phone detector latency.
- SAHI inference latency.
- Action inference latency.
- Association/fusion latency.
- Full pipeline latency.
- Effective FPS.
- Peak memory.
- Alert emission delay.

Không gọi cached-feature FPS là camera runtime FPS.

Không sử dụng số liệu ước lượng để công bố performance thực tế.

---

# 15. PHASE N — REQUIRED TESTS

Bắt buộc:

- Existing V18 regression tests.
- Full pytest.
- Python compile.
- Ruff.
- Backend smoke.
- Frontend build.
- Checkpoint integrity.
- Dataset permissions.
- Holdout protection.
- Annotation schema.
- Group split.
- No GT leakage.
- Event metric reconciliation.

Thêm tests:

1. Old V2 detector checkpoint loads with correct class mapping.
2. V5/V6 action model preprocessing contract.
3. Correct temporal config is recorded for each replay.
4. Missing action head cannot produce fake E3 scores.
5. HANDHELD aliases work.
6. UNKNOWN frame does not corrupt evidence state.
7. Stream reset occurs on actual session/track switch.
8. Timestamp gaps are handled explicitly.
9. Phone bbox scaling is correct.
10. SAHI overlapping tile duplicates merge correctly.
11. Mounted phone is not automatic driver violation.
12. Passenger phone is not automatic driver violation.
13. SAFE with visible but unused phone is valid.
14. Unverified driver association remains UNKNOWN.
15. Confirmed alert metrics exclude NEEDS_REVIEW.
16. No GT boxes enter autonomous inference.
17. `gA-1` is excluded from train/calibration.
18. Previously consumed holdouts stay protected.
19. Raw DMD files cannot be staged for Git commit.
20. Event TP/FP/FN reconcile exactly.
21. Runtime benchmarks cannot masquerade as accuracy measurements.
22. Generated reports cannot claim measurements without source artifacts.

## Test result report

Track:
- Tests run.
- Tests passed.
- Tests failed.
- Tests skipped.
- Test log path.
- Environment.
- Before/after regression findings.

659 PASS is the previous reported baseline, not a guarantee of this run.

---

# 16. PHASE O — DELIVERABLES

Create:

reports/v19_metric_recovery/

Suggested artifacts, only when genuinely supported:

01_FULL_CODE_AUDIT.md

02_CHECKPOINT_RECOVERY.md

03_HISTORICAL_METRIC_REPRODUCTION.md

04_V17_V18_REPORT_RELIABILITY_AUDIT.md

05_GA1_HOLDOUT_CONTAMINATION_AUDIT.md

06_SCIENTIFIC_LITERATURE_MATRIX.md

07_FN_FP_ROOT_CAUSE.md

08_V18_FIX_EVENT_REPLAY.md

09_PHONE_DETECTOR_SAHI_ABLATION.md

10_ACTION_MODEL_RECOVERY.md

11_FUSION_TEMPORAL_INVESTIGATION.md

12_SEATBELT_RECOVERY.md

13_RUNTIME_AND_TEST_RESULTS.md

14_FINAL_AI_LEAD_DECISION.md

Machine-readable:

- pipeline_graph.json
- checkpoint_inventory.json
- training_history.json
- report_claim_audit.json
- holdout_exposure_audit.json
- root_cause_events.json
- replay_experiments.json
- model_manifest.json
- metrics_comparison.json
- tests_summary.json
- candidate_readiness.json
- reproduction_commands.ps1

Không bắt buộc sinh file nếu không có nội dung thật; các blocker có thể được gộp thành một tài liệu ngắn có bằng chứng.

Không đưa raw video hoặc personal identifiers vào public reports.

---

# 17. WINDOWS POWERSHELL EXECUTION GUIDANCE

Trong workspace, kiểm tra trước:

```powershell
Set-Location "D:\.idea\giangdoantotnghiep"
git status --short
git branch --show-current
git rev-parse HEAD
git log -5 --oneline
```

Kiểm tra các checkpoint tiêu biểu:

```powershell
Get-ChildItem .\models -Recurse -File -Include *.pt,*.npz | Select-Object FullName,Length
```

Kiểm tra V18 tests:

```powershell
py -m pytest tests/test_v18_bug_fixes.py -v
```

Full verification khi đủ dependencies:

```powershell
py -m compileall -q backend training tests
py -m ruff check backend training tests
py -m pytest -q
```

Nếu command/script khác trong repository phù hợp hơn, dùng lệnh thực tế và ghi lại command.

Không chạy full high-resolution inference trên nhiều GB video nếu chưa kiểm tra CPU/GPU và cost.

Không tự tải dữ liệu lớn.

Không chạy fitting hoặc upload media khi chưa được phép.

---

# 18. EXECUTION PRIORITY — KHÔNG DỪNG Ở BÁO CÁO

## Stage 1: SOURCE VERIFICATION

1. Audit Git.
2. Audit repository.
3. Audit checkpoints.
4. Audit training histories.
5. Audit existing reports.
6. Audit gA-1 exposure.

## Stage 2: ROOT-CAUSE RECOVERY

7. Reproduce baseline from raw records.
8. Verify detector/action/fusion configurations.
9. Trace 13 FN and 4 FP.
10. Verify V18 bug fixes.
11. Identify first blocking stage for each event.

## Stage 3: IMPLEMENTATION

12. Fix proven code defects.
13. Build/repair reproducible replay.
14. Run pre/post-V18 event comparison.
15. Try frozen source-resolution phone proposals.
16. Compare with baseline under fixed protocol.
17. Audit seatbelt checkpoint behavior.

## Stage 4: SCIENTIFIC EVALUATION

18. Verify applicable research literature.
19. Select no more than 3 high-value experiments.
20. Run permitted experiments.
21. Keep/reject each change based on true metrics.
22. Recommend warm-start fitting only if needed.

## Stage 5: DELIVERY

23. Execute tests.
24. Record runtime.
25. Verify no raw DMD data is staged.
26. Produce source-backed final report.
27. Leave production unchanged.
28. Do not push GitHub without explicit authorization.

---

# 19. DEFINITION OF DONE

## Engineering Done

- Actual source code audited.
- Historical models recovered.
- No checkpoint overwritten.
- Correct preprocessing verified.
- V18 bugs regression-tested.
- Replay tool runnable.
- Raw metric records traceable.
- Tests passed or blockers explained.

## Scientific Improvement Done

- New predictions actually generated.
- Same cohort and frozen metric protocol.
- TP/FP/FN compared.
- Improvement attributable to controlled change.
- No outcome-based tuning on independent holdout.
- Negative context false alarms evaluated.
- Results reproducible.

## AI85 Independently Established

- Independent dataset not exposed to model selection.
- Locked model and thresholds.
- Verified GT and permissions.
- Adequate subject/event coverage.
- Event P/R/F1 above target.
- False alarms within target.
- Per-subject results.
- Reproducible raw outputs.
- No leakage.
- Proper uncertainty and limitations.

Nếu chưa đạt, ghi:

INDEPENDENT_AI85_NOT_ESTABLISHED

Không tự đổi production_approved thành true.

---

# 20. FINAL MANDATORY RESPONSE

Báo cáo cuối phải trả lời:

1. HEAD hiện tại là gì?
2. Đã đọc bao nhiêu source files?
3. Model nào đã train và tồn tại thực tế?
4. Có bao nhiêu unique checkpoint hashes?
5. Checkpoint tốt nhất của từng task là gì?
6. Metrics cũ nào đã reproduce?
7. Metrics nào chỉ kế thừa từ báo cáo?
8. Có cần train tiếp không và vì sao?
9. 13 FN thực sự bị chặn ở đâu?
10. 4 FP có thể loại bỏ bằng cách nào?
11. Những bug V18 nào đã chứng minh sửa đúng?
12. Có chạy event replay trước/sau không?
13. Baseline và candidate TP/FP/FN bằng bao nhiêu?
14. Recall/F1 có tăng thật không?
15. Có tăng false alarms không?
16. SAHI có cải thiện AP hay chỉ tăng proposals?
17. SAFE Recall có cải thiện không?
18. Worst-subject recall có cải thiện không?
19. Seatbelt classifier còn lỗi gì?
20. gA-1 đã bị exposure ở mức nào?
21. Có independent test hợp lệ chưa?
22. Bài báo khoa học nào thực sự được dùng?
23. Runtime là bao nhiêu trên hardware nào?
24. Tests PASS/FAIL thực tế?
25. Còn những blockers nào?
26. Ba hành động tiếp theo có giá trị nhất là gì?

**FINAL PRINCIPLE:**

ĐỪNG HUẤN LUYỆN LẠI CHỈ VÌ F1 THẤP.

ĐỪNG TUYÊN BỐ BUG FIX TĂNG ACCURACY KHI CHƯA CÓ REAL EVENT REPLAY.

ĐỪNG DÙNG DMD gA-1 ĐỂ VỪA PHÁT TRIỂN VỪA ĐÁNH GIÁ ĐỘC LẬP.

ĐỪNG CHO RẰNG 659 PYTEST PASS TƯƠNG ĐƯƠNG AI85.

**HÃY TẬN DỤNG CHECKPOINT ĐÃ TRAIN, CHỨNG MINH ROOT CAUSE BẰNG RAW OUTPUTS, SỬA ĐÚNG TẦNG GÂY LỖI, VÀ ĐO MỨC TĂNG AI THẬT SỰ.**

BẮT ĐẦU THỰC HIỆN TRỰC TIẾP TRONG WORKSPACE.