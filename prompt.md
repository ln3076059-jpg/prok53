# ROADWATCH V13 — AI RESEARCH LEAD MASTER PROMPT
## HOÀN THIỆN AI PIPELINE, NÂNG CAO ĐỘ CHÍNH XÁC, ĐÁNH GIÁ KHOA HỌC

Bạn là Principal AI Scientist, Senior Computer Vision Engineer, Video Action Recognition Specialist, MLOps Lead và Technical Lead của dự án Roadwatch.

Nhiệm vụ: TRỰC TIẾP KIỂM TRA, SỬA MÃ NGUỒN, TRIỂN KHAI CÁC CẢI TIẾN CÓ CƠ SỞ KHOA HỌC VÀ ĐÁNH GIÁ KẾT QUẢ THỰC TẾ.

Không dừng ở việc tạo prompt, kế hoạch hoặc file Markdown.

Không cam kết kết quả 85% khi chưa có dữ liệu và benchmark hợp lệ.

## 1. THÔNG TIN DỰ ÁN

Repository:
https://github.com/ln3076059-jpg/prok53

Branch:
roadwatch-v4-candidate

Local workspace:
D:\.idea\giangdoantotnghiep

Commit đã được xác minh:
943c723ba031e89a4ca3d5de91e17d9d7a4e746f

Nghiên cứu V12:
d1d081596090fd3177a4588d5869ddb2db3a3858

Đọc AGENTS.md nếu có.

Trước khi thay đổi:
- Kiểm tra git status, HEAD và branch.
- Kiểm tra uncommitted changes.
- Đọc reports/v11_research và reports/v12_research.
- Đọc source của candidate, fusion và temporal engine.
- Đọc quyền dữ liệu đang có hiệu lực.
- Kiểm tra checkpoint, RAM, GPU và disk.
- Không tự động reset, clean, xóa, ghi đè dữ liệu hoặc checkpoint.
- Không push hoặc merge lên main khi chưa được cho phép.

## 2. BASELINE V12

Đọc machine-readable artifacts để xác minh các số sau.

PHONE EVENT:
TP=16
FP=4
FN=13
Precision=80.00%
Recall=55.17%
F1=65.31%
FA/min=0.1813

E3 CLIP CLASSIFICATION:
Macro-F1=56.39%
SAFE Recall=60.38%
PHONE Recall=66.78%
OTHER Recall=43.26%
Worst-subject SAFE Recall=54.36%

SEATBELT:
Best development Macro-F1=58.52%
UNCERTAIN Recall=4%

ENGINEERING:
Full Pytest=618 PASS theo báo cáo V12.
OnlineE3 batch/stream feature parity=PASS.
E3 fitted head/scaler=NOT AVAILABLE.
E3 Event F1=NOT MEASURED.
Phone AP=NOT MEASURED.
Independent holdout=NOT AVAILABLE.
Production approval=FALSE.

Điểm mới của V12:
- 12/13 FN có human-selected phone boxes.
- 1/13 không có valid proposal.
- Reviewed phone width trong 64×64 cache chỉ khoảng 2.19–11.73 pixels.
- Driver/occupant association chưa có verified ground truth.
- 500 seatbelt review candidates vẫn PENDING.
- Đã phát hiện 25 rows có predict label khác argmax của normalized one-vs-rest probabilities.

Không đánh đồng các metrics thuộc nhiệm vụ hoặc tập dữ liệu khác nhau.

## 3. MỤC TIÊU NGHIÊN CỨU

PHONE EVENTS:
- Precision >=85%.
- Recall >=85%.
- F1 >=85%.
- FA/min <=0.15.

ACTION:
- SAFE Recall >=85%.
- PHONE Recall >=85%.
- Clip Macro-F1 >=85%.
- Không để một subject có Recall suy sụp.

SEATBELT:
- Macro-F1 >=85%.
- UNCERTAIN Recall >=85%.
- Giảm false-unfastened, nhưng phải báo abstention/coverage.

DETECTION:
- Phone AP50 >=85% nếu có confirmed bbox ground truth.

Các mục tiêu chỉ được xác nhận bằng independent evaluation hợp lệ.

## 4. P0 — RIGHTS, LABELS, SPLITS

Bắt buộc kiểm tra live dataset authorization.

Hiện V12 ghi:
RIGHTS_ACCEPTED=false
CANONICAL_ELIGIBLE=false
TEMPORAL_DEVELOPMENT_ELIGIBLE=true

Không tự sửa các cờ này để cho phép training.

Phân biệt:
- Quyền đọc dữ liệu.
- Quyền diagnostics.
- Quyền fitting/training.
- Quyền upload.
- Quyền xuất bản.
- Quyền đánh giá.

Cần có quyền fitting hợp lệ trước khi train linear head, calibrator, fusion head, temporal boundary model hoặc backbone trên DMD.

Không sử dụng:
gZ-37
gE-28

cho huấn luyện, tuning, model selection hoặc đọc trộm protected holdout artifacts.

Ba subjects:
gB-9
gC-14
gZ-36

chỉ là repeatedly-used development.

Nếu thiếu dữ liệu mới, không tự tạo independent holdout giả.

Xây dựng thủ tục tiếp nhận dữ liệu mới:
- Provenance.
- Dataset license.
- Consent/authorization nếu cần.
- Human-confirmed labels.
- Subject/session/camera grouping.
- Near-duplicate checking.
- Train/validation/test split locking.
- Hashes.
- Audit logs.

Tuyệt đối không để frame cùng một video rơi vào cả train và test.

Nếu quyền không cho phép fitting, chỉ thực hiện read-only diagnostics được cấp phép, synthetic fixtures, module implementation và test chuẩn bị.

## 5. P1 — GIẢI QUYẾT MẤT CHI TIẾT ĐIỆN THOẠI

Đây là ưu tiên kỹ thuật số một.

V12 xác nhận phone boxes chỉ còn 2.19–11.73 pixels trong 64×64 cache.

Hãy kiểm tra:
training/dmd/behavior_roi.py
training/dmd/driver_phone_association.py

và các ROI utilities thực sự đang tồn tại.

Thiết kế một original-resolution ROI pipeline.

So sánh:

R0 — Frozen 64×64 baseline.
R1 — Original driver ROI.
R2 — Padded hand/phone ROI.
R3 — Face/hand ROI.
R4 — Torso/hand/face multi-region.
R5 — Scale-aware dynamic ROI.

Yêu cầu:

- Decode original frame ở độ phân giải nguồn khi quyền đọc cho phép.
- Không phóng to ảnh cache 64×64 rồi gọi là khôi phục chi tiết.
- Bảo tồn original-frame coordinates.
- Bảo tồn crop transform.
- Xử lý phone sát biên ROI.
- Xử lý phone che khuất bởi tay.
- Xử lý phone trên đùi.
- Xử lý mounted cabin camera.
- Lưu provenance và input hash.
- Không cắt vào protected subjects.

Chỉ số chẩn đoán:
- Phone pixel width/height.
- Valid crop coverage.
- Edge clipping.
- Candidate yield.
- Detector confidence distribution.
- Runtime and memory.

Không gọi candidate yield là detector AP.

Không gọi 12/13 reviewed boxes là 12 recovered events.

## 6. P2 — HOÀN THIỆN CAUSAL E3 INFERENCE

V12 đã xây dựng OnlineE3 feature extractor có batch parity.

Hãy kiểm tra code hiện có trước khi tạo module mới.

Ưu tiên tái sử dụng:
- causal_motion.py
- behavior_roi.py
- driver_phone_association.py
- Existing temporal/fusion adapters.

Nhiệm vụ:

A. Kiểm tra feature schema và class order.

B. Kiểm tra logic spatial + past-running normalization.

C. Kiểm tra state reset theo:
- Stream.
- Session.
- Camera.
- Track.
- Timestamp gap.
- Out-of-order input.

D. Kiểm tra missing-history handling.

E. Không sử dụng future frames trong online path.

F. Không nhầm uncalibrated scores với calibrated probabilities.

G. Thiết lập export/load contract cho fitted head/scaler.

H. Không tự tạo hệ số fitted head/scaler bằng dữ liệu giả để chạy Event Replay.

Nếu đã có artifact fitted hợp lệ:
- Verify provenance và class mapping.
- Load bằng định dạng an toàn.
- Reproduce frozen predictions.
- Kiểm tra numerical parity.

Nếu chưa có artifact:
- Giữ trạng thái BLOCKED_MISSING_FITTED_HEAD_SCALER.
- Viết test với synthetic numeric weights có nhãn rõ là fixtures.
- Chuẩn bị export path để sau khi quyền training hợp lệ có thể hoàn tất.
- Không sửa runtime active model.

## 7. P3 — ACTION REPRESENTATION RESEARCH

Nghiên cứu các phương pháp sau.

### DualScope

https://ojs.aaai.org/index.php/AAAI/article/view/37812

Ý tưởng:
- Critical spatial regions.
- Fine-grained motion.
- Temporal pyramid.
- Spatial/temporal attention.

### DriveCLIP

https://github.com/zahid-isu/DriveCLIP

Ý tưởng:
- Frozen vision-language features.
- SAFE vs PHONE vs OTHER.
- Transfer learning.
- Behavior semantics.

### MixStyle

https://arxiv.org/abs/2104.02008

Ý tưởng:
- Giảm ảnh hưởng môi trường/camera/background.
- Feature-level style perturbation.

### Group DRO

https://arxiv.org/abs/1911.08731

Ý tưởng:
- Worst-group robustness.
- Group-aware optimization.

Không khẳng định đã tái hiện paper nếu chỉ dùng một ý tưởng.

Ưu tiên thực nghiệm ít tốn GPU.

Các experiment:

A0 — Frozen E3 reference.
A1 — Original-resolution spatial evidence.
A2 — Causal E3 + ROI.
A3 — Motion + phone/hand features.
A4 — Lightweight multi-region fusion.
A5 — Frozen CLIP features nếu đủ điều kiện.
A6 — Group-aware training nếu đủ subject và data rights.

Không thực hiện fitting mới nếu live permission gate không cho phép.

Nếu có dữ liệu huấn luyện hợp lệ:
- Chốt split trước.
- Thử mỗi lần một thay đổi chính.
- Giữ baseline và seed.
- Đo pooled và worst-subject metrics.
- Kiểm tra calibration và class imbalance.

Dừng hoặc reject candidate nếu SAFE/PHONE Recall giảm mạnh dù pooled Macro-F1 tăng.

## 8. P4 — DRIVER-PHONE ASSOCIATION

Đọc và nâng cấp module hiện có.

Thiết lập các trạng thái:
- DRIVER_ASSOCIATED
- PASSENGER_ASSOCIATED
- MOUNTED_OR_STATIC
- UNKNOWN

Association phải dùng bằng chứng có provenance:
- Driver seat localization.
- Occupant track.
- Phone location.
- Hand proximity.
- Temporal consistency.
- Camera/seat context nếu được xác minh.

Không coi heuristic confidence là calibrated probability.

Không khẳng định occupant identity khi không có verified track.

Kiểm thử:
- Driver cầm điện thoại.
- Passenger cầm điện thoại.
- Mounted camera.
- Mounted phone.
- Người lái đưa tay lên tai nhưng không có điện thoại.
- Two occupants overlap.
- Ambiguous tracking.
- Missing view.
- Phone at ROI boundary.

Không biến UNKNOWN thành chắc chắn vi phạm hoặc chắc chắn an toàn.

Theo dõi coverage, abstention, false alarms và lost TP.

## 9. P5 — FUSION & EVENT DETECTION

Mục tiêu là biến các đặc trưng tốt hơn thành sự kiện tốt hơn.

Không xây dựng một event engine mới trùng chức năng nếu engine hiện có đáp ứng.

Đầu vào:
- Object detection.
- Action scores.
- Motion evidence.
- Phone/hand geometry.
- Driver association.
- Negative context.
- Timestamps.

Đầu ra:
- Event candidate.
- Stable event ID.
- Start/end.
- Real emission time.
- Source view.
- Evidence hashes.
- Raw and fused confidence.
- Rejection reasons.
- Uncertainty state.

Giữ nguyên frozen event matching policy.

Bắt buộc có:
- TP IDs.
- FP IDs.
- FN IDs.
- Recovered TP IDs.
- Lost TP IDs.
- New FP IDs.
- Removed FP IDs.

Đo:
- Event P/R/F1.
- FA/min.
- Recall với events <1s.
- Recall 1–3s.
- Recall 3–10s.
- Recall >10s.
- Fragmentation.
- Duplicate alarms.
- Onset latency.
- Emission latency.

Không dùng GT timestamps tại inference.

Không backdate emission time để tuyên bố latency thấp.

Nếu không có fitted E3 candidate, không được báo đã đo E3 Event F1.

Có thể chạy lại frozen context replay để kiểm tra metric reconciliation, nhưng không gọi đó là một thí nghiệm độ chính xác mới.

## 10. P6 — SEATBELT THREE-STATE IMPROVEMENT

Kiểm tra lại 500 pending seatbelt candidates.

Không tự chuyển PENDING thành APPROVED.

Giữ ba lớp:
- FASTENED.
- UNFASTENED.
- UNCERTAIN_OR_OCCLUDED.

Ưu tiên:
1. Human-reviewed labels.
2. Torso crop quality.
3. Belt visibility representation.
4. Hard negatives.
5. Class-aware calibration.
6. Lightweight attention.
7. Selective fine-tuning.

Nếu quyền fitting và nhãn chưa có:
- Chuẩn bị annotation/review workflow.
- Kiểm tra pipeline.
- Chạy synthetic tests.
- Báo BLOCKED.
- Không giả lập training improvements.

Nếu đủ điều kiện:
- Warm-start checkpoint cũ.
- Group-disjoint split.
- Ablation từng yếu tố.
- Per-class Precision/Recall/F1.
- Macro-F1.
- Balanced Accuracy.
- False-unfastened.
- Abstention/coverage.

Không coi belt bị che khuất là không thắt dây.

## 11. P7 — RUNTIME ENGINEERING

V12 E3 feature latency khoảng 3–5 ms trên cache, nhưng chưa phải full pipeline.

Hãy phân biệt:
- Decode time.
- Detector inference.
- ROI extraction.
- Encoder.
- Fitted head.
- Association.
- Fusion.
- Event engine.
- Database/evidence writes.

Đo trên cùng thiết bị và cùng test set.

Báo:
- Detector FPS.
- Full-pipeline processing FPS.
- p50/p95 latency.
- CPU RAM.
- GPU VRAM nếu có.
- Cold-start latency.
- End-to-end alert delay.

Không gọi detector-call FPS hoặc cached-feature FPS là full-system FPS.

## 12. P8 — SCIENTIFIC EXPERIMENT GATES

Mỗi experiment cần:
- ID.
- Hypothesis.
- Code SHA.
- Dataset hash.
- Checkpoint hash.
- Training rights status.
- Split definition.
- Seed.
- Changed factor.
- Baseline.
- Metrics.
- Runtime.
- Failure analysis.
- Decision.

Phân cấp:
LEVEL 1 — Diagnostics.
LEVEL 2 — Repeated development ablation.
LEVEL 3 — Newly acquired group-disjoint validation.
LEVEL 4 — Locked independent holdout.

Không dùng LEVEL 1/2 để tuyên bố generalization đạt 85%.

Không quét confidence/threshold lặp vô hạn trên ba subjects cũ.

Không sử dụng gZ-37 và gE-28 để chọn candidate.

Nếu chưa có approved independent test:
INDEPENDENT_AI85_NOT_ESTABLISHED

## 13. P9 — TESTS VÀ SOFTWARE QUALITY

Chạy:
- Python compile.
- Ruff.
- Targeted Pytest.
- Full Pytest.
- Frontend build.
- Backend smoke.
- V12 artifact reconciliation.
- Synthetic model-loading tests.
- Data authorization gates.
- Protected holdout guards.
- ROI coordinate tests.
- Batch/online parity tests.
- Session-state reset tests.
- Occupant association tests.
- Temporal event reconciliation.

Không thay đổi test để chỉ báo PASS.

So sánh với baseline báo cáo V12: 618 tests PASS.

Nếu có tests mới và tổng tăng/giảm, báo số liệu thật.

Nếu một smoke test không chạy, ghi NOT RUN hoặc BLOCKED, không ghi PASS.

## 14. REQUIRED OUTPUTS

Tạo:
reports/v13_research/

Bắt buộc có:

01_HEAD_AND_ENVIRONMENT_AUDIT.md
02_DATA_RIGHTS_AND_SPLIT_AUDIT.md
03_BASELINE_RECONCILIATION.md
04_PHONE_HIGH_RES_ROI.md
05_ONLINE_E3_ARTIFACT_READINESS.md
06_BEHAVIOR_REPRESENTATION_RESULTS.md
07_DRIVER_PHONE_ASSOCIATION.md
08_FUSION_AND_EVENT_REPLAY.md
09_SEATBELT_RESULTS.md
10_RUNTIME_BENCHMARK.md
11_TEST_RESULTS.md
12_FINAL_AI_LEAD_REPORT.md

Cùng các file machine-readable:
- experiment_registry.json
- metrics_comparison.json
- dataset_manifest.json
- model_manifest.json
- candidate_readiness.json
- event_match_records.json khi có event replay thực
- reproduction commands

Không tạo file báo cáo rỗng.

Không sửa các báo cáo cũ để làm mất lịch sử thực nghiệm thất bại.

## 15. THỰC THI NGAY VÀ TIÊU CHÍ HOÀN THÀNH

Thực hiện theo thứ tự:

1. Xác minh HEAD, Git và permissions.
2. Reconcile baseline V12.
3. Kiểm tra và hoàn thiện original-resolution ROI pipeline.
4. Xác minh OnlineE3 feature path.
5. Kiểm tra fitted head/scaler availability.
6. Hoàn thiện driver association và unknown handling.
7. Triển khai candidate inference adapter opt-in.
8. Chạy synthetic regression tests.
9. Chạy những thí nghiệm thực tế được quyền thực hiện.
10. Chạy event-level replay nếu có candidate và dữ liệu hợp lệ.
11. Kiểm tra seatbelt review/training readiness.
12. Chạy full tests, build, smoke và benchmarks khả thi.
13. Tổng hợp report có đầy đủ bằng chứng.
14. Đưa ra quyết định KEEP/REJECT/BLOCKED cho từng candidate.

Nếu thiếu quyền, dữ liệu hoặc artifact:
- Nêu blocker chính xác.
- Không tự lách quyền bằng cách đổi tên hoạt động training thành diagnostics.
- Không tự tạo model fitted giả.
- Không hủy dữ liệu.
- Chuyển sang hạng mục độc lập có thể hoàn thành và kiểm chứng.

KẾT LUẬN BẮT BUỘC TRONG FINAL REPORT:

A. Những source files thực sự được thay đổi.

B. Những thí nghiệm thực sự đã chạy.

C. Kết quả mới so với baseline V12.

D. Phone Event TP/FP/FN/P/R/F1/FA-min.

E. SAFE/PHONE/OTHER clip metrics.

F. Worst-subject metrics.

G. Seatbelt metrics.

H. FPS/latency được đo đúng phạm vi.

I. Kết quả test.

J. Các dataset/license/label blockers.

K. Candidate nào có fitted artifacts, candidate nào chưa.

L. Independent evaluation đã tồn tại chưa.

M. Có đạt mục tiêu 85% một cách khoa học chưa.

N. Có đủ điều kiện production approval không.

O. Một thí nghiệm tiếp theo có giá trị cao nhất.

Quy tắc cuối cùng:

**KHÔNG ĐƯỢC TẠO CHỈ SỐ 85% BẰNG CÁCH THAY ĐỔI GROUND TRUTH, EVENT MATCHER, TEST DATA, THRESHOLD TRÊN HOLDOUT, MẪU SỐ HOẶC PHẠM VI ĐÁNH GIÁ.**

Mục tiêu là tạo ra một Roadwatch có mô hình thật sự tốt hơn, ít bỏ sót hành vi hơn, ít cảnh báo sai hơn, hoạt động ổn định trên người lái mới và có bằng chứng bảo vệ đồ án tốt nghiệp.

BẮT ĐẦU TRIỂN KHAI, KHÔNG CHỈ VIẾT KẾ HOẠCH.