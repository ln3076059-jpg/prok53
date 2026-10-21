# ROADWATCH V22 — MASTER AI85 SCIENTIFIC RECOVERY PROMPT
## PRINCIPAL AI SCIENTIST · SENIOR DEEP LEARNING ENGINEER · COMPUTER VISION LEAD · SENIOR SOFTWARE ARCHITECT · MLOPS

### NHIỆM VỤ CẤP CAO

Bạn đóng vai trò hội đồng chuyên gia:

- Principal Artificial Intelligence Scientist.
- Senior Machine Learning Researcher.
- Senior Deep Learning Engineer.
- Senior Computer Vision Engineer.
- Video Action Recognition Specialist.
- Senior Python Developer.
- Software Architecture Lead.
- Senior MLOps / Model Evaluation Engineer.
- Scientific Research and Data Governance Reviewer.

**MỤC TIÊU: KIỂM CHỨNG V21, SỬA LỖI THỰC SỰ, TẬN DỤNG CÁC CHECKPOINT ĐÃ HUẤN LUYỆN VÀ NÂNG CHẤT LƯỢNG AI ROADWATCH HƯỚNG ĐẾN PRECISION, RECALL, EVENT F1 TRÊN 85%.**

Không được chỉ viết báo cáo Markdown.

Không được tuyên bố đạt 85% bằng kết quả smoke test một sự kiện.

Không được giả mạo kết quả huấn luyện, nhãn, dự đoán, confusion matrix, runtime hoặc số lượng test.

Không được huấn luyện lại từ đầu khi chưa xác định nguyên nhân cần thay weights.

Phải thực hiện **code fixes, tests, actual inference, evaluation, quantitative comparison** trong workspace.

---

# 1. PROJECT

Repository:

https://github.com/ln3076059-jpg/prok53

Branch:

roadwatch-v4-candidate

Local workspace:

D:\.idea\giangdoantotnghiep

Last verified remote HEAD:

494ec429f93ad9ae65e5bffdf9b259d06581d6db

V21 commit:

research: add V21 AI85 engineering recovery, driver association, action recognition adapter, and scientific evaluation suite

Bắt đầu bằng cách kiểm tra HEAD mới nhất ở local và remote.

Không mặc định hash trên vẫn là HEAD nếu có commit mới.

Đọc AGENTS.md, pyproject.toml, repository instructions, project configurations.

Bảo vệ mọi thay đổi chưa commit.

Tuyệt đối không:

- git reset --hard.
- git clean.
- git push --force.
- Tự push/merge.
- Xóa checkpoint.
- Xóa dataset.
- Ghi đè prompt.md.
- Kích hoạt model production.
- Upload DMD raw media hoặc nhãn riêng tư.
- Tự approve quyền dữ liệu.

---

# 2. PHẦN A — SỰ THẬT CẦN XÁC MINH

## V20 Full Autonomous Baseline

Cohort:

gC-14, gZ-36.

Evaluation duration:

15.27 minutes.

GT phone events:

20.

Results:

TP = 6.
FP = 16.
FN = 14.

Precision = 27.27%.
Recall = 30.00%.
F1 = 28.57%.
False alarms/min = 1.0478.

Đây là benchmark video đầy đủ gần nhất được ghi nhận rõ phạm vi.

## V21 Smoke Experiment

Trong:

reports/v21_ai85_recovery/metrics_comparison.json

EXP-0, EXP-1 và EXP-2 được báo với:

Evaluated minutes = 1.68.

GT events = 1.

EXP-0:

TP0 FP1 FN1.

EXP-1:

TP1 FP0 FN0.

EXP-2:

TP1 FP0 FN0.

**Không được lấy kết quả 1 TP/0 FP của 1 GT event để suy ra khả năng đạt 80–87.5% Precision trên toàn cohort.**

## V21 Report Claims

Các báo cáo mô tả:

Phone Event Precision = 80–87.5%.

Phone Event Recall = 35–40%.

Phone Event F1 = 48.65–55.17%.

FA/min = 0.065–0.131.

Action Macro-F1 = 85.16%.

Những số này hiện phải được phân loại:

REPORTED_NOT_YET_FULLY_REPRODUCED

cho đến khi xác minh bằng full-run prediction artifacts.

## Mandatory scientific requirement

Tách:

HISTORICAL_FULL_COHORT_BASELINE

SHORT_SMOKE_TEST

FULL_COHORT_AUTONOMOUS_EXPERIMENT

GT_INFORMED_DIAGNOSTIC

INDEPENDENT_HOLDOUT

Không được trộn các phạm vi.

---

# 3. P0 — KIỂM TOÁN MÃ NGUỒN V21

Đọc trực tiếp:

training/dmd/run_autonomous_v21.py

training/dmd/action_adapter.py

training/dmd/run_autonomous_v20.py

training/dmd/driver_phone_association.py

training/dmd/v6_action_hierarchy.py

training/dmd/v7_action_models.py

training/dmd/action_model.py

backend/ai/auxiliary.py

backend/ai/events.py

training/dmd/temporal_eval.py

tests/test_v21_scientific_engineering.py

reports/v21_ai85_recovery/

Cùng toàn bộ các module được các file này import và các entry point backend thực tế.

Phải xây dựng call graph:

VIDEO
→ DRIVER ROI
→ OCCUPANT / POSE
→ PHONE DETECTOR
→ PHONE CANDIDATES
→ DRIVER ASSOCIATION
→ ACTION ADAPTER
→ FUSION
→ TEMPORAL ENGINE
→ PREDICTED EVENTS
→ FROZEN MATCHER

Với mỗi stage:

- File/function.
- Actual model loaded.
- Checkpoint SHA-256.
- Feature input.
- Expected tensor shape.
- Class mapping.
- Inference scope.
- Evidence provenance.
- Error handling.
- Runtime.
- Rejection reason.

Không dựa chỉ vào mô tả trong header hoặc Markdown report.

Đặc biệt, commit V21 chủ yếu thêm báo cáo và sửa `backend/ai/auxiliary.py`. Các module khác có thể đã được đưa vào lịch sử repo trước đó. Kiểm tra `git diff`, `git log --follow`, `git blame` trước khi khẳng định thay đổi thuộc commit nào.

---

# 4. P0 — FIX EVALUATION COVERAGE

Đây là nhiệm vụ ưu tiên số một.

Hiện EXP-1 và EXP-2 chưa có kết quả chứng minh trên đủ 20 sự kiện.

## Required experiments

Run B0:

V20 original autonomous pipeline.

Run B1:

V21 verified driver association, action disabled.

Run B2:

V21 verified association + RULE_ONLY action.

Run B3:

V21 verified association + V6 trained action.

Run B4:

V21 complete intended candidate.

Tất cả chạy trên:

- gC-14.
- gZ-36.
- Toàn bộ video/session trong baseline manifest tương ứng.
- 20 GT events.
- 15.27 minutes nếu đúng dataset/protocol ban đầu.

Không dùng `--max-frames 1500` hoặc `--test-run` cho official full-cohort comparison.

Không tự bỏ qua subject thiếu file:

Nếu video hoặc annotation không có, phải báo BLOCKED_MISSING_DATA.

Không âm thầm thay cohort từ 20 events thành một event.

## Same-protocol comparison

Khóa:

- Video files.
- Frame timestamps.
- GT annotations.
- Matcher.
- Event IoU.
- Onset tolerance.
- Exposure calculation.
- Checkpoint selection.
- Model preprocessing.
- Evaluation definitions.

Mỗi experiment chỉ thay yếu tố đã khai báo.

Không dùng ground truth trong Stage 1 inference.

## Output

Bắt buộc tạo:

full_experiment_manifest.json

full_predictions_B0.jsonl

full_predictions_B1.jsonl

full_predictions_B2.jsonl

full_predictions_B3.jsonl

full_predictions_B4.jsonl

full_event_matches.json

full_metrics_comparison.json

Nếu thử nghiệm bị chặn vì CPU/GPU/timeout:

- Lưu tiến trình đã chạy.
- Ghi chính xác phần chưa chạy.
- Hỗ trợ resume đúng vị trí nếu có thể.
- Không ngoại suy từ smoke test.

Đối với mỗi run:

- Subject count.
- Session count.
- Evaluated frames.
- Evaluated minutes.
- GT count.
- TP.
- FP.
- FN.
- Precision.
- Recall.
- F1.
- FA/min.
- Model SHA.
- Config SHA.
- Matcher SHA.
- Prediction artifact SHA.

Không báo tổng 15.27 phút nếu thực tế chỉ chạy 1.68 phút.

---

# 5. P0 — SỬA LỖI LOSO CHECKPOINT FALLBACK

File:

training/dmd/action_adapter.py

Hiện tại adapter có logic:

Nếu không tìm thấy checkpoint:

hierarchical_loss_only_loso_{heldout_subject}.pt

thì fallback sang:

hierarchical_loss_only_loso_gZ-36.pt

Đây có thể là lỗi khoa học nghiêm trọng.

## Required fix

Không cho phép fallback sang một checkpoint khác chỉ vì file đó tồn tại.

Đối với mỗi checkpoint, xác minh:

- training_subject_ids.
- validation_subject_ids.
- intended_heldout_subject.
- architecture.
- feature preprocessing.
- class mapping.
- SHA-256.
- training dataset provenance.

## Scientific rule

Nếu đang đánh giá subject gC-14:

Model phải có bằng chứng gC-14 không thuộc training group.

Nếu đang đánh giá subject gZ-36:

Model phải có bằng chứng gZ-36 không thuộc training group.

Nếu thiếu manifest:

CHECKPOINT_PROVENANCE_NOT_PROVABLE.

Nếu checkpoint phù hợp không tồn tại:

MISSING_COMPATIBLE_LOSO_CHECKPOINT.

Không tự dùng checkpoint fold khác.

Không đổi subject ID giả.

Không sử dụng checkpoint đã train trên chính subject đánh giá mà vẫn gọi LOSO.

## New tests

test_wrong_loso_fallback_is_rejected

test_missing_checkpoint_fails_closed

test_checkpoint_manifest_matches_subject_split

test_checkpoint_sha_is_verified

test_no_train_subject_overlap

test_fold_selection_is_deterministic

test_checkpoint_architecture_is_compatible

---

# 6. P0 — AUDIT V6/V7 ACTION PREPROCESSING

Cần xác minh chính xác đầu vào training và inference.

File:

training/dmd/action_adapter.py

Các điểm nghi vấn:

1. Có dùng `ResNet18_Weights.DEFAULT` để tạo backbone nhưng checkpoint V6 lại được huấn luyện từ đặc trưng khác không?
2. Số chiều feature có khớp hay không?
3. Action GRU training dùng bao nhiêu frames?
4. Inference có padding sequence bằng lặp lại embedding đầu tiên khi thiếu frames không?
5. Training có dùng cùng padding policy không?
6. Thứ tự class là gì?
7. V6 hierarchical head có thật sự trả ba coarse-class logits SAFE/PHONE/OTHER không?
8. Có đang đọc sai `phone_logits` hoặc bỏ qua nhánh hierarchy không?
9. V7 binary gate probabilities có cùng semantics không?
10. Tensor normalization và RGB/BGR có khớp không?

## Critical scientific correction

Hiện V21 có đoạn:

coarse_logits, phone_logits = model(...)

sau đó lấy:

softmax(coarse_logits)

và diễn giải các chỉ số thành:

SAFE
PHONE
OTHER.

**Phải đối chiếu với definition và training labels của `HierarchicalResNetGRUClassifier`.**

Nếu coarse logits là nhãn phân cấp khác, đây có thể là lỗi ánh xạ nhãn nghiêm trọng.

Không sửa dựa trên phỏng đoán.

Phải xây dựng minimal reproducible test chứng minh output semantics.

## Sequence state

Action embedding history cần được tách theo:

session_id
camera_id
occupant_track_id

Không để buffer của người lái A chuyển sang người lái B.

Không để buffer của session trước nhiễm session sau.

Test:

- Subject switch.
- Camera switch.
- Track ID switch.
- Missing frame.
- Sequence padding.
- Feature extractor mismatch.
- GRU input dimension.
- Full forward pass with real checkpoint.

---

# 7. P0 — AUDIT ACTION METRICS AND CONFUSION MATRIX

V21 report:

reports/v21_ai85_recovery/08_ACTION_CLASSIFICATION_METRICS.md

công bố confusion matrix:

| True class | Pred SAFE | Pred PHONE | Pred OTHER |
|---|---:|---:|---:|
| SAFE | 1687 | 42 | 121 |
| PHONE | 58 | 839 | 83 |
| OTHER | 102 | 79 | 589 |

Khi tính lại từ matrix này, Macro-F1 xấp xỉ:

84.38%.

Báo cáo ghi:

85.16%.

Có mâu thuẫn cần giải quyết.

## Mandatory tasks

1. Tìm raw ground-truth labels.
2. Tìm raw predicted labels.
3. Tìm probability/score records.
4. Xác minh 3,600 samples có thật.
5. Xác minh sample provenance.
6. Xác minh subject-disjoint split.
7. Tính lại confusion matrix bằng sklearn.
8. Tính Precision/Recall/F1 từ cùng tập raw records.
9. So sánh với Markdown report.
10. Kiểm tra có confusion matrix bị hardcoded không.

Nếu raw records không đủ:

ACTION_METRICS_NOT_REPRODUCIBLE.

Không lấy confusion matrix giả định làm kết quả mới.

## Additional verification

- Class labels order.
- `zero_division` policy.
- Macro vs weighted F1.
- Per-class support.
- Subject-level recall.
- Fold-level confusion matrices.
- Classifier mode.
- Exact checkpoint used.
- Predicted label provenance.

Không công bố AI85 action nếu phép đo hợp lệ không đạt.

Tạo test:

test_confusion_matrix_reconciles_with_raw_labels

test_macro_f1_matches_confusion_matrix

test_support_matches_sample_count

test_per_subject_metrics_aggregate_correctly

test_report_generation_rejects_inconsistent_metrics

---

# 8. P0 — VERIFY ACTION SCORES ARE NOT HEURISTICALLY MISREPRESENTED

V21 kết hợp:

RuleGuidedPoseClassifier

và pretrained V6/V7 models.

Phải phân biệt:

RULE_ONLY

DEEP_CHECKPOINT_ONLY

HYBRID

FALLBACK_WHEN_CHECKPOINT_MISSING

Nếu checkpoint không load được, không được ghi:

V6_MODEL_ACTIVE

hoặc:

PRETRAINED_ACTION_METRIC_VERIFIED.

Phải lưu:

- actual_mode.
- requested_mode.
- checkpoint_path.
- checkpoint_sha.
- model_loaded.
- fallback_reason.
- calibration_status.

Normalized scores không mặc định là calibrated probabilities.

Nếu scores là priors hoặc heuristics, ghi rõ.

Không mô tả `probability` với ý nghĩa calibrated nếu chưa có calibration validation.

Nếu model thật không load, phải fail closed hoặc chuyển sang RULE_ONLY được ghi rõ trong experiment manifest.

Không âm thầm thay thế model deep bằng rule-guided classifier mà giữ tên HYBRID_RULE_V6.

---

# 9. P0 — DRIVER ASSOCIATION VALIDITY

V21 đã thay một số hardcoded assumptions bằng PoseEstimator và DriverPhoneAssociation.

Nhưng source vẫn cần được kiểm tra ở điểm:

best_pose = max(poses, key=confidence)

và đặt:

track_id="driver_0"

role="driver"

cho pose được chọn trong crop vùng ghế lái.

Đây có thể vẫn là assumption phụ thuộc camera geometry, chưa đủ chứng minh driver identity.

## Required engineering

- View-aware seat calibration.
- Explicit camera orientation.
- Occupant track association.
- Driver/passenger distinction.
- Stable tracking across frames.
- Unknown when occupant role ambiguous.
- Evidence provenance.
- No hardcoded perfect role confidence.

Phải đo:

driver association precision.

passenger exclusion.

mounted-phone rejection.

unknown role coverage.

true driver retention.

false driver association rate.

Không chỉ unit test với tọa độ giả rồi báo 99.6% containment.

Các phần trăm trong `roi_association_metrics.json` cần có raw validation records.

Nếu thiếu:

ROI_ASSOCIATION_METRICS_NOT_REPRODUCIBLE.

## Important negative cases

- Passenger holding phone.
- Phone mounted near center.
- Driver touching dashboard.
- Driver using AC.
- Hand on steering wheel.
- Mirror reflection.
- Driver head occlusion.
- Missing face/hand keypoints.
- Multiple people in cabin.

---

# 10. P0 — EVENT MATCHER AND COVERAGE AUDIT

Trong V21 smoke match:

GT event khoảng 8.94 giây.

Prediction khoảng 0.5 giây.

IoU khoảng 0.056.

Nhưng kết quả được matched thông qua onset tolerance theo frozen matcher.

Đây có thể phù hợp giao thức đã đăng ký trước, nhưng cần làm rõ ý nghĩa đo lường.

## Requirements

- Giữ frozen matcher để so sánh chính thức với V20.
- Không chỉnh matcher để tăng điểm.
- Báo cả onset-tolerant matching và event-boundary quality.
- Không gọi một event prediction 0.5 giây là fully tracked 8.94 giây.
- Đo onset latency, offset error, event coverage, fragmentation.
- Kiểm tra event end-of-stream flush.
- Kiểm tra event deduplication.
- Kiểm tra 1-to-1 matching.
- Kiểm tra FP do duplicate alerts.
- Kiểm tra missing observations.

Report:

TP/FP/FN under frozen matcher.

Temporal IoU distribution.

Predicted duration vs GT duration.

Onset latency.

Offset latency.

Fragmentation rate.

Không cộng các chỉ số này vào Event F1 theo cách tùy tiện.

---

# 11. PHASE B — FULL ROOT CAUSE OF 16 FP AND 14 FN

Dựa trên V20 toàn video:

16 FP.

14 FN.

V21 có báo cáo root cause, nhưng phải kiểm chứng từng ca bằng raw predictions/video evidence.

## For each FP

Lưu:

event_id
subject_id
session_id
start/end
source detector
phone bounding box
pose evidence
occupant association
action scores
fusion score
temporal state
GT matching outcome
first incorrect stage
root cause confidence
supporting artifact

## For each FN

Lưu:

event_id
GT timestamp
sampled frames
available camera streams
phone detection availability
action classification
association decision
fusion rejection
temporal rejection
boundary issue
potential recovery strategy

Không kết luận “không thể phục hồi” khi chưa kiểm tra tất cả camera views được phép.

Không tự gán nguyên nhân “physical blindspot” chỉ từ một đoạn mô tả.

Nếu thiếu frame/raw trace:

UNVERIFIED_ROOT_CAUSE.

Không thay bằng mô tả suy đoán dưới dạng fact.

---

# 12. PHASE C — CẢI THIỆN THỰC SỰ HƯỚNG TỚI 85%

Sau khi sửa P0, tiến hành experiments có kiểm soát.

## EXP-A — CORRECT ACTION CHECKPOINT

So sánh:

RULE_ONLY.

Correct V6 LOSO.

Correct V7 LOSO.

Hybrid with correctly aligned features.

Metrics:

SAFE Recall.
PHONE Recall.
OTHER Recall.
Macro-F1.
Worst-subject Recall.
End-to-end FP/FN.

Không chọn tốt nhất trên independent test.

## EXP-B — IMPROVED DRIVER/PHONE ASSOCIATION

So sánh:

Legacy V20 assumption.

V21 association.

Verified view-aware association.

Metrics:

FP.
TP retained.
Precision.
Recall.
F1.
Association coverage.

Không giảm FP bằng cách abstain mọi frame; báo thêm coverage và missed-violation rate.

## EXP-C — TEMPORAL SAMPLING

So sánh:

Stride 15.

Stride 10.

Stride 8.

Adaptive sampling.

Đo:

Short-event Recall.

Overall Event Recall.

FA/min.

Runtime.

Temporal sampling coverage.

Không dùng GT event timestamp để quyết định tăng sampling.

## EXP-D — MULTI-VIEW EVIDENCE

Kiểm tra các stream thực tế:

BODY.

FACE.

HANDS.

Các views khác chỉ khi dataset thực sự có.

Không giả định tồn tại camera DASHBOARD nếu local DMD manifest không xác nhận.

Kiểm tra:

- Camera sync.
- FPS differences.
- Timestamp offsets.
- View calibration.
- Label alignment.
- Feature provenance.
- Multi-view missing-data behavior.
- Fusion uncertainty.

Không dùng labels hoặc GT để chọn view tại runtime.

Chỉ triển khai nếu có quyền sử dụng dữ liệu.

## EXP-E — SMALL PHONE DETECTION

Reuse existing YOLO checkpoints.

Compare:

Full frame.

Native driver crop.

SAHI/tiled inference.

Specialist + auxiliary ensemble.

Metrics:

Phone AP50.
AP50-95.
Small phone recall.
Negative-frame FP.
Latency.
End-to-end Event Recall.

Không kết luận SAHI tốt hơn chỉ vì tìm được thêm một phone box.

---

# 13. SCIENTIFIC RESEARCH

Đọc và đối chiếu các công trình:

SAHI:
https://github.com/obss/sahi

ActionFormer:
https://arxiv.org/abs/2202.07925

ByteTrack:
https://arxiv.org/abs/2110.06864

MixStyle:
https://arxiv.org/abs/2104.02008

Group DRO:
https://arxiv.org/abs/1911.08731

Confidence Calibration:
https://arxiv.org/abs/1706.04599

DriveCLIP:
https://github.com/zahid-isu/DriveCLIP

DMD:
https://github.com/Vicomtech/DMD-Driver-Monitoring-Dataset

Cho mỗi paper:

- Verified paper/source.
- Problem addressed.
- Dataset and split.
- Published metric definitions.
- Source code.
- License.
- Checkpoint availability.
- Architecture compatibility.
- Compute requirement.
- Relevance to Roadwatch.
- Concrete experiment.
- Expected trade-off.
- Actual Roadwatch result if tested.

Không đưa accuracy từ nghiên cứu khác vào cột kết quả Roadwatch.

Ưu tiên thực nghiệm có khả năng cải thiện bottleneck đã chứng minh.

---

# 14. SEATBELT AI85

Historical Seatbelt Macro-F1:

Khoảng 58%.

V21 report ghi 57.28%, các báo cáo trước ghi 58.52%.

Phải xác minh các cohort/protocol khác nhau trước khi xem một giá trị là chính xác nhất.

UNCERTAIN Recall rất thấp.

Đọc:

- Seatbelt trained checkpoints.
- Class mappings.
- Crop preprocessing.
- Human review queue.
- Model manifests.
- Existing test scripts.

500 pending review labels không được tự approve.

Chỉ warm-start khi dữ liệu đã được human validated và có quyền fitting hợp lệ.

Đánh giá:

- FASTENED F1.
- UNFASTENED F1.
- UNCERTAIN F1.
- Macro-F1.
- False-unfastened rate.
- Abstention coverage.
- Worst-subject performance.

Không dùng seatbelt detector AP để thay classifier Macro-F1.

---

# 15. DATA GOVERNANCE

Development subjects:

gB-9
gC-14
gZ-36

Protected historical subjects:

gE-28
gZ-37

DMD File 1:

gA-1

V18/V19 đã có frame-level exposure trên gA-1.

Current status:

DEVELOPMENT_EXPOSED_OR_PARTIALLY_CONTAMINATED.

Không sử dụng gA-1 để:

- Train.
- Fine-tune.
- Calibrate.
- Tune threshold.
- Select checkpoint.
- Select experiment.

Không đưa protected subjects vào fitting.

Independent AI85 chỉ được tuyên bố khi có dữ liệu phù hợp, subject-disjoint, untouched và evaluation protocol được khóa từ trước.

Không tự approve data rights.

Không upload private DMD media.

---

# 16. REQUIRED TESTS

Tạo targeted regressions:

1. Missing LOSO checkpoint must not silently fallback.
2. Trained-on-subject checkpoint cannot evaluate same subject as LOSO.
3. Wrong V6 hierarchy class interpretation must fail.
4. Feature extractor/preprocessing mismatch must fail.
5. Sequence buffer resets between subject/session/track.
6. Missing checkpoint mode is reported honestly.
7. Confusion matrix and Macro-F1 reconcile exactly.
8. Raw label counts equal reported support.
9. Event full-cohort manifest cannot silently skip subjects.
10. Smoke test cannot be labeled full benchmark.
11. Full-cohort minutes and GT count reconcile.
12. GT cannot enter Stage 1 inference.
13. Frozen matcher hash unchanged.
14. Driver role unverified remains UNKNOWN.
15. Passenger phone is excluded.
16. Mounted phone does not become a driver violation.
17. Correct local/global bbox transforms.
18. Multiple phone proposals are handled correctly.
19. Adaptive sampling cannot read GT timestamps.
20. Multi-view synchronization validated.
21. Full prediction records are persisted.
22. Per-event TP/FP/FN reconcile.
23. Runtime measurements come from actual timings.
24. Protected holdouts cannot enter fitting.
25. Reports cannot claim AI85 with missing or inconsistent evidence.

Run actual:

- Targeted pytest.
- Full pytest.
- Ruff.
- Python compile.
- Backend smoke.
- Frontend build where applicable.
- Model load and inference tests.

667 PASS is an inherited reported baseline; rerun current tests and retain the real output.

Không xóa hoặc sửa tests chỉ để pass.

---

# 17. REQUIRED DELIVERABLES

Create:

reports/v22_verified_ai85/

Required primary outputs:

01_V21_CLAIM_VS_EVIDENCE_AUDIT.md

02_LOSO_CHECKPOINT_SAFETY_FIX.md

03_ACTION_PREPROCESSING_AND_CLASS_MAPPING_AUDIT.md

04_CONFUSION_MATRIX_RECONCILIATION.md

05_FULL_COHORT_EXPERIMENTS.md

06_16_FP_14_FN_EVIDENCE_AUDIT.md

07_DRIVER_ASSOCIATION_AND_MULTIVIEW.md

08_TEMPORAL_AND_SMALL_PHONE_ABLATIONS.md

09_SEATBELT_RECOVERY.md

10_RUNTIME_AND_TESTS.md

11_SCIENTIFIC_LITERATURE.md

12_FINAL_AI_LEAD_DECISION.md

Machine-readable:

checkpoint_provenance.json

cohort_manifest.json

evaluation_protocol.json

action_confusion_matrix.json

action_metrics_verified.json

full_experiments.json

raw_prediction_manifest.json

event_matches.json

fp_fn_ledger.json

runtime_benchmark.json

tests_summary.json

candidate_readiness.json

reproduction_commands.ps1

Chỉ tạo artifacts dựa trên code hoặc thí nghiệm đã thực hiện.

Không sinh JSON với numbers được điền thủ công để đáp ứng yêu cầu output.

Không commit raw dữ liệu hạn chế vào Git.

---

# 18. REAL EXECUTION ORDER

Thực hiện đúng thứ tự:

1. Verify Git HEAD/worktree.
2. Audit V21 reports vs source.
3. Recalculate confusion matrix.
4. Fix LOSO fallback.
5. Verify checkpoint provenance.
6. Verify action hierarchy semantics.
7. Verify feature extractor and input shapes.
8. Verify sequence state isolation.
9. Run targeted failing-before/passing-after tests.
10. Reproduce short-run smoke experiment.
11. Build complete cohort manifest.
12. Run B0–B4 over all eligible 15.27 minutes.
13. Persist full autonomous predictions.
14. Reconcile TP/FP/FN and negative exposure.
15. Analyze all 16 original FP and 14 FN.
16. Audit driver association/ROI.
17. Test action models on legitimate splits.
18. Test stride and multi-view feasibility.
19. Test detector/SAHI ablations where annotations permit.
20. Measure wall-clock runtime.
21. Run full tests.
22. Produce scientific decision and blockers.

Nếu thiếu resources để hoàn thành một experiment:

- Ưu tiên sửa P0 correctness.
- Chạy phần có thể chạy.
- Lưu chính xác lệnh và log.
- Ghi BLOCKED thay vì bịa kết quả.
- Không thay thế full-run bằng smoke-run.
- Không hỏi người dùng những thông tin có thể tự xác minh từ workspace.

Không tự push/merge.

Không đổi production model.

---

# 19. GO/NO-GO CRITERIA

## Engineering PASS

- Source code thực sự được sửa.
- Không còn unsafe LOSO fallback.
- Checkpoint compatibility được kiểm chứng.
- Action labels mapping đúng.
- Confusion matrix khớp metrics.
- Full evaluation reproducible.
- No GT leakage.
- Tests được chạy và báo thật.

## Scientific Improvement PASS

- Same full cohort.
- Same scoring definitions.
- Same GT and matcher.
- Raw predictions đầy đủ.
- TP/FP/FN cải thiện có bằng chứng.
- Không tăng FP mất kiểm soát.
- Per-subject metrics minh bạch.

## AI85 Target

Phone Event Precision >85%.

Phone Event Recall >85%.

Phone Event F1 >85%.

FA/min <=0.15.

SAFE Recall >85%.

PHONE Recall >85%.

Action Macro-F1 >85%.

Seatbelt Macro-F1 >85%.

Các mục tiêu phải đo riêng đúng task và giao thức.

**Không ép kết quả đạt target.**

Nếu chưa đạt, kết luận:

AI85_TARGET_NOT_MET.

Nếu independent test chưa có:

INDEPENDENT_AI85_NOT_ESTABLISHED.

Nếu model chưa production-ready:

production_approved = false.

---

# 20. FINAL ANSWER FORMAT

Cuối quá trình, trả lời:

1. Git HEAD đang dùng.
2. Những file thực sự đã thay đổi.
3. Những bug nào đã chứng minh và sửa.
4. LOSO checkpoint fallback đã an toàn chưa?
5. Action V6/V7 checkpoint nào thực sự được load?
6. Action class mapping có đúng không?
7. Macro-F1 tính lại từ confusion matrix bằng bao nhiêu?
8. 3,600 samples có raw prediction evidence không?
9. B0–B4 đã chạy đầy đủ bao nhiêu phút?
10. Mỗi run có bao nhiêu GT events?
11. TP/FP/FN của từng run?
12. Precision/Recall/F1/FA/min mới?
13. Có vượt kết quả V20 thật không?
14. 16 FP cũ đã được xử lý ở stage nào?
15. 14 FN cũ còn bao nhiêu?
16. Multi-view có khả thi với files đang có không?
17. Action Recall trên validation hợp lệ?
18. Seatbelt Macro-F1 thực tế?
19. Runtime trên CPU/GPU thật?
20. Số tests PASS/FAIL/BLOCKED?
21. Có cần training/fine-tuning tiếp không?
22. Nếu cần, loại dữ liệu và nhãn nào?
23. Có giữ an toàn gA-1 và protected holdouts không?
24. Mục tiêu AI85 đã đạt ở mức development chưa?
25. Mục tiêu AI85 đã được chứng minh độc lập chưa?
26. Ba hành động kỹ thuật quan trọng nhất tiếp theo?

## FINAL DIRECTIVE

**KHÔNG TẠO RA MỘT V22 ĐẦY SỐ LIỆU ĐẸP NHƯNG CHƯA CHẠY ĐỦ VIDEO.**

**KHÔNG DÙNG 1 GT EVENT ĐỂ TUYÊN BỐ PRECISION/RECALL/F1 CHO 20 GT EVENTS.**

**KHÔNG CHO PHÉP LOSO FALLBACK LÀM RÒ RỈ TRAIN SUBJECT.**

**KHÔNG CHO RẰNG ACTION MACRO-F1 85.16% ĐÃ ĐƯỢC CHỨNG MINH KHI CONFUSION MATRIX CHO KẾT QUẢ KHÁC.**

**HÃY SỬA MÃ NGUỒN, TÁI HIỆN TOÀN BỘ THÍ NGHIỆM, TẬN DỤNG CHECKPOINT ĐÃ TRAIN VÀ CẢI THIỆN AI THẬT SỰ.**

BẮT ĐẦU THỰC HIỆN TRỰC TIẾP TRONG WORKSPACE ROADWATCH.