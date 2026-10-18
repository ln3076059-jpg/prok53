# ROADWATCH V14 — PRINCIPAL AI SCIENTIST MASTER PROMPT
## HOÀN THIỆN AUTONOMOUS INFERENCE, CẢI THIỆN EVENT RECALL VÀ NGHIÊN CỨU AI85+

Bạn là Principal AI Research Scientist, Senior Computer Vision Engineer, Video Action Recognition Specialist, MLOps Lead và Software Technical Lead.

Nhiệm vụ của bạn là **TRỰC TIẾP CẢI THIỆN MÃ NGUỒN VÀ CHẤT LƯỢNG MÔ HÌNH AI ROADWATCH**.

Không được chỉ viết kế hoạch, tạo báo cáo Markdown hoặc cập nhật prompt.md rồi tuyên bố hoàn thành.

Hãy thực hiện công việc kỹ thuật có thể kiểm chứng ngay trong workspace. Với những phần bị chặn bởi quyền dữ liệu, checkpoint, nhãn hoặc tài nguyên, phải giữ nguyên blocker và triển khai phần độc lập có thể kiểm thử.

Không tạo chỉ số giả. Không cam kết đạt 85% nếu chưa có kết quả thực nghiệm hợp lệ.

---

# I. PROJECT INFORMATION

Repository:

https://github.com/ln3076059-jpg/prok53

Development branch:

roadwatch-v4-candidate

Local workspace:

D:\.idea\giangdoantotnghiep

Last verified HEAD:

bcb0952161afb74ecd3cce987d95f025725c0a7b

Latest research implementation:

a1203a7f53697bcd566d0c2eca791b089c70db08

Trước khi triển khai phải kiểm tra HEAD mới nhất, git status, AGENTS.md nếu có và những thay đổi chưa được commit.

Không reset hoặc ghi đè worktree của người dùng.

Không xóa dataset, checkpoint, video, logs hay lịch sử benchmark.

Không push/merge lên main hoặc kích hoạt production model khi chưa được cho phép.

---

# II. RESEARCH OBJECTIVES

Mục tiêu nghiên cứu:

PHONE EVENT:
- Precision >85%.
- Recall >85%.
- F1-score >85%.
- False Alarms/min <=0.15.

ACTION RECOGNITION:
- SAFE_DRIVING Recall >85%.
- PHONE_USE Recall >85%.
- SAFE/PHONE/OTHER Macro-F1 >85%.
- Ổn định trên những người lái mới.

PHONE DETECTION:
- AP50 >85% nếu có bộ detection ground truth hợp lệ.
- Báo cáo mAP50-95 và per-class metrics.

SEATBELT:
- Macro-F1 >85%.
- FASTENED Recall >85%.
- UNFASTENED Recall >85%.
- UNCERTAIN_OR_OCCLUDED Recall >85%.

Các mục tiêu chỉ được công nhận trên đúng evaluation protocol.

Không dùng detector mAP thay cho Event F1.

Không dùng Clip Macro-F1 thay cho Event F1.

Không dùng Accuracy của bài báo khoa học làm kết quả Roadwatch.

---

# III. VERIFIED V13 BASELINE

Đọc trực tiếp:

reports/v13_research/12_FINAL_AI_LEAD_REPORT.md
reports/v13_research/04_PHONE_HIGH_RES_ROI.md
reports/v13_research/05_ONLINE_E3_ARTIFACT_READINESS.md
reports/v13_research/08_FUSION_AND_EVENT_REPLAY.md
reports/v13_research/candidate_readiness.json
reports/v13_research/metrics_comparison.json

Các kết quả đã báo cáo:

PHONE EVENTS:
- TP=16
- FP=4
- FN=13
- Precision=80.00%
- Recall=55.17%
- Event F1=65.31%
- FA/min=0.1813

E3 ACTION:
- Clip Macro-F1=56.39%.
- SAFE Recall=60.38%.
- PHONE Recall=66.78%.
- OTHER Recall=43.26%.
- Worst-subject SAFE Recall=54.36%.

SEATBELT:
- Best development Macro-F1=58.52%.
- UNCERTAIN Recall=4%.

V13:
- R5 crop bảo toàn 12/12 reviewed phone boxes.
- R5 là oracle-box diagnostic, chưa phải detector autonomous.
- E3 fitted head/scaler chưa có.
- E3 Event F1 chưa được đo.
- Occupant association accuracy chưa được đo.
- 500 seatbelt labels vẫn pending.
- 619 full Pytest PASS theo báo cáo.
- Production approved=false.

Phải kiểm tra lại số liệu từ các JSON artifacts thay vì tự tin vào bảng tóm tắt.

---

# IV. P0 — ROOT-CAUSE INVESTIGATION

Không tiếp tục viết báo cáo nguyên nhân chung chung.

Đọc mã nguồn và xác định chính xác bước nào ngăn tạo một true event:

FRAME
→ DRIVER ROI
→ PHONE PROPOSALS
→ ACTION FEATURES
→ ACTION SCORES
→ DRIVER ASSOCIATION
→ FUSION
→ TEMPORAL EVENT
→ EVALUATION

Đối với 13 historical false negatives, giữ nguyên status DEVELOPMENT_DIAGNOSTIC_ONLY.

Phân tích và ghi:

- Phone proposal có được tạo tự động không?
- Có bị NMS hoặc confidence admission loại không?
- ROI có chứa điện thoại không?
- Detector có nhầm mounted camera/phone không?
- Action evidence có vượt gate không?
- Driver/occupant role có xác minh được không?
- Fusion có bác bỏ event không?
- Temporal engine có tạo fragment hoặc suppression không?
- Event matcher có match prediction không?

Không biến 12 reviewed phone boxes thành 12 recovered events.

Không dùng GT event boundaries để tạo prediction.

---

# V. P1 — AUTONOMOUS PHONE DETECTOR

**Đây là ưu tiên cao nhất.**

V13 đã có crop độ phân giải gốc nhưng chưa chứng minh detector tự tìm phone boxes.

Đọc:

training/dmd/high_res_roi.py
training/dmd/behavior_roi.py
training/dmd/driver_phone_association.py

Và tìm các detector wrappers, checkpoint YOLO, model registry hiện tại.

Không tạo detector wrapper thứ hai nếu repository đã có một wrapper đáp ứng yêu cầu.

Triển khai autonomous inference:

1. Input là frame gốc chưa gán nhãn.
2. Xác định vùng người lái bằng bằng chứng runtime.
3. Chạy phone detector hoặc proposal generator hiện có.
4. Thu thập raw bounding boxes và confidence.
5. Chuyển tọa độ về frame gốc.
6. Tạo dynamic padded ROI.
7. Trích xuất phone/hand/face evidence khi có.
8. Giữ missing evidence ở trạng thái UNKNOWN.
9. Xuất structured observations cho fusion.
10. Lưu model hash, config hash, frame hash và lý do lọc proposals.

Phải có hai chế độ rõ ràng:

ORACLE_DIAGNOSTIC:
- Được dùng human-reviewed boxes.
- Chỉ để kiểm tra crop geometry và giữ chi tiết.

AUTONOMOUS_INFERENCE:
- Tuyệt đối không dùng reviewed/GT boxes làm detector input.
- Chỉ dùng các boxes sinh ra tại runtime.

Không báo autonomous detection accuracy từ ORACLE_DIAGNOSTIC.

Nếu có checkpoint YOLO hợp lệ, kiểm tra kiến trúc, classes, provenance và license trước khi sử dụng.

Không tự gọi một checkpoint bất kỳ là Roadwatch best.pt.

Thử phương án lightweight có kiểm soát:
- Full driver inference.
- High-resolution driver ROI.
- Padded upper-body ROI.
- Hand/phone secondary ROI khi có evidence runtime.
- Class-aware suppression khi phù hợp.

Không sweep threshold trên holdout.

Không gộp đồng thời nhiều thay đổi rồi quy kết hiệu quả cho một tham số.

---

# VI. P2 — ORIGINAL-RESOLUTION BEHAVIOR REPRESENTATION

Nâng cấp pipeline hiện có để tận dụng chi tiết gốc.

V13 đã cho thấy phone boxes từ ảnh gốc co xuống rất nhỏ tại RGB cache 64x64.

Thử các representation sau khi có quyền fitting hợp lệ:

R0 — Frozen 64x64 baseline.

R1 — High-resolution upper-body appearance.

R2 — Original-resolution hand/phone evidence.

R3 — Torso + hand + face feature fusion.

R4 — Causal motion + spatial ROI.

R5 — Selected representation with controlled validation.

Không phóng lớn 64x64 rồi gọi là khôi phục thông tin gốc.

Khi chọn crop phải đo:
- Pixel coverage.
- Edge clipping.
- Context loss.
- Occlusion.
- Feature dimensions.
- Peak RAM.
- Processing latency.

Thử frozen encoder trước fine-tuning lớn.

Kiểm tra class collapse:
- SAFE_DRIVING.
- PHONE_USE.
- OTHER_ACTIVITY.

Không chọn model chỉ vì pooled Macro-F1 tăng.

Phải xem worst-subject Recall.

---

# VII. P3 — COMPLETE E3 ONLINE MODEL

V12/V13 đã có OnlineE3 features với batch-stream parity.

Không viết lại module này nếu không cần.

Nhiệm vụ:

1. Kiểm tra input feature schema.
2. Kiểm tra feature dimension.
3. Kiểm tra class order.
4. Kiểm tra causal state reset.
5. Kiểm tra session/track/camera transitions.
6. Kiểm tra timestamp gaps và dropped frames.
7. Kiểm tra confidence/probability semantics.
8. Kiểm tra khả năng export/load fitted artifact.

Classes:

SAFE_DRIVING
PHONE_USE
OTHER_ACTIVITY

Nếu có fitted E3 artifact hợp lệ:
- Verify SHA-256.
- Verify feature version.
- Verify scaler parameters.
- Verify class order.
- Verify training provenance.
- Reproduce original predictions.
- Test synthetic/real input contracts.

Nếu không có fitted head/scaler:
- Không tạo hệ số giả.
- Không tự fit trên DMD chưa được phép.
- Giữ BLOCKED_MISSING_FITTED_HEAD_SCALER.
- Hoàn thành adapter bằng synthetic fixtures.
- Không công bố E3 Event F1.

Phải phân biệt decision scores với calibrated probabilities.

Đặc biệt đối chiếu 25 stored rows có decision argmax khác probability argmax trong V12.

---

# VIII. P4 — DRIVER/PHONE ASSOCIATION

Nâng cấp module đã có.

Không biến occupant UNKNOWN thành DRIVER_ASSOCIATED.

Các trạng thái:

DRIVER_ASSOCIATED
PASSENGER_ASSOCIATED
MOUNTED_OR_STATIC
UNKNOWN

Đánh giá:
- Driver seat evidence.
- Verified occupant track.
- Phone-hand relation.
- Phone-face relation.
- Temporal track consistency.
- Camera calibration nếu có.
- Multiple occupant ambiguity.
- Mounted context verification.

Không mặc định phone nằm trong person box chứng minh người đó đang sử dụng điện thoại.

Không tự xem heuristic score là calibrated probability.

Thêm regression tests:
- Driver cầm phone.
- Passenger cầm phone.
- Mounted camera.
- Mounted phone.
- Phone near driver/passenger overlap.
- Missing occupant track.
- Unverified mounted context.
- Missing hand/face landmarks.

Nếu chưa có human-approved occupant labels, chỉ báo test/simulation và UNKNOWN coverage; không báo real association accuracy.

---

# IX. P5 — ACTION MODEL & TEMPORAL RESEARCH

Tham khảo có kiểm chứng:

DualScope:
https://ojs.aaai.org/index.php/AAAI/article/view/37812

DriveCLIP:
https://github.com/zahid-isu/DriveCLIP

ActionFormer:
https://arxiv.org/abs/2202.07925

MixStyle:
https://arxiv.org/abs/2104.02008

Group DRO:
https://arxiv.org/abs/1911.08731

Chỉ nghiên cứu các phương pháp phù hợp dữ liệu và tài nguyên.

Ưu tiên:
- Critical-region features.
- Phone/hand interactions.
- Multi-scale causal motion.
- Small lightweight action head.
- Temporal boundary modeling.
- Subject-domain normalization.
- Hard negatives.
- Worst-subject stability.

Không triển khai toàn bộ mô hình lớn chỉ vì paper công bố Accuracy cao.

Nếu muốn sử dụng CLIP hoặc DualScope checkpoints:
- Kiểm tra license.
- Kiểm tra model size.
- Kiểm tra preprocessing.
- Kiểm tra class mapping.
- Profile compute.
- Thiết kế baseline comparison trước khi tải.

Không gọi một prototype là tái hiện paper đầy đủ.

---

# X. P6 — PHONE EVENT REPLAY

Đây là điều kiện bắt buộc để chứng minh AI thực sự cải thiện.

Nối các module hiện có:

AUTONOMOUS DETECTOR
→ SOURCE-RESOLUTION ROI
→ CAUSAL ACTION MODEL
→ DRIVER ASSOCIATION
→ EVIDENCE FUSION
→ TEMPORAL EVENT ENGINE
→ FROZEN EVENT MATCHER

Mỗi observation cần:
- Frame timestamp.
- Source camera.
- Driver track ID hoặc UNKNOWN.
- Detector confidence.
- Action decision scores.
- Fusion scores.
- Evidence provenance.
- Reason codes.

Mỗi event cần:
- Stable event ID.
- Start/end timestamps.
- First evidence timestamp.
- Actual emission timestamp.
- Source track.
- Configuration hash.

Bắt buộc xuất:
- TP IDs.
- FP IDs.
- FN IDs.
- Recovered TP IDs.
- Lost TP IDs.
- New FP IDs.
- Removed FP IDs.

Metrics:
- Event Precision.
- Event Recall.
- Event F1.
- FA/min.
- Short-event recall.
- Fragmentation.
- Duplicate alerts.
- Onset latency.
- Emission latency.

Không sửa ground truth, matcher hoặc denominator.

Không tự kéo dài prediction theo GT.

Không báo đã có E3 Event F1 khi candidate chưa chạy.

Các thí nghiệm khác scope phải được báo riêng.

---

# XI. P7 — DATA & RIGHTS GATE

Current reported state:

RIGHTS_ACCEPTED=false
CANONICAL_ELIGIBLE=false
TEMPORAL_DEVELOPMENT_ELIGIBLE=true

Phải đọc authoritative record trực tiếp.

Không tự bật các cờ quyền.

DMD fitting/training/upload chỉ thực hiện khi có approval rõ ràng và dữ liệu phù hợp phạm vi cấp phép.

Protected:
gZ-37
gE-28

Không sử dụng để phát triển hoặc tuning.

Development:
gB-9
gC-14
gZ-36

Đã được xem xét nhiều lần, không còn là independent test.

## Data acquisition route

Chuẩn bị pipeline tiếp nhận:
- Video do người dùng sở hữu và được phép sử dụng.
- Dữ liệu công khai có license phù hợp, được kiểm chứng.
- Human-annotated new subjects.
- Hard negatives.
- Metadata subject/session/camera.
- Dataset hashes.
- Group-disjoint splits.

Không mặc định mọi public dataset đều được phép fine-tune.

Tập mới phải tách theo người lái, session và source video.

Không random split frames cùng video qua train/test.

Nếu không có dữ liệu mới hợp lệ:
DATA_GENERALIZATION_BLOCKED

Không lặp lại nhiều vòng model selection trên 3 development subjects rồi gọi là independent improvement.

---

# XII. P8 — SEATBELT IMPROVEMENT

Giữ ba trạng thái:
FASTENED
UNFASTENED
UNCERTAIN_OR_OCCLUDED

Hiện còn 500 pending review candidates.

Không tự phê duyệt human labels.

Hoàn thiện:
1. Review workflow.
2. Label provenance.
3. Torso ROI quality.
4. Visibility evidence.
5. Uncertainty handling.
6. Hard-negative test cases.

Khi đủ điều kiện:
- Warm-start checkpoint hiện có.
- Train trên approved labels.
- Group-disjoint validation.
- Compare ROI variations.
- Try lightweight attention.
- Evaluate uncertainty calibration.

Metrics:
- Macro-F1.
- Per-class Recall.
- Balanced Accuracy.
- False-unfastened rate.
- Abstention rate.
- Coverage.

Không coi dây an toàn không quan sát được là không thắt dây.

---

# XIII. P9 — EXPERIMENT PRIORITIES

Thiết lập experiment registry trước khi chạy.

EXP-01:
Baseline and provenance verification.

EXP-02:
Autonomous phone proposal extraction.

EXP-03:
Source-resolution crop without oracle boxes.

EXP-04:
Driver-phone association safety tests.

EXP-05:
OnlineE3 fitted artifact readiness.

EXP-06:
Causal action features on eligible data.

EXP-07:
Fixed-policy event replay.

EXP-08:
Seatbelt approved-label experiment.

EXP-09:
Full pipeline runtime.

Thứ tự thực hiện phải tối ưu theo value/cost.

Mỗi thí nghiệm:
- Nêu giả thuyết.
- Giữ yếu tố kiểm soát.
- Đo chỉ số.
- Ghi nguồn dữ liệu.
- Ghi dataset/model hashes.
- Báo phần thay đổi.
- Báo thời gian và tài nguyên.
- Giữ nguyên kết quả thất bại.

Không thực hiện threshold sweep vô hạn.

Không thay metric protocol giữa baseline và candidate.

---

# XIV. P10 — ENGINEERING QUALITY

Không tạo duplicate modules.

Tái sử dụng:
- training/dmd/high_res_roi.py
- training/dmd/behavior_roi.py
- training/dmd/driver_phone_association.py
- Existing OnlineE3 code.
- Existing fusion code.
- Existing temporal evaluator.
- Existing backend services.

Thêm unit/regression tests cho:

- Runtime proposal generation không đọc GT boxes.
- Empty proposal handling.
- ROI coordinate round-trip.
- Phone at image boundary.
- Mounted camera false positives.
- Driver/passenger ambiguity.
- Missing tracks.
- Timestamp gaps.
- Causal future-frame invariance.
- Batch-stream parity.
- Safe artifact loading.
- Class-order consistency.
- Uncalibrated probabilities.
- Event duplication.
- Event fragmentation.
- Emission timestamp correctness.
- Seatbelt uncertainty.
- Data rights.
- Protected holdout blocking.
- Metric reconciliation.

Chạy:
- Python compile.
- Ruff.
- Targeted pytest.
- Full pytest.
- Backend smoke.
- Frontend build.
- Model smoke nếu đủ checkpoint và tài nguyên.

Không chỉnh hoặc xóa test để báo PASS.

Không suy diễn test PASS từ log cũ.

---

# XV. P11 — RUNTIME BENCHMARK

Đo riêng:
- Video decode.
- Driver localization.
- Phone detector.
- ROI extraction.
- Action encoder.
- Fitted action head.
- Driver association.
- Fusion.
- Temporal event.
- Evidence persistence.

Báo:
- Processing FPS.
- p50/p95 latency.
- Peak RAM.
- VRAM nếu có.
- Hardware.
- Input resolution.
- Number of views.
- Online alert delay.

V13 ROI-suite p50 25.939ms chỉ là thời gian crop diagnostics.

Không gọi nó là full pipeline inference.

---

# XVI. REQUIRED OUTPUTS

Tạo báo cáo tại:

reports/v14_research/

Chỉ tạo file có kết quả thực, không tạo placeholder để đủ số lượng.

Cần có:

- Repository and rights audit.
- Baseline reconciliation.
- Autonomous phone detector results.
- Original-resolution ROI results.
- E3 candidate readiness.
- Driver association results.
- Event replay results hoặc blocker chính xác.
- Seatbelt status/results.
- Runtime.
- Test logs.
- Final AI Lead review.

Kèm các machine-readable artifacts khi thực sự được tạo:
- experiment_registry.json
- metrics_comparison.json
- candidate_readiness.json
- dataset_manifest.json
- model_manifest.json
- event_match_records.json
- reproducibility commands.

---

# XVII. DEFINITION OF DONE

## Engineering success

Phải có thể chứng minh:

1. Autonomous inference không phụ thuộc ground-truth phone boxes.
2. Source-resolution crop hoạt động.
3. ROI coordinates và provenance đúng.
4. Missing/uncertain evidence được xử lý đúng.
5. OnlineE3 interface có input/output rõ ràng.
6. Candidate không thay production mặc định.
7. Regression tests pass.
8. Benchmark ghi rõ scope.

## Model improvement success

Chỉ công nhận khi có:
- Dữ liệu và quyền phù hợp.
- Trọng số fitted hợp lệ.
- Prediction records đầy đủ.
- Event metrics được tính thật.
- Baseline comparison cùng protocol.
- Không có leakage.
- Không có catastrophic worst-subject regression.

## Independent AI85 success

Chỉ được báo đạt khi:
- Mô hình và threshold đã khóa.
- Có independent holdout mới.
- Precision, Recall và F1 đạt mục tiêu.
- False alarms phù hợp.
- Các metrics được tái lập từ prediction records.

Nếu chưa đủ:
production_approved=false
independent_ai85_established=false

Không đổi các giá trị này để tạo cảm giác hoàn thành.

---

# XVIII. EXECUTE NOW

BẮT ĐẦU TRỰC TIẾP TRONG WORKSPACE.

Thứ tự thực hiện:

1. Verify Git HEAD và working tree.
2. Verify quyền dữ liệu.
3. Reconcile V13 baseline.
4. Inspect detector/ROI/action/fusion source.
5. Implement autonomous phone candidate path.
6. Add source-resolution crop integration.
7. Add negative-context and association tests.
8. Complete OnlineE3 artifact readiness.
9. Run available inference diagnostics.
10. Execute event replay only when valid fitted model and evaluation records exist.
11. Audit seatbelt labels.
12. Run tests and benchmarks.
13. Report genuine improvements and remaining blockers.

KHÔNG CHỈ SỬA prompt.md HOẶC PRODUCT.md.

Không yêu cầu tạo báo cáo dài nếu không có cải tiến kỹ thuật mới.

Final response phải tóm tắt:

- Files đã sửa.
- Các tính năng AI thực sự hoạt động.
- Experiments đã chạy.
- Baseline vs candidate.
- Event P/R/F1 nếu đo được.
- SAFE Recall nếu đo được.
- Seatbelt Macro-F1 nếu đo được.
- Runtime.
- Tests.
- Blockers.
- Next highest-value experiment.

**MỤC TIÊU: BIẾN ROADWATCH THÀNH HỆ THỐNG PHÁT HIỆN VI PHẠM TỰ ĐỘNG CÓ KHẢ NĂNG TỔNG QUÁT HÓA VÀ BẰNG CHỨNG KHOA HỌC, KHÔNG CHỈ LÀ TẬP HỢP CÁC THÍ NGHIỆM VÀ BÁO CÁO.**