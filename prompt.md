# MASTER PROMPT — ROADWATCH V10 AI RESEARCH & ENGINEERING
## RESEARCH-DRIVEN AI OPTIMIZATION — TARGET 85%+
### Dành cho Codex chạy trực tiếp trong repository prok53

---

# 0. VAI TRÒ VÀ YÊU CẦU TỔNG QUÁT

Bạn đóng vai trò đồng thời là:

- AI Research Lead / Principal AI Scientist.
- Senior Computer Vision Research Engineer.
- Video Action Recognition Researcher.
- YOLO Object Detection Optimization Engineer.
- Multi-View & Temporal Fusion Specialist.
- MLOps & Model Evaluation Lead.
- Senior Python / FastAPI Engineer.
- Scientific Reproducibility Reviewer.

Bạn có trách nhiệm trực tiếp phân tích, thiết kế, triển khai, thực nghiệm và kiểm thử các cải tiến AI cho dự án Roadwatch.

KHÔNG CHỈ VIẾT KẾ HOẠCH HOẶC BÁO CÁO.

Nếu mã nguồn, checkpoint và dữ liệu hợp lệ đã sẵn có, hãy trực tiếp chạy các bước phù hợp.

Nếu thiếu GPU hoặc dữ liệu được phê duyệt, hãy hoàn thành những phần có thể thực hiện an toàn trên CPU, bổ sung instrumentation, tests, configs và chuẩn bị experiment runner có thể tái lập.

Tuyệt đối không bịa kết quả thực nghiệm.

Mục tiêu 85% là research target, không phải số liệu được phép ép đạt.

---

# 1. THÔNG TIN DỰ ÁN

Repository:

https://github.com/ln3076059-jpg/prok53

Development branch:

roadwatch-v4-candidate

Workspace Windows:

D:\.idea\giangdoantotnghiep

Commit tham chiếu gần nhất đã xác minh:

01c9161c2535206225e1165c0c0880874458b9eb

Tên đề tài đồ án:

"Nghiên cứu và xây dựng hệ thống ứng dụng trí tuệ nhân tạo phát hiện hành vi vi phạm giao thông từ dữ liệu camera giám sát."

Hai bài toán bắt buộc:

1. Phát hiện người lái sử dụng điện thoại.
2. Phát hiện người lái không thắt dây an toàn.

Hệ thống cần có:

- Nhận ảnh/video hoặc nguồn camera được hỗ trợ.
- Computer Vision inference.
- Bounding boxes và bằng chứng hành vi.
- Temporal event detection.
- Cảnh báo và quản lý sự kiện.
- Human review.
- Thống kê, lịch sử và báo cáo.
- Backend FastAPI.
- Frontend React/TypeScript.
- Metrics và benchmark có thể tái lập.

Phân biệt rõ năng lực đã chứng minh trên camera cabin với mục tiêu camera giám sát ngoài đường.

Không tự tuyên bố hệ thống đã hoạt động tốt trên camera ngoài đường khi chưa có dữ liệu kiểm chứng tương ứng.

---

# 2. NHIỆM VỤ ĐẦU TIÊN — XÁC MINH CODE MỚI NHẤT

Trước khi làm bất kỳ experiment nào:

1. Kiểm tra git status.
2. Kiểm tra current branch.
3. Kiểm tra local HEAD.
4. Kiểm tra remote tracking branch.
5. Kiểm tra uncommitted changes.
6. Kiểm tra commit history sau V9.
7. Liệt kê source files liên quan tới AI.
8. Liệt kê checkpoint có thật trên máy.
9. Kiểm tra dữ liệu và quyền sử dụng.
10. Kiểm tra test suite hiện tại.
11. Kiểm tra CPU/GPU/RAM/VRAM/dung lượng ổ đĩa.

Không giả định commit tham chiếu vẫn là HEAD mới nhất.

Không ghi đè thay đổi của người dùng.

Không tự động dùng:

- git reset --hard
- git clean -fd
- git restore đối với dữ liệu người dùng
- Các lệnh xóa hoặc di chuyển toàn bộ dataset

Không xóa checkpoint cũ, log cũ hoặc artifact thực nghiệm lịch sử.

Không push, force-push hoặc merge lên main khi chưa có yêu cầu rõ ràng.

---

# 3. BASELINE V9 CẦN GIỮ NGUYÊN

Theo kết quả nghiên cứu cuối cùng đã được kiểm tra:

## Controlled DMD development comparator

- Số ground-truth phone events: 29.
- Development subjects: gB-9, gZ-36, gC-14.
- True Positive: 16.
- False Positive: 4.
- False Negative: 13.
- Event Precision: 80.00%.
- Event Recall: 55.17%.
- Event F1: 65.31%.
- False Alarms/min: 0.1813.

## V9 Phone-Context Replay

Đánh giá riêng trên 20 events, hai subjects:

- Context-naive: TP6 / FP4 / FN14.
- Context-aware: TP6 / FP3 / FN14.
- Context-aware Precision: 66.67%.
- Context-aware Recall: 30.00%.
- Context-aware F1: 41.38%.

KHÔNG so sánh trực tiếp kết quả replay hai subjects này với controlled comparator 29 events.

## V9 diagnostics

- 13/13 FN có review records.
- 12 trường hợp có selected phone proposals.
- 1 trường hợp điện thoại có thể nhìn thấy nhưng không có proposal hợp lệ.
- 0 trường hợp có verified occupant track.
- Subject silhouette: 0.4095.
- Action silhouette: 0.0486.
- Subject/action separability ratio: 8.42×.
- Dense sampling vẫn bỏ sót 4/4 sự kiện dưới một giây.
- Seatbelt best development Macro-F1: 58.52%.
- Seatbelt UNCERTAIN Recall: 4%.
- Full Pytest theo báo cáo: 569 PASS.
- Production approval: FALSE.
- Independent AI85 validation: NOT ESTABLISHED.

Đọc trực tiếp:

- reports/v9_research/10_FINAL_SCIENTIFIC_REPORT.md
- reports/v9_research/04_REPRESENTATION_EXPERIMENTS.md
- reports/v9_research/05_FUSION_ABLATIONS.md
- reports/v9_research/14_PHONE_CONTEXT_GATE_ABLATION.md
- reports/v9_research/15_PHONE_CONTEXT_EVENT_REPLAY.md
- reports/v9_research/human_phone_review_summary.json
- reports/ai85_plus/16_FINAL_RESEARCH_REPORT.md

Phải kiểm tra các giá trị từ machine-readable artifacts thay vì chỉ tin vào Markdown.

---

# 4. MỤC TIÊU HIỆU NĂNG

## A. Phone-use event detection

Research targets:

- Event Precision >= 85%.
- Event Recall >= 85%.
- Event F1 >= 85%.
- False Alarms/min <= 0.15.
- SAFE_DRIVING Recall >= 85%.
- Clip Action Macro-F1 >= 85%.

## B. Object detection

Khi có dataset bounding-box ground truth hợp lệ:

- Phone AP50 >= 85%.
- Overall mAP50 >= 85%.
- Báo cáo mAP50-95.
- Báo cáo per-class Precision, Recall và PR curves.

## C. Seatbelt classification

- Macro-F1 >= 85%.
- FASTENED Recall >= 85%.
- UNFASTENED Recall >= 85%.
- UNCERTAIN_OR_OCCLUDED Recall >= 85%.
- Balanced Accuracy >= 85%.

## D. Generalization

Phải có subject-disjoint evaluation và independent holdout mới được khóa trước model selection.

Nếu chưa có independent holdout hợp lệ, ghi rõ:

INDEPENDENT_AI85_NOT_ESTABLISHED

Không được tuyên bố đã đạt mục tiêu cuối cùng chỉ bằng development data.

### Ví dụ toán học

Nếu vẫn xét 29 events lịch sử, một kịch bản mục tiêu là:

- TP = 25.
- FP = 3.
- FN = 4.

Khi đó:

- Precision = 89.29%.
- Recall = 86.21%.
- F1 = 87.72%.
- FA/min khoảng 0.136 nếu tổng thời lượng vẫn là 22.06 phút.

Đây CHỈ LÀ VÍ DỤ MỤC TIÊU, tuyệt đối không được tạo hoặc chỉnh dữ liệu để đưa kết quả về các con số này.

---

# 5. PHASE A — SCIENTIFIC LITERATURE REVIEW

Đọc các nghiên cứu sau, kiểm tra DOI, PDF, mã nguồn và checkpoint thực tế.

## Paper 1 — DualScope, AAAI 2026

DualScope: Capturing Critical Spatial and Temporal Cues for Distracted Driving Activity Recognition

Paper:
https://ojs.aaai.org/index.php/AAAI/article/view/37812

Code:
https://github.com/Qqqzzjsupr3me/DualScope

Model:
https://huggingface.co/Qqqzzj/DualScope

Nghiên cứu:

- Behavior-centric spatial representation.
- Spatial knowledge distillation.
- Attention vào những bộ phận liên quan hành vi.
- Temporal motion representation.
- Multi-scale temporal modeling.
- Redundant frame suppression.

Ưu tiên các ý tưởng có thể triển khai như một frozen feature encoder hoặc lightweight adapter.

Không xây dựng lại toàn bộ DualScope nếu không có checkpoint, GPU và dữ liệu cần thiết.

## Paper 2 — DriveCLIP

Vision-Language Models Can Identify Distracted Driver Behavior from Naturalistic Videos

Paper:
https://doi.org/10.1109/TITS.2024.3381175

Code:
https://github.com/zahid-isu/DriveCLIP

Nghiên cứu:

- CLIP embeddings.
- Zero-shot behavior recognition.
- Vision-language representation.
- SAFE vs PHONE vs OTHER discrimination.
- Subject-invariant feature hypothesis.

Thử frozen visual features trước fine-tuning.

Kiểm tra xem code và pretrained checkpoints có sử dụng được cho các clips Roadwatch không.

Không mặc định frame-based code đã có đầy đủ video-level implementation.

## Paper 3 — Multi-Task Fusion

Enhancing Driver Monitoring Systems Based on Novel Multi-Task Fusion Algorithm

Paper:
https://doi.org/10.3390/s25216799

Nghiên cứu:

- 3D ResNet18.
- BiLSTM.
- BODY/FACE/HANDS fusion.
- Action recognition.
- Gaze estimation.
- Hand activity cues.
- Task-aware attention.

Áp dụng khi có đủ các camera views và data labels tương ứng.

Missing view phải được mask hoặc abstain, không thay bằng chứng thiếu bằng negative evidence.

## Paper 4 — DGMB-Net

Towards Efficient Risky Driving Detection: A Benchmark and a Semi-Supervised Model

Paper:
https://doi.org/10.3390/s24051386

Nghiên cứu:

- Teacher–student learning.
- Semi-supervised object detection.
- Hierarchical feature pyramid.
- Pseudo-label uncertainty.
- Class-dependent confidence.
- Annotation efficiency.

Chỉ đưa semi-supervised learning vào training khi có dữ liệu được phép sử dụng và quy trình human review phù hợp.

Không dùng pseudo-labels từ chính ground-truth evaluation subjects để chứng minh cải thiện trên các subjects đó.

## Paper 5 — RST-Net / Lightweight Seatbelt

Seatbelt Detection Algorithm Improved with Lightweight Approach and Attention Mechanism

Paper:
https://doi.org/10.3390/app14083346

Nghiên cứu:

- Torso/windshield localization.
- Lightweight detection.
- Triplet attention.
- Small-object features.
- False-positive reduction.

Phân biệt detection của dải dây và classification trạng thái thắt dây.

## Paper 6 — Overhead Traffic Surveillance

Real-time Detection of Seat Belt Usage in Overhead Traffic Surveillance Using YOLOv7

Paper:
https://doi.org/10.35784/acs_7594

Nghiên cứu:

- Vehicle ROI.
- Windshield ROI.
- Occupant ROI.
- Seatbelt evidence.
- Day/night robustness.
- Camera-domain transfer.

Không coi kết quả overhead dataset là bằng chứng Roadwatch đã hoạt động tốt trên roadside videos.

## Literature review deliverable

Tạo bảng so sánh:

- Full title.
- Authors.
- Venue/year.
- DOI.
- Code URL.
- Checkpoint URL.
- License.
- Dataset.
- Evaluation split.
- Main metrics.
- Metric definition.
- Hardware.
- Inference cost.
- Reproducibility status.
- Roadwatch integration feasibility.
- Risks of domain mismatch.

Nếu paper không cung cấp một số metric hoặc thông tin license, ghi NOT VERIFIED.

Không tự tạo các con số 95%, 97%, 98% hoặc mức tăng 5–12%.

Đặc biệt KHÔNG so sánh trực tiếp:

- Classification Top-1 với Event F1.
- YOLO mAP50 với Phone-use Event F1.
- Frame F1 với video-event F1.
- DMD cabin với roadside surveillance.
- Development benchmark với independent holdout.

---

# 6. PHASE B — ROOT-CAUSE ANALYSIS

Mục tiêu là xác định chính xác vì sao 13 events bị bỏ sót.

Tạo một bảng audit mỗi event gồm:

- Event ID.
- Subject.
- Camera view.
- Action label.
- Start/end timestamps.
- Event duration.
- Phone visibility.
- Phone proposal available?
- Proposal confidence.
- Phone box quality.
- Hand proximity.
- Face proximity.
- Occupant association.
- Action model probability.
- SAFE/PHONE/OTHER prediction.
- Temporal activation state.
- Fusion rejection reason.
- Match result.
- Likely failure stage.

Phân loại:

1. Detector miss.
2. Incorrect ROI.
3. Small-object resolution loss.
4. Occlusion.
5. Missing phone boxes.
6. Driver/occupant ambiguity.
7. Action representation collapse.
8. Domain/subject bias.
9. Fusion suppression.
10. Temporal-event failure.
11. Annotation ambiguity.
12. Insufficient visual evidence.

Không tự xem toàn bộ 13 FN là detector failures.

Không xem có một positive phone proposal ở một frame là đã phục hồi được một event.

Phân biệt detector candidate yield với actual event recall.

Xuất bằng chứng cấp frame, clip và event.

---

# 7. PHASE C — PHONE DETECTION & DRIVER ASSOCIATION

## C1. Reuse Existing Checkpoints

Kiểm tra các YOLO models, bao gồm checkpoint `best.pt` thực sự có trên máy.

Xác minh:

- Architecture.
- SHA-256.
- Dataset provenance.
- Class mapping.
- Image size.
- Training framework version.
- Train/val/test split.
- License.

Không giả định checkpoint YOLO11m từ project hoặc notebook khác thuộc Roadwatch.

Không tự tải checkpoint dung lượng lớn khi chưa biết dung lượng ổ đĩa.

## C2. Small-Object Detection

Thử nghiệm có kiểm soát:

- Original full-frame inference.
- Driver ROI inference.
- Padded driver ROI.
- High-resolution ROI.
- Phone/hand interaction ROI.
- Scale-aware cropping.
- Context-preserving crop.
- Camera-aware preprocessing.

Bảo đảm chuyển tọa độ bbox từ ROI về ảnh gốc chính xác.

Đặc biệt phân tích các trường hợp điện thoại sát rìa ROI.

## C3. Negative Context

Sử dụng thông tin:

- HANDHELD_USE.
- MOUNTED_OR_STATIC.
- PHONE_VISIBLE_BUT_NOT_USED.
- UNKNOWN.
- POSSIBLE_USE.

Không tự coi điện thoại xuất hiện là vi phạm.

Không tự coi camera gắn cố định là điện thoại.

Giữ raw confidence và admitted confidence riêng biệt.

Lưu rejection reason.

Context chưa được xác thực không được sử dụng như bằng chứng chắc chắn.

## C4. Occupant Tracking

Thiết kế module driver association:

- Driver ROI hoặc driver seat position.
- Occupant/person track.
- Phone association.
- Hand association.
- Temporal identity consistency.
- Unknown driver identity handling.

Có thể thử ByteTrack hoặc cơ chế association nhẹ khi thích hợp.

Không mặc định ByteTrack tự xác định ai là tài xế.

Khi driver association chưa đủ tin cậy:

- Mark UNKNOWN.
- Không khẳng định hành khách sử dụng điện thoại là tài xế vi phạm.

Thực hiện regression tests cho:

- Driver holding phone.
- Passenger holding phone.
- Mounted phone.
- Mounted cabin camera.
- Hand near ear without phone.
- Phone occluded.
- Phone on lap.
- Multiple occupants.

---

# 8. PHASE D — BEHAVIOR-CENTRIC REPRESENTATION

Đây là ưu tiên nghiên cứu cao nhất.

V9 báo cáo subject silhouette lớn hơn action silhouette 8.42 lần.

Giả thuyết: Frozen embeddings còn mang nhiều thông tin nhận diện subject/background và chưa đủ mạnh để phân biệt hành vi.

Phải kiểm nghiệm giả thuyết thay vì coi nó là nguyên nhân đã được chứng minh.

## Experiment D0 — Frozen Baseline

Giữ toàn bộ model hiện tại.

Đo:

- Subject silhouette.
- Action silhouette.
- SAFE/PHONE/OTHER confusion matrix.
- Clip Macro-F1.
- Event F1.
- Per-subject predictions.

## Experiment D1 — Torso/Hand/Face Crops

So sánh:

- Full driver ROI.
- Upper-body ROI.
- Torso ROI.
- Hand ROI.
- Face ROI.
- Full ROI + selected body-part features.

Không áp dụng một crop cố định cho mọi camera nếu crop làm mất hành vi quan trọng.

## Experiment D2 — Motion Features

Thử:

- RGB frame difference.
- Optical-flow-inspired motion statistics.
- Temporal residual features.
- Multi-scale temporal pooling.
- Hand trajectory features.

Ưu tiên lightweight methods trước.

## Experiment D3 — CLIP Frozen Features

Nếu code/checkpoint được phép dùng:

- Test frozen visual encoder.
- Extract compatible embeddings.
- Normalize properly.
- Train small controlled head trên development training split hợp lệ.
- Measure subject/action separability.

Không mặc định CLIP tốt hơn ResNet.

## Experiment D4 — ResNet18 + GRU/BiLSTM

Kế thừa models có sẵn.

Thử:

- Better input representation.
- Balanced negatives.
- Residual motion information.
- Multi-ROI fusion.
- Lightweight temporal head.

Không lặp lại thí nghiệm attention hoặc Native112 thất bại nếu không có giả thuyết mới.

## Experiment D5 — Representation Fusion

Kết hợp có kiểm soát:

- Frozen visual embeddings.
- Motion embeddings.
- Phone evidence.
- Hand/face geometry.
- Temporal action probabilities.

Đánh giá ablation của từng thành phần.

Không bổ sung nhiều yếu tố cùng lúc khiến không xác định được nguyên nhân tăng/giảm F1.

---

# 9. PHASE E — CLASS IMBALANCE AND HARD NEGATIVES

Tập trung vào ba lớp:

- SAFE_DRIVING.
- PHONE_USE.
- OTHER_ACTIVITY.

Kiểm tra:

- Distribution.
- Per-class recall.
- Class collapse.
- Label noise.
- Duplicate clips.
- Train/validation leakage.
- Subject identity bias.

Nghiên cứu:

- Class-balanced sampling.
- Class weights.
- Focal loss.
- Hard-negative mining.
- Supervised contrastive loss khi labels đủ tốt.
- Consistency regularization.
- Margin-based classification.
- Confidence calibration.

Hard-negative examples nên bao gồm:

- Người lái đưa tay gần tai.
- Chỉnh gương.
- Uống nước.
- Điều chỉnh vô lăng.
- Tay gần mặt.
- Điện thoại gắn giá đỡ.
- Camera cabin bị phát hiện nhầm.
- Hành khách cầm điện thoại.
- Vật thể tương tự điện thoại.

Chỉ dùng labels đáng tin cậy.

Không lấy các FN development đã xem nhiều lần đưa thẳng vào training rồi đánh giá lại trên chính những event đó như thể là generalization.

---

# 10. PHASE F — TEMPORAL ACTION & EVENT RECOVERY

Các thí nghiệm V8 cho thấy:

- Dense sampling phục hồi một long event.
- Dense sampling tạo nhiều false positives.
- 4/4 subsecond events vẫn bị bỏ sót.

Do đó không lặp lại fixed dense sampling đơn thuần.

## Nghiên cứu mới

- Motion-aware clip representation.
- Event proposal generation.
- Per-frame action evidence.
- Short-action detection.
- Background/transition modeling.
- Onset and offset localization.
- Class-aware temporal confidence.
- Event fragmentation control.
- Confidence hysteresis.
- Temporal consistency without excessive delay.

## Causal vs Non-causal

Kiểm tra kỹ inherited BiGRU context.

Nếu inference sử dụng future frames, phải báo cáo thật độ trễ phát hiện.

Không gọi hệ thống real-time/causal nếu action encoder cần nhìn trước các khung hình tương lai.

Có thể thiết kế riêng:

1. Offline high-accuracy video analysis.
2. Online low-latency mode.

Không gộp hai chế độ khi tính latency.

## Event integrity

Giữ nguyên:

- Ground truth.
- Event matcher.
- Event matching IoU.
- Frozen matching tolerance.
- Prediction-to-event semantics.

Không kéo dài predicted intervals chỉ để tăng overlap.

Không dùng ground-truth start/end để kích hoạt event proposal lúc inference.

---

# 11. PHASE G — MULTI-VIEW FUSION

Dữ liệu DMD có BODY/FACE/HANDS, nhưng không phải subject nào cũng có đủ views.

Thiết kế:

- BODY-primary features.
- FACE cues cho phone call.
- HANDS cues cho texting.
- Confidence-calibrated fusion.
- Reliability-aware weighting.
- Missing-view mask.
- Negative evidence gating.
- Driver association evidence.

Không cho camera phụ tự tạo event nếu không có bằng chứng người lái phù hợp.

Không sử dụng một view bị thiếu như bằng chứng negative.

Thử ít nhất:

G0 — Existing late fusion.

G1 — Geometry-aware fusion.

G2 — Motion + action fusion.

G3 — Context-aware negative evidence.

G4 — Multi-ROI behavior fusion.

G5 — Selected candidate với evaluation hợp lệ.

Mọi thí nghiệm phải xuất:

- Recovered TP IDs.
- Lost TP IDs.
- New FP IDs.
- Removed FP IDs.
- FN changes.
- FA/min.
- Per-subject metrics.

Không chỉ báo cáo tăng Precision mà bỏ qua Recall.

---

# 12. PHASE H — SEATBELT RESEARCH

Roadwatch phải chứng minh được cả nhiệm vụ dây an toàn.

Hiện tại:

- Seatbelt Macro-F1 dưới 60%.
- UNCERTAIN Recall rất thấp.
- Một số labels đang chờ human approval.

## H1. Data Audit

Kiểm tra:

- FASTENED.
- UNFASTENED.
- UNCERTAIN_OR_OCCLUDED.
- Label source.
- Human approval.
- Occlusion.
- Camera angle.
- Color/lighting.
- False-unfastened cases.
- Dataset subject groups.

Không gộp UNCERTAIN vào UNFASTENED.

Không dùng labels model-assisted pending approval làm confirmed GT.

## H2. ROI Experiments

- Original torso ROI.
- Better upper-body ROI.
- Belt-line-aware features.
- Context-preserving crop.
- Appropriate TTA.

Không tự mặc định tăng resolution từ 224 lên 320 giúp cải thiện: thí nghiệm trước cho thấy có thể giảm Macro-F1.

## H3. Architecture Research

Có thể thử:

- ResNet18 fine-tune.
- Lightweight attention module.
- Triplet Attention.
- Two-stage seatbelt localization.
- Multi-task belt visibility/state prediction.

Thực hiện mỗi thay đổi bằng ablation.

## H4. Uncertainty Handling

Tách:

- No belt visible.
- Confirmed unfastened.
- Occluded/unknown.

Khi không đủ bằng chứng, trả về UNCERTAIN.

Đo:

- Macro-F1.
- Per-class Precision/Recall/F1.
- Balanced accuracy.
- False-unfastened rate.
- Abstention rate.
- Calibration error.

Đừng tối ưu false-unfastened rate bằng cách abstain trên mọi ảnh.

Báo cáo coverage và selective risk.

---

# 13. PHASE I — DATASET GOVERNANCE & GENERALIZATION

Đây là điều kiện bắt buộc trước khi công bố mục tiêu 85%.

## Protected subjects

Không sử dụng cho training hoặc tuning:

- gZ-37.
- gE-28.

Giữ nguyên trạng thái holdout đã khóa.

## Development subjects

Các subject:

- gB-9.
- gZ-36.
- gC-14.

đã bị sử dụng lặp lại qua nhiều vòng.

Không tiếp tục xem chúng là independent test.

## Rights & Licensing

DMD intake trước đây còn ghi:

- RIGHTS_ACCEPTED=false.
- CANONICAL_ELIGIBLE=false.

Phải xác minh trạng thái hiện tại trước mọi hoạt động sử dụng dữ liệu mới, tải lên dịch vụ bên ngoài, đào tạo hoặc phát hành artifact.

Không tự giả định quyền sử dụng.

Không tải video, dữ liệu cá nhân hoặc holdout lên Kaggle/Hugging Face nếu chưa có quyền và approval rõ ràng.

## New subjects

Nếu có dữ liệu mới hợp lệ:

1. Verify provenance.
2. Verify licenses.
3. Inspect annotations.
4. Deduplicate.
5. Group by subject/camera.
6. Define train/validation split.
7. Lock independent holdout.
8. Record dataset hash.
9. Train/select models only on eligible training and validation.
10. Freeze model, threshold and event policy.
11. Evaluate independent holdout.

Không được nhìn independent holdout để quyết định architecture hoặc threshold.

Nếu thiếu subject mới, ghi:

DATA_GENERALIZATION_BLOCKED

Không tạo synthetic subject IDs để giả lập independent evaluation.

---

# 14. PHASE J — TRAINING STRATEGY

Không train from scratch nếu có checkpoint phù hợp.

Ưu tiên:

1. Existing checkpoint audit.
2. Frozen feature comparison.
3. Light classifier/head training.
4. Partial backbone fine-tune.
5. Detector fine-tuning with approved labels.
6. Larger model training only when justified.

## CPU-only

Khi không có CUDA:

- Run artifact verification.
- Build feature extraction pipeline.
- Execute lightweight probes.
- Verify input/output contracts.
- Perform error analysis.
- Train small heads if feasible.
- Run unit tests.
- Prepare GPU configs.

Không báo GPU benchmark khi thực tế chỉ có CPU.

## GPU

Khi có GPU:

- Verify GPU type.
- Verify VRAM.
- Run one controlled pilot.
- Profile memory.
- Set suitable batch size.
- Record GPU hours.
- Use early stopping.
- Store training curves.
- Preserve checkpoints.

Không cam kết thời gian training 15–20 phút nếu chưa đo.

## Warm-start

Nếu checkpoint best.pt phù hợp:

- Verify SHA.
- Verify classes.
- Load pretrained weights.
- Use a new experiment name.
- Fine-tune only with eligible data.

Nếu cần exact resume, verify last.pt và training state.

Không nhầm warm-start và resume.

## Zero-retraining

Chỉ thử TTA/NMS/resolution khi có giả thuyết cụ thể.

Các thí nghiệm này đã từng cho kết quả không thuyết phục trong V8.

Không tiếp tục quét threshold trên development subjects đã xem nhiều lần.

Khi đo mAP của YOLO phải sử dụng evaluation settings phù hợp để xây dựng PR curve.

Không thay đổi confidence, NMS, TTA và image size đồng thời rồi quy tất cả cải thiện cho một yếu tố.

---

# 15. PHASE K — EXPERIMENT DESIGN

Mọi experiment phải có research hypothesis.

## Required fields

- Experiment ID.
- Commit SHA.
- Hypothesis.
- Independent variable.
- Controlled variables.
- Model checkpoint hash.
- Dataset hash.
- Split definition.
- License status.
- Config.
- Training seed.
- GPU/CPU environment.
- Prediction files.
- Metrics.
- Runtime.
- Limitations.
- Decision: KEEP / REJECT / INCONCLUSIVE / BLOCKED.

## Phân cấp mức độ bằng chứng

LEVEL 1:
Diagnostic only.

LEVEL 2:
Development ablation.

LEVEL 3:
Subject-disjoint validation.

LEVEL 4:
Independent locked holdout.

LEVEL 5:
Deployment evaluation.

Không dùng LEVEL 1 kết luận LEVEL 4.

## Phone Metrics

- TP.
- FP.
- FN.
- Precision.
- Recall.
- Event F1.
- FA/min.
- Per-subject F1.
- Worst-subject FA/min.
- Clip Macro-F1.
- SAFE_DRIVING Recall.
- Event onset latency.
- Short-event recall.
- Duplicate/fragmentation count.

## Seatbelt Metrics

- Per-class P/R/F1.
- Macro-F1.
- Balanced Accuracy.
- False-unfastened.
- Abstention/coverage.
- Calibration.

## Object Detection Metrics

- mAP50.
- mAP50-95.
- Per-class AP.
- PR curves.
- False positive classes.
- Inference latency.

Không lẫn lộn các metric này.

---

# 16. PHASE L — SOFTWARE ENGINEERING AND TESTS

Sau khi sửa code:

- Run Python compile.
- Run Ruff critical gate.
- Run scoped Ruff checks.
- Run targeted pytest.
- Run full pytest.
- Run event-metric reconciliation.
- Run data-split guard tests.
- Run subject-leakage tests.
- Run bbox ROI-coordinate tests.
- Run missing-view tests.
- Run geometry-context tests.
- Run seatbelt uncertainty tests.
- Run backend smoke.
- Run frontend build.
- Run inference smoke if model weights available.

Bổ sung regression tests cho:

1. Mounted camera false detection.
2. Phone near ROI border.
3. Phone partially occluded.
4. No valid phone proposal.
5. Unknown driver association.
6. Missing FACE/HANDS view.
7. SAFE driving with hand near face.
8. Short positive event.
9. Overlapping event boundaries.
10. False-unfastened due to occlusion.

Không xóa hoặc nới lỏng test để báo PASS.

Không sửa metric matcher để tăng Event F1.

## Runtime

Đo:

- Detector latency.
- ROI extraction latency.
- Feature encoder latency.
- Temporal head latency.
- Fusion latency.
- Seatbelt inference latency.
- Full-pipeline FPS.
- GPU memory.
- CPU RAM.
- p50/p95 latency.

Không dùng cached-feature FPS thay cho full-system FPS.

Không gọi 21 FPS detector trên Tesla T4 là 21 FPS toàn hệ thống.

---

# 17. PHASE M — MODEL SELECTION & PRODUCTION GATE

Một model chỉ được xem xét chọn khi:

- Có checkpoint và provenance hợp lệ.
- Có đủ dataset rights.
- Có đúng evaluation protocol.
- Có kết quả P/R/F1/FA minh bạch.
- Không dựa trên selected FN development cases.
- Có subject-disjoint validation đáng tin cậy.
- Có confidence/uncertainty handling.
- Không làm mất các TP quan trọng mà không báo cáo.
- Có performance runtime phù hợp.
- Không có regression nghiêm trọng.

Nếu chưa đủ điều kiện:

production_approved = false

Không tự động chỉnh production model config.

Không tự động merge candidate.

Không báo "85% achieved" khi chỉ có development results.

---

# 18. PHASE N — REQUIRED DELIVERABLES

Tạo:

reports/v10_research/

Bao gồm:

01_REPOSITORY_AND_ENVIRONMENT_AUDIT.md

02_PAPER_REVIEW_AND_REPRODUCIBILITY.md

03_DATASET_AND_LICENSE_AUDIT.md

04_MODEL_CHECKPOINT_INVENTORY.md

05_DMD_FN_FP_ROOT_CAUSE.md

06_PHONE_DETECTION_AND_ASSOCIATION.md

07_BEHAVIOR_CENTRIC_REPRESENTATION.md

08_MOTION_AND_VIDEO_ACTION_ABLATIONS.md

09_CLASS_IMBALANCE_AND_SAFE_DRIVING.md

10_MULTI_VIEW_FUSION_EXPERIMENTS.md

11_TEMPORAL_EVENT_EXPERIMENTS.md

12_SEATBELT_RESEARCH_RESULTS.md

13_METRICS_COMPARISON.md

14_GENERALIZATION_VALIDATION.md

15_RUNTIME_BENCHMARK.md

16_TEST_RESULTS.md

17_FINAL_AI_LEAD_REVIEW.md

Cùng:

- experiment_registry.json
- metrics_comparison.json
- dataset_manifest.json
- model_manifest.json
- prediction artifacts
- confusion matrices
- per-subject results
- event matching evidence
- source hashes
- model hashes
- reproduction commands
- failure logs

Không tạo các báo cáo trống chỉ để đủ tên file.

Không điền số liệu giả vào những experiment chưa chạy.

---

# 19. PRIORITY ORDER — THỰC HIỆN THEO THỨ TỰ

## P0 — REPRODUCIBILITY / SAFETY

- Audit Git.
- Audit rights and dataset roots.
- Verify baseline.
- Verify checkpoint.
- Preserve holdouts.

## P1 — DATA & REPRESENTATION

- Analyze subject bias.
- Examine 13 FN.
- Compare ROI/motion/frozen embeddings.
- Improve hand-phone-face evidence.

## P2 — EVENT RECALL

- Better video action representation.
- Short-event recognition.
- Driver/occupant association.
- Context-aware candidate generation.

## P3 — PRECISION & FUSION

- Reduce mounted/static false positives.
- Reliability-aware fusion.
- Preserve recovered TP.
- Reconcile full event metrics.

## P4 — SEATBELT

- Confirm labels.
- Improve UNCERTAIN class.
- ROI/attention experiments.
- Measure false-unfastened risk.

## P5 — TRAINING

- Fine-tune only when lawful data and experiment design are ready.
- Reuse existing weights.
- Run controlled ablations.
- Measure real improvement.

## P6 — VALIDATION

- New group-disjoint subjects.
- Frozen thresholds.
- Locked holdout.
- Honest scientific report.

---

# 20. FINAL EXECUTION CONTRACT

BẮT ĐẦU NGAY.

Không dừng lại ở kế hoạch.

Thực hiện tuần tự:

1. Audit repository và environment.
2. Đọc papers và lập research gap matrix.
3. Phân tích 13 FN cùng diagnostic artifacts.
4. Kiểm tra dataset và checkpoint provenance.
5. Xây dựng behavior-centric ROI representation.
6. Thử frozen feature comparison.
7. Phát triển phone-hand-face / driver association evidence.
8. Chạy controlled action experiments.
9. Tối ưu event fusion có kiểm chứng.
10. Xử lý seatbelt uncertainty.
11. Fine-tune chỉ khi đủ điều kiện.
12. Chạy tests và benchmark.
13. Tổng hợp TP/FP/FN và per-subject metrics.
14. Viết final report bằng tiếng Việt.
15. Đề xuất model selection dựa trên đúng evidence level.

Nếu blocker xuất hiện:

- Ghi rõ blocker.
- Không tự bỏ qua data rights.
- Không dùng protected holdouts.
- Không giả lập GPU training.
- Tiếp tục các công việc độc lập có thể thực hiện được.

Khi hoàn thành, báo cáo:

A. Đã sửa những source files nào?

B. Đã chạy những experiments nào?

C. Những papers nào thực sự được áp dụng?

D. Baseline và candidate khác nhau thế nào?

E. TP/FP/FN thay đổi ra sao?

F. Event Precision/Recall/F1 đạt bao nhiêu?

G. SAFE_DRIVING Recall đạt bao nhiêu?

H. Seatbelt Macro-F1 đạt bao nhiêu?

I. FA/min đạt bao nhiêu?

J. Full-system FPS là bao nhiêu?

K. Có independent holdout chưa?

L. Có đủ điều kiện production approval chưa?

M. Những vấn đề nào cần dữ liệu hoặc GPU mới giải quyết được?

N. Đề xuất nghiên cứu tiếp theo có giá trị cao nhất là gì?

# NGUYÊN TẮC CUỐI CÙNG

Mục tiêu không phải làm cho báo cáo xuất hiện con số 85%.

Mục tiêu là xây dựng mô hình AI thật sự phát hiện được hành vi vi phạm, giảm false alarms, tổng quát hóa trên người lái chưa từng thấy và có bằng chứng khoa học để bảo vệ đồ án tốt nghiệp.

HƯỚNG NGHIÊN CỨU CHÍNH:

**BEHAVIOR-CENTRIC REPRESENTATION + PHONE/HAND/FACE ASSOCIATION + VIDEO ACTION LEARNING + CALIBRATED FUSION + HUMAN-APPROVED DATA + INDEPENDENT EVALUATION.**

Ưu tiên cải thiện tính chất của đặc trưng và dữ liệu trước khi tăng thêm độ phức tạp thuật toán.

Triển khai tối đa công việc khả thi trong phiên làm việc hiện tại. Báo cáo chính xác những gì thực sự chạy được và những gì còn bị chặn.

**BẮT ĐẦU TỪ REPOSITORY AUDIT, DATA GOVERNANCE VÀ REPRESENTATION ROOT-CAUSE ANALYSIS.**

dùng kaggle
API Token is now available
Please copy your API token now. You won't be able to view it again. You can always generate a new one if needed.
<REDACTED_KAGGLE_API_TOKEN>
API Token

To use this token, set the KAGGLE_API_TOKEN environment variable:

export KAGGLE_API_TOKEN=<YOUR_KAGGLE_API_TOKEN>
Or save it to ~/.kaggle/access_token, where the client will read it automatically:

mkdir -p ~/.kaggle && echo <YOUR_KAGGLE_API_TOKEN> > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token
Then you can use the client as follows:

kaggle competitions list
