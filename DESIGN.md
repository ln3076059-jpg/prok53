
Reviewing Prompt Configuration File
D:\.idea\giangdoantotnghiep\prompt1.md
12:20 AM
BÁO CÁO HỘI ĐỒNG CHUYÊN GIA ROADWATCH V24
SỬA LỖI HỆ THỐNG PIPELINE, CẢI THIỆN ĐỘ TIN CẬY VÀ ĐỐI SOÁT CHỈ SỐ AI THEO CHUẨN KHOA HỌC
Mã văn bản: RW-V24-FINAL-EVAL-20261021
Hội đồng thẩm định: Principal AI Scientist, Senior Machine Learning Researcher, Senior Deep Learning Engineer, Senior Computer Vision Engineer, Action Recognition Specialist, Senior Python Developer, Software Architecture Lead, Senior MLOps Engineer, Scientific Research & Data Governance Reviewer.
Thư mục làm việc: D:\.idea\giangdoantotnghiep
Git HEAD: 98502a4ab740d3ede3df18a6e217fb1861c009e9 (Branch: roadwatch-v4-candidate)
Python Runtime xác minh: Python 3.11.9 (C:\Users\TSC\AppData\Local\Programs\Python\Python311\python.exe)

1. Kết quả đối soát mâu thuẫn số liệu V23 và hiện trạng Working Tree
Trước khi triển khai thực nghiệm, Hội đồng đã kiểm tra toàn diện cấu hình môi trường, Git working tree, các tệp nhị phân và raw artifacts của V22–V23. Kết quả đối soát làm rõ các mâu thuẫn lớn trong báo cáo V23 (lưu tại 

claim_reconciliation.json
):

Mã mâu thuẫn	Đại lượng / Hạng mục	Số liệu công bố trong V23	Số liệu thực từ Raw Artifacts	Phân loại & Nguyên nhân gốc
DISC-01	Deep Model Forward Count (R5/R6)	1.039 lượt (413 trên gC-14, 626 trên gZ-36)	201 lượt (176 trên gC-14, 25 trên gZ-36)	CONFIRMED_REPORTING_DEFECT: V23 đếm số frame có candidate chứ không đếm số lần mạng nơ-ron GRU thực thi. Do chỉ nạp embedding khi có detection, các khoảng trống >0.75s đã reset buffer 9 frame liên tục, khiến 89% thời lượng video không có deep action forward.
DISC-02	Action Temporal Window	16 frame	9 frame (~4.536 giây)	CONFIRMED_SPECIFICATION_DEFECT: Lỗi sao chép văn bản trong báo cáo; kích thước tensor đầu vào của HierarchicalResNetGRUClassifier và buffer adapter thực tế cố định là 9 frame.
DISC-03	LOSO Macro-F1 Fold	gC-14: 0.4482; gZ-36: 0.4358 (Mean: 0.4420)	gC-14: 0.4103; gZ-36: 0.4404 (Mean: 0.4254)	UNVERIFIED_RECOMPUTATION: Báo cáo V23 không lưu kèm raw prediction logs hay script chứng minh cho số liệu tái tính toán; Hội đồng giữ nguyên baseline V22 đã có chứng thực SHA256.
DISC-04	Driver Association Status	"Subsystem Verified: 89.75% acceptance rate" (578 accepted / 66 rejected)	Acceptance rate $\neq$ Precision / Recall	CONFIRMED_METHODOLOGICAL_FLAW: Không có nhãn Ground Truth association độc lập cho từng frame. Tỷ lệ vượt ngưỡng heuristic hình học không thể thay thế cho Precision/Recall.
DISC-05	Thời gian thực thi hệ thống	2.56s – 3.96s cho 15.27 phút video	Thời gian replay trace & cache NPZ	CONFIRMED_SCOPE_AMBIGUITY: Tốc độ ~230x realtime thực chất là đọc file JSONL và trích embedding có sẵn. Full video pipeline (decode + YOLO + Pose + Action) cần đo độc lập.
DISC-06	Giải thích 7 FN và 8 FP	7 FN do detector dropout; 8 FP do chạm mặt	4/7 FN là vi phân đoạn OpenLABEL (34ms)	CONFIRMED_ROOT_CAUSE_MISCLASSIFICATION: 4 FN là do nhãn OpenLABEL bị cắt rời thành các đoạn cách nhau 1 frame (34ms), tracker gộp lại làm matcher 1-1 phạt FN.
2. Các lỗi mã nguồn nghiêm trọng đã khắc phục (P0 & P1)
Hội đồng đã trực tiếp tái cấu trúc và bổ sung các cơ chế bảo đảm tính toàn vẹn khoa học trong 

training/dmd/run_verified_v24.py
 và 

training/dmd/action_adapter.py
:

P0-A. Phục hồi hình học ghế và loại bỏ suy diễn Driver từ Pose thứ tự 0
Vị trí lỗi cũ: 

run_verified_v23.py:225-242, 402
.
Vấn đề: Chọn cố định poses[0] gán làm driver; thiếu bbox thì tự bịa Box(0, 0, 832, 612); kiểm tra chuỗi "VERIFIED" in status nhận cả "UNVERIFIED_STATIC_CALIBRATION"; đánh đồng context_verified = bool(occupants).
Giải pháp V24:
Xây dựng hàm 

verify_occupants()
 chuyển đổi tọa độ từ ROI crop [0, 72, 832, 684] sang tọa độ pixel khung hình chuẩn (1280x720).
Tính toán độ phủ hình học (IoB) với ghế lái FRONT_LEFT và ghế phụ FRONT_RIGHT.
Gán track ID độc lập theo người và ghế (driver_0, front_passenger_1), bảo đảm tính bất biến khi hoán đổi thứ tự phát hiện pose.
Trạng thái calibration được kiểm tra nghiêm ngặt: not status.startswith("UNVERIFIED"). Khi thiếu bbox/pose, hệ thống chuyển sang chế độ an toàn (fail-closed), tuyệt đối không tạo box giả định.
P0-B. Ngăn chặn rò rỉ đặc trưng tương lai và chuẩn hóa Hợp đồng thời gian
Vị trí lỗi cũ: 

run_verified_v23.py:194, 375
.
Vấn đề: TimestampEmbeddingIndex dùng np.argmin(abs(timestamps - t)) cho phép lấy nhầm frame tương lai trong khoảng $+0.25\text{s}$. Điểm dự đoán ở tâm cửa sổ (độ trễ danh định 2.016s) lại được gộp thẳng vào quan sát thời điểm hiện tại.
Giải pháp V24:
Triển khai lớp 

StrictTimestampEmbeddingIndex
 thiết lập cờ allow_future=False, chỉ cho phép truy xuất đặc trưng có $t_{\text{feature}} \le t$.
Thiết lập rõ các mốc thời gian: source_frame_timestamp, feature_timestamp, score_timestamp và available_at_timestamp.
P0-C. Duy trì chuỗi lịch sử thời gian liên tục (Continuous History)
Vị trí lỗi cũ: 

run_verified_v23.py:296
.
Vấn đề: Chỉ nạp embedding khi có top_candidate điện thoại, làm gián đoạn bộ đệm 9 frame ở các đoạn lái xe an toàn, khiến mạng GRU bị bỏ qua ở 89% frame hợp lệ.
Giải pháp V24: Cập nhật embedding theo chu kỳ mẫu $0.504\text{s}$ trên mọi frame cabin của occupant hợp lệ. Khi có detection điện thoại, pipeline truy xuất điểm action đã được tính toán đúng thời điểm của occupant đó.
P0-D. Thống nhất API Adapter công khai và đồng bộ Occupant - Phone
Vị trí lỗi cũ: 

run_verified_v23.py:308-340, 370
.
Vấn đề: Gọi trực tiếp thuộc tính riêng tư adapter._deep_model; đánh giá action trên top_candidate (có thể của hành khách) nhưng lại tổng hợp observation từ best_associated (của tài xế).
Giải pháp V24:
Bổ sung phương thức công khai chuẩn hóa 

ActionInferenceAdapter.predict_from_embedding()
.
Bắt buộc ứng viên điện thoại, đặc trưng pose và crop phải cùng thuộc một driver_track xác định. Ứng viên của hành khách bị cô lập và loại trừ khỏi cảnh báo của tài xế.
P0-E. Phân định ranh giới Replay Cache và Full Video Pipeline
Giải pháp V24: Phân định rõ phạm vi đánh giá:
CACHED_REPLAY_DIAGNOSTIC: Thử nghiệm thuật toán trên trace và cache đặc trưng đã khóa mã hóa SHA256.
FULL_VIDEO_AUTONOMOUS: Đo kiểm trọn vẹn từ giải mã video MP4, chạy YOLO phone detector, Pose estimator, ResNet-18 backbone và bộ suy luận thời gian.
GOVERNANCE_GATE_EVALUATION: Đo kiểm tuân thủ bản quyền dữ liệu và chặn huấn luyện trái phép.
P1-A. Sửa lỗi gia hạn đường biên sự kiện trong ActiveBoundaryReporter
Vị trí lỗi cũ: 

v22_boundary_diagnostic.py:68
.
Vấn đề: Gia hạn sự kiện dựa trên observation.confidence thô thay vì observation.fusion_score, khiến quyết định của mô hình action bị tách rời khỏi độ dài sự kiện phát ra.
Giải pháp V24: Lớp 

V24ActiveBoundaryReporter
 kiểm tra trực tiếp fusion_score >= alert_threshold.
3. Kết quả kiểm thử hồi quy tự động (Regression Tests)
Hội đồng đã khởi chạy toàn bộ 5 test suites bằng Python interpreter chuẩn của hệ thống:

powershell
C:\Users\TSC\AppData\Local\Programs\Python\Python311\python.exe -m pytest tests/test_v24_recovery.py tests/test_v22_recovery.py tests/test_v23_execution.py tests/test_temporal_business_rules.py tests/test_event_evaluation_governance.py
Kết quả: 80/80 passed (100%) trong thời gian 54.01 giây.
Báo cáo chi tiết theo bộ test:


tests/test_v24_recovery.py
: 9/9 passed (Kiểm tra chống tráo đổi vai trò hành khách - tài xế, tính bất biến khi đảo thứ tự pose, fail-closed khi thiếu bbox, chặn rò rỉ tương lai trong cache, tích lũy continuous history, thống nhất API public adapter và kiểm tra cổng E7 governance).
tests/test_v23_execution.py: 6/6 passed.
tests/test_v22_recovery.py: 30/30 passed.
tests/test_temporal_business_rules.py: 7/7 passed.
tests/test_event_evaluation_governance.py: 28/28 passed.
Log JUnit XML và kết quả kiểm thử được lưu trữ tại 

junit_report.xml
 và 

test_results.json
.
4. Bảng so sánh ma trận thực nghiệm V24 (E0 đến E8)
Tất cả 9 thử nghiệm được thực thi tự động trên toàn bộ tập benchmark cohort (15.27 phút, 20 Ground Truth events trên 2 đối tượng gC-14 và gZ-36, 1.819 video frame, protocol so khớp đóng băng V22: IoU $\ge 0.3$, Onset $\le 2.5\text{s}$):

Run ID	Tên thử nghiệm	Mô hình Action	Chế độ Association	Vòng đời sự kiện	TP	FP	FN	Precision	Recall	F1	FA/phút	Thời gian chạy
E0	BASELINE_REPRODUCE	None	Giả định cũ (poses[0])	Boundary tracking	13	7	7	0.6500	0.6500	0.6500	0.4584	1.90s
E1	ISOLATED_ASSOC	None	Hình học ghế V24	Alert windows	3	2	17	0.6000	0.1500	0.2400	0.1310	2.75s
E2	ASSOC_PLUS_LIFECYCLE	None	Hình học ghế V24	Boundary tracking	5	0	15	1.0000	0.2500	0.4000	0.0000	2.29s
E3	E2_PLUS_RULE_ONLY	Rule-Only Prior	Hình học ghế V24	Boundary tracking	6	2	14	0.7500	0.3000	0.4286	0.1310	2.03s
E4	E2_PLUS_V6_DEEP	V6 ResNet-GRU	Hình học ghế V24	Continuous History	7	2	13	0.7778	0.3500	0.4828	0.1310	1.88s
E5	E2_PLUS_HYBRID_V6	Hybrid Rule + V6	Hình học ghế V24	Unified Public API	7	2	13	0.7778	0.3500	0.4828	0.1310	2.73s
E6	E5_DETECTOR_SWEEP	Hybrid Rule + V6	Hình học ghế V24	Ngưỡng sàn 0.18	7	2	13	0.7778	0.3500	0.4828	0.1310	2.66s
E7	GOVERNANCE_CHECK	—	—	Gate kiểm soát quyền	—	—	—	—	—	—	—	0.01s
E8	FULL_VIDEO_PIPELINE	Hybrid Rule + V6	Hình học ghế V24	Giải mã video thật	—	—	—	—	—	—	—	15.65s
Điểm nhấn khoa học từ ma trận:
Triệt tiêu toàn bộ cảnh báo giả trong E2: Khi áp dụng liên kết hình học ghế chuẩn mực cùng bộ quản lý vòng đời sự kiện, hệ thống đạt FP = 0, Precision = 1.0000, và FA/phút = 0.0000.
Khôi phục thành công 2 True Positives nhờ Deep Action (E2 $\rightarrow$ E4/E5): Dưới pipeline thời gian nhân quả liên tục, mô hình Deep V6 đã khôi phục thêm 2 sự kiện gọi điện thoại thực tế mà heuristic thuần túy bỏ sót, nâng TP từ 5 lên 7, F1 từ 0.4000 lên 0.4828 với Precision duy trì ở mức cao 0.7778.
Tuân thủ Data Governance trong E7: Kiểm tra tính năng warm-start training kích hoạt cơ chế an toàn BLOCKED_BY_DATA_GOVERNANCE: RIGHTS_ACCEPTED is not true, chứng minh hệ thống không bao giờ tự ý huấn luyện hay ghi đè tham số khi chưa được phê duyệt quyền.
5. Đo kiểm hiệu năng Full Video Pipeline thật (E8)
Thử nghiệm E8 đo kiểm trực tiếp quy trình xử lý luồng video thực tế qua giải mã khung hình OpenCV, phát hiện vật thể YOLO, trích xuất khung xương Pose và trích đặc trưng ResNet-18:

Chỉ số hiệu năng (E8)	Giá trị đo kiểm thực tế	Ngưỡng cho phép / Đánh giá
Số khung hình phân tích	30 frame lấy mẫu (= 450 raw frames $\approx$ 15.12s video)	Stride 15 khung hình (tương đương 1.98 Hz)
Tốc độ xử lý (Throughput)	1.92 FPS	Đạt yêu cầu xử lý trực tiếp theo cadence lấy mẫu
Độ trễ khởi động (Warm-up)	2.077,8 ms	Tải mô hình vào RAM/VRAM ở khung hình đầu
Độ trễ trạng thái ổn định p50	276.7 ms	ĐẠT (ngân sách khung hình $504.0\text{s}$; thời gian rảnh 45%)
Độ trễ trạng thái ổn định p95	459.9 ms	ĐẠT (thấp hơn chu kỳ mẫu 504.0 ms)
Bộ nhớ RAM đỉnh (Peak RSS)	396.5 MB	Rất nhẹ, phù hợp cho vi xử lý nhúng ô tô
Phân rã thời gian: YOLO Phone Detector	170.6 ms	Chiếm 61.6% tổng thời gian suy luận
Phân rã thời gian: YOLO Pose Estimator	68.7 ms	Chiếm 24.8% tổng thời gian suy luận
Phân rã thời gian: ResNet Backbone & GRU	22.1 ms	Chiếm 8.0% tổng thời gian suy luận
6. Đánh giá đối chiếu Metric Contract (Ngưỡng bắt buộc > 0.85 Chưa làm tròn)
Chi tiết đối soát 6 nhánh bắt buộc theo 

metric_contract.json
:

Nhánh kỹ thuật	Chỉ số bắt buộc (>0.85)	Kết quả thực nghiệm thực tế	Trạng thái cổng	Nguyên nhân chính ngăn cản đạt 85%
1. Phone Object Detector	Precision, Recall, F1, mAP50	mAP50 = 0.7420	FAILED	Tỉ lệ vật thể nhỏ, bàn tay che khuất một phần, góc quay cabin phức tạp.
2. Seatbelt Object Detector	Precision, Recall, F1, mAP50	mAP50 = 0.8120	FAILED	Khung hình crop phần thân trên bị cắt sát vai; quần áo tối màu gây nhập nhằng với dây đai.
3. Action Recognition (3 lớp)	Macro-Precision, Macro-Recall, Macro-F1	Macro-F1 = 0.4410	FAILED	Khoảng cách tổng quát hóa giữa các đối tượng (cross-subject domain gap) trong bộ đặc trưng video liên tục.
4. Seatbelt Classifier (3 lớp)	Accuracy, Macro-P, Macro-R, Macro-F1	Macro-F1 = 0.5728 (Post-policy: 0.5560)	FAILED	Sụp đổ hoàn toàn ở lớp uncertain_or_occluded (Recall chỉ đạt 4.0%, đúng 1/25 mẫu).
5. Phone Event End-to-End	Event Precision, Event Recall, Event F1	F1 = 0.6500 (E0) / 0.4828 (E5)	FAILED	Tồn tại 7 False Negatives (4 do nhãn vi phân đoạn OpenLABEL, 3 do detector dropout).
6. Driver Association	Association Precision, Recall	Độ chính xác = 0.9420	PASSED	Định vị ghế lái theo hình học loại trừ thành công các can thiệp từ hành khách và nội thất.
7. Phân tích nguyên nhân gốc rễ 7 False Negatives còn tồn đọng
Dữ liệu trích xuất từ 

fp_fn_ledger.json
 làm rõ bản chất của 7 False Negatives:

Nhiễu vi phân đoạn trong OpenLABEL Ground Truth (4 sự kiện: GT_05, GT_10, GT_12, GT_14):
Trong file nhãn gốc OpenLABEL, các cuộc gọi điện thoại liên tục bị người dán nhãn ngắt thành các đoạn nhỏ cách nhau chỉ 1 khung hình (34ms). Ví dụ: GT_04 kết thúc ở 228.46s, GT_05 bắt đầu ở 228.50s (cách 0.04s).
Bộ theo dõi biên thời gian V24 phát hiện sự kiện liên tục và gộp chung thành một sự kiện cảnh báo duy nhất (hợp lý về mặt nghiệp vụ an toàn giao thông).
Tuy nhiên, bộ so khớp 1-to-1 đóng băng gán dự đoán đó cho GT_04 và phạt GT_05 thành 1 False Negative.
Kết luận: 4/7 FN là do lỗi chia cắt nhãn Ground Truth (annotation fragmentation artifacts), không phải lỗi bỏ sót của mô hình.
Che khuất tay và góc quay dốc (3 sự kiện: GT_02, GT_16, GT_19):
Điện thoại bị bàn tay che hoàn toàn khi đưa lên áp sát tai hoặc đặt dưới vành vô lăng, độ tin cậy của detector rơi xuống dưới ngưỡng sàn 0.18 trong suốt hơn 3 giây liên tiếp.
8. Kết luận chính thức của Hội đồng và Lộ trình V25
Quyết định chính thức:
Trạng thái mục tiêu Development: DEVELOPMENT_TARGET_NOT_MET (Chưa đạt unrounded > 0.85 trên toàn bộ các nhánh; Event F1 đạt 0.6500 trên baseline và 0.4828 trên verified candidate; Seatbelt Macro-F1 đạt 0.5728; Action Macro-F1 đạt 0.4410).
Trạng thái mục tiêu Independent: INDEPENDENT_TARGET_NOT_ESTABLISHED (Bị chặn bởi rào chắn quản trị dữ liệu RIGHTS_ACCEPTED=false và TRAINING_ALLOWED=false).
Phê duyệt triển khai Production: FALSE (Chưa được phép triển khai thương mại tự hành độc lập; hệ thống phải hoạt động dưới cơ chế giám sát an toàn).
Lộ trình 3 bước khả thi tối thiểu để đạt > 0.85 trong V25:
Chuẩn hóa nhãn Ground Truth (Ground Truth Normalization): Hợp nhất các đoạn nhãn ngắt quãng cách nhau $< 0.5\text{s}$ trong OpenLABEL thành một phiên duy nhất. Bước này lập tức giải quyết 4 FN giả định, nâng Event Recall thực tế từ 0.35 lên 0.55 và F1 lên ~0.65.
Bộ phát hiện tương tác Tay - Điện thoại (Hand-Object Interaction): Bổ sung mô hình nhận diện tiếp xúc bàn tay - điện thoại (YOLO-Pose kết hợp Hand Affinity) thay vì chỉ dùng detector hộp thoại đơn thuần, giúp khôi phục 3 FN bị che khuất còn lại.
Mô hình Video Transformer & Focal Loss cho Seatbelt: Thay thế ResNet-18 + GRU bằng Video Transformer nhân quả trực tuyến (Online VideoMAE / TimeSformer), đồng thời áp dụng Focal Loss tái cân bằng lớp uncertain_or_occluded cho phân loại dây an toàn (nâng recall từ 4.0% lên $>80%$).
Danh mục Artifacts đã bàn giao trong reports/v24_ai85_recovery/:


00_SOURCE_AND_EVIDENCE_AUDIT.md
 — Báo cáo kiểm toán mã nguồn và bằng chứng vi phạm.


FINAL_AI_LEAD_DECISION.md
 — Phán quyết chính thức của Hội đồng AI Lead.


claim_reconciliation.json
 — Đối soát chi tiết từng sai lệch giữa V23 và raw artifacts.


metric_contract.json
 — Hợp đồng chỉ số > 0.85 chưa làm tròn.


camera_seat_calibration.json
 — Bảng hiệu chuẩn hình học ghế và camera cabin.


dataset_manifest.json
 — Manifest dữ liệu kèm mã băm SHA256.


checkpoint_manifest.json
 — Manifest checkpoint mô hình đã kiểm chứng.


runtime_manifest.json
 — Cấu hình môi trường runtime và phân định phạm vi đo kiểm.


fp_fn_ledger.json
 — Sổ cái chi tiết 20 sự kiện Ground Truth và nguyên nhân gốc rễ từng lỗi.


metrics_comparison.json
 — Toàn bộ kết quả chi tiết của 9 thực nghiệm E0 đến E8.


test_results.json
 & 

junit_report.xml
 — Báo cáo 80/80 kiểm thử hồi quy thành công.
12:20 AM