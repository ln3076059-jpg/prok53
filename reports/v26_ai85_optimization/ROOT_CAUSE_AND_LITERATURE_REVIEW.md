# Vì sao ROADWATCH chưa đạt 85% và học gì từ nghiên cứu liên quan?

Ngày kiểm tra: 21/10/2026. Phạm vi: code, trace, checkpoint manifest và kết quả phát hiện sự kiện điện thoại trên gC-14/gZ-36; đối chiếu nguồn khoa học gốc. Đây là tập development đã được xem nhiều lần, chưa phải kiểm định độc lập.

Đối chiếu mở rộng với paper, GitHub chính chủ, mức tương ứng của metric và thứ tự thử nghiệm nằm tại [RESEARCH_GITHUB_GAP_ANALYSIS.md](RESEARCH_GITHUB_GAP_ANALYSIS.md); dữ liệu máy đọc được nằm tại [research_github_matrix.json](research_github_matrix.json).

Kết luận: điểm thấp đến từ cả lỗi tích hợp, thiếu bằng chứng điện thoại/association và cách tạo khoảng thời gian sự kiện. Đồng thời, các con số 92–98% trong nghiên cứu thường đo phân loại ảnh/clip, khác với precision/recall/F1 của sự kiện trên video dài. Chưa có cơ sở kết luận chỉ thay backbone hoặc tăng GPU sẽ đạt 85%.

## 1. Các kết quả phải tách riêng

| Phép đo | TP / FP / FN | Precision | Recall | F1 | Ý nghĩa |
|---|---:|---:|---:|---:|---|
| V26 E2, cache D5, không action head | 12 / 13 / 8 | 48.00% | 60.00% | 53.33% | Tốt nhất trong E2–E5; 20 GT của hai người |
| V26 E5, cache D5, hybrid action | 12 / 14 / 8 | 46.15% | 60.00% | 52.17% | Action head chưa đem lại lợi ích sự kiện ở cấu hình này |
| Raw D2 trước sửa seat/pose, hai người | 0 / 0 / 20 | 0 theo quy ước evaluator | 0% | 0% | 262 ứng viên đều association UNKNOWN |
| Raw D2 sau sửa seat/pose, chỉ gZ-36 | 4 / 4 / 7 | 50.00% | 36.36% | 42.11% | Hoàn tất gZ; gC tương ứng chưa hoàn tất; không gộp với gC bản cũ |

Nguồn: `experiments/E*/metrics.json`; `experiments/V26-FULL-D2/{gc14_run,gz36_run_final,gz36_run_seatfix2}/subjects/*/metrics.json`. Tỷ lệ trong bảng được tính từ TP/FP/FN; một số raw JSON đã làm tròn sẵn.

E2 là mã thí nghiệm association/lifecycle, **không phải detector D2**. `run_verified_v24.py` đọc trace V22; `inference_summary.json` của cả hai video ghi D5. D5 dùng specialist + auxiliary COCO, trong khi các raw run trên dùng D2 specialist-only:

| Người | Ứng viên specialist trong cache D5 | Auxiliary trong cache D5 | Specialist raw D2 |
|---|---:|---:|---:|
| gC-14 | 92 | 365 | 109 |
| gZ-36 | 136 | 117 | 153 |

Các số này đếm ứng viên sau những bước export/filter tương ứng, không phải bbox recall. Không thể lấy tỷ lệ 262/710 để suy ra chất lượng detector. Cần đồng nhất detector, ROI, filtering, timestamp và temporal config trước khi kiểm tra tương đương giữa cache và raw.

## 2. Nguyên nhân có bằng chứng trực tiếp

### 2.1 Lỗi đưa pose vào association khiến lần chạy raw đầu mất toàn bộ sự kiện

Runner trước sửa tạo occupant chưa verified; association không có driver hợp lệ. Trace ghi:

- gC: 109 ứng viên UNKNOWN; 74 `NO_VERIFIED_OCCUPANT_TRACK`, 35 `UNVERIFIED_MOUNTED_CONTEXT`.
- gZ: 153 ứng viên UNKNOWN; 149 `NO_VERIFIED_OCCUPANT_TRACK`, 4 `UNVERIFIED_MOUNTED_CONTEXT`.

Sau sửa liên kết pose với seat ROI, gZ có 149 `DRIVER_ASSOCIATED` và 4 UNKNOWN, phát hiện được 4 GT. Đây là bằng chứng lỗi tích hợp gây suy giảm mạnh; chưa chứng minh việc gán vai trò đã đúng ngoài thực tế.

Giới hạn: calibration có chuỗi `VERIFIED_STATIC_CALIBRATION` nhưng chưa có ground truth driver/passenger độc lập xác nhận các tọa độ. Chọn pose có confidence cao nhất trong crop vẫn có thể sai khi nhiều người. Kiểm thử code kiểm tra hợp đồng đầu vào, không chứng minh camera đã được hiệu chuẩn vật lý. Đã bỏ fallback lấy toàn bộ crop làm bbox khi pose không có bbox.

### 2.2 Lỗi hết hạn trạng thái và lưu biên sự kiện

Cached reporter trước sửa không gọi `TemporalEventEngine.expire()`. Sự kiện còn active qua khoảng mất pose; trên gZ từng xuất hiện khoảng nối từ khoảng 52.9 đến 376.5 giây. Sau thêm expiry, E5 tăng từ TP=6, FN=14 lên TP=12, FN=8; FP đồng thời tăng từ 0 lên 14. Vì vậy precision cao trước đó không phải bằng chứng hệ thống tốt: nó bỏ sót nhiều và gộp quá mức.

Một lỗi khác trong raw reporter: lưu khoảng kích hoạt ban đầu nhưng chưa kéo dài biên theo active state. Trong raw gZ sau sửa seat, cả 8 khoảng dự đoán đều chỉ dài khoảng 1.5121 giây, dù có GT kéo dài hàng chục giây. Code hiện có `RawVideoBoundaryCollector` để cập nhật biên; regression test kiểm tra hai đoạn dương tính cách nhau bởi khoảng mất pose không bị nối. **Chưa chạy lại toàn bộ video với collector mới**, nên F1=42.11% không phải kết quả của thay đổi này. Cần kiểm tra cả release, bridging và finalization trên video thật; không bảo đảm kéo dài biên sẽ tăng điểm.

### 2.3 Phân rã 8 FN của E2 theo trace

| GT | Người; khoảng thời gian (giây) | Mẫu / mẫu có candidate / mẫu association được chấp nhận | Chẩn đoán |
|---|---|---:|---|
| GT_02 | gC; 96.0013–108.6358 | 25 / 20 / 1 | Nhiều candidate bị loại ở association/context |
| GT_05 | gC; 225.9073–226.3105 | 1 / 1 / 1 | Chỉ một mẫu bên trong GT |
| GT_10 | gZ; 52.4530–53.2594 | 1 / 1 / 1 | Chỉ một mẫu bên trong GT |
| GT_12 | gZ; 110.7191–111.3239 | 1 / 1 / 1 | Chỉ một mẫu bên trong GT |
| GT_14 | gZ; 162.0632–162.9704 | 2 / 0 / 0 | Không có candidate trong trace, không nên quy thành thiếu hai mẫu |
| GT_16 | gZ; 227.0497–248.8911 | 43 / 0 / 0 | Không có candidate trong trace trong khoảng dài |
| GT_17 | gZ; 296.3038–346.6398 | 100 / 3 / 3 | Bằng chứng association rất thưa |
| GT_19 | gZ; 400.2016–460.0134 | 119 / 2 / 2 | Bằng chứng rất thưa; không mẫu nào fusion ≥0.18 |

Stride 15 ở 29.76 fps cho khoảng cách mẫu khoảng 0.504 giây. Engine yêu cầu hai observations trải ít nhất 0.5 giây. Báo cáo cũ dùng điều kiện duration <1 giây nên phân loại sai GT_14; code audit đã sửa dùng span của các timestamp thực tế. Kết quả đúng: **3 thiếu temporal support, 2 không có exported proposal, 1 association/context, 2 sparse evidence**.

Đây là cách quy nguyên nhân ưu tiên từ log, không phải tám thí nghiệm can thiệp độc lập. Một prediction có bằng chứng ngoài GT vẫn có thể match theo protocol. Do đó không được suy ra “recall tối đa 80%” chỉ từ số GT dưới một giây. Trace cũng không chứa toàn bộ proposal trước filter để kết luận detector hoàn toàn không nhìn thấy điện thoại.

### 2.4 Phân rã 13 FP của E2

Có 25 predictions: 12 TP và 13 FP. Trong FP, 6 predictions cùng trỏ tới GT đã được đếm; 7 predictions không match theo quy tắc đã khóa. Đây là dấu hiệu cần xử lý phân mảnh/trùng sự kiện và kiểm tra biên, không chỉ giảm confidence threshold.

Matcher hiện tại dùng lựa chọn GT tốt nhất theo IoU rồi xử lý duplicate, không phải phép gán tối ưu toàn cục. Giữ nguyên matcher để so sánh; nếu nghiên cứu sensitivity với thuật toán gán khác phải báo cáo riêng, không thay kết quả chính sau khi xem điểm.

## 3. Các vấn đề đã thấy trong code nhưng cần thí nghiệm để chứng minh tác động

| Vấn đề | Bằng chứng | Thí nghiệm phân biệt nguyên nhân |
|---|---|---|
| Ít đa dạng người khi học action | Mỗi LOSO checkpoint gC/gZ chỉ có 2 training subjects; validation subjects rỗng; fold có 909/795 training samples, training accuracy cuối đạt 1.0 | Tách train/validation/test theo người và session; tăng người, báo cáo mean/std theo fold. Training 100% chưa tự chứng minh overfit |
| Mất chi tiết điện thoại/tay | `action_adapter.py`: crop giảm xuống 64×64 rồi nội suy lên 112×112 | So backbone/branch cùng split với input gốc 224/336 và local hand/face/phone crop; retrain phù hợp preprocessing, không nạp checkpoint cũ sai hợp đồng |
| Action score không đủ hữu ích cho event | E2 F1 53.33%; E4/E5 52.17% | Đo class confusion và calibration trước fusion; ablation no-action / late-fusion / gating trên validation |
| Temporal input và latency | 9 embedding cách nhau 0.5 giây, label ở giữa | Phân biệt causal với centered có khoảng 2 giây lookahead; báo cáo độ trễ thực, không gọi centered là realtime zero-latency |
| ROI/proximity và pose che khuất | Heuristic pixel radius/seat ROI; GT dài vẫn rất ít candidate | Thêm bbox/role GT độc lập, log full-frame và ROI, pose và lý do loại trước/sau từng bước |
| Seatbelt thiếu đánh giá tương đương | Gate hiện có macro-F1 ba lớp khoảng 0.572753; thiếu exhaustive bbox GT | Đánh giá riêng detector và classifier ba lớp, có uncertain/occluded và person-disjoint split |

## 4. Nghiên cứu liên quan có điểm cao — đọc đúng phép đo

| Nghiên cứu gốc | Số liệu và protocol | Bài học áp dụng |
|---|---|---|
| Ortega et al., DMD, ECCV Workshops 2020 | Table 4: MobileNetV2 MaxP-MLP RGB, 4 segments, **92.0% test Top-1**; IR+Depth 8 segments **93.9%**. Clip khoảng 50 frames; split 24/7/6 drivers. | Baseline nhẹ và temporal consensus đáng tái lập; kết quả phân loại clip không đo event boundary. [PDF](https://arxiv.org/pdf/2008.12085) |
| Hasan et al., DriveCLIP, IEEE T-ITS 2024 | Tables III–IV: DMD VideoCLIP **98.44% Top-1**, ViT-L/14@336px, 1 fps, có held-out driver sets; multi-frame **93.39%**, single-frame **82.65%**, zero-shot **48.12%**. | Thử representation pretrained và temporal learning. **98.44% không phải zero-shot** và không phải event-F1. [PDF](https://arxiv.org/pdf/2306.10159) |
| Eraqi et al., AUC distracted driving, 2019 | Table 1: random-image split, ensemble **95.98%**, InceptionV3 **95.17%**. Table 4: unique-driver test, InceptionV3 **90.068%**. | Cùng InceptionV3 chênh 5.102 điểm phần trăm khi đổi protocol; không dùng random-frame score để chứng minh khả năng tổng quát sang người mới. [PDF](https://arxiv.org/pdf/1901.09097) |
| Qiu et al., seatbelt RST-Net, Applied Sciences 2024 | Table 4 báo cáo **98.25% accuracy/F1**, bài toán seatbelt nhị phân trên dữ liệu tự xây dựng. Chưa xác nhận person-disjoint split trong lần đối chiếu này. | Thử tách localization và classification; không so trực tiếp với macro-F1 ba lớp có uncertain. [Bài gốc](https://www.mdpi.com/2076-3417/14/8/3346) |

Không chuyển số liệu của các bài này thành mục tiêu chắc chắn cho hệ thống. Cần tái lập trên cùng người, camera, nhãn, đầu vào, độ trễ và evaluator. Riêng DriveCLIP, Table IX còn báo single-frame mean F1 DMD 0.819: một headline cao không có nghĩa mọi cấu hình/mọi lớp đều vượt 85%.

Các hướng bổ sung có cơ sở nhưng chưa kiểm chứng ở ROADWATCH:

- **SAHI, ICIP 2022**: thử slicing cho điện thoại nhỏ. Bài báo báo tăng AP inference 6.8/5.1/5.3 điểm cho ba detector trên benchmark của họ; mức 12.7/13.4/14.5 có thêm fine-tuning. Không được gộp thành “chỉ inference tăng 6–14%”. DOI đúng: `10.1109/ICIP46576.2022.9897990`; ma trận cũ trong repo ghi sai hậu tố. [Nguồn](https://arxiv.org/abs/2202.06934).
- **ActionFormer, ECCV 2022**: học phân loại và hồi quy biên theo thời gian, phù hợp câu hỏi event localization. Báo cáo 71.0% mAP@tIoU0.5 trên THUMOS14, không phải cam kết F1 điện thoại >85% hay bằng chứng xử lý tốt mọi sự kiện dưới một giây. [Nguồn](https://arxiv.org/abs/2202.07925).
- **DualScope, AAAI 2026**: hướng global/local vùng hành vi và temporal modeling nhiều thang. Trang proceedings xác nhận phương pháp; chưa xác minh đủ full-text split và tái lập code trong lần kiểm tra này, nên không dùng con số headline làm gate. [Nguồn chính thức](https://ojs.aaai.org/index.php/AAAI/article/view/37812).

## 5. Thứ tự thực nghiệm để tiến tới >85%

1. **Khóa so sánh đúng**: cùng video/stride/D5/checkpoint/ROI/filter/temporal config; ghi hash. Hoàn tất raw cả gC và gZ cùng một phiên bản. Log Observation, pose, pre/post-filter candidates, association và state transition để replay không cần suy đoán.
2. **Sửa và kiểm tra biên**: dùng collector đã có, kiểm tra active/release/expiry/cooldown; lượng hóa 6 duplicate FP và 7 unmatched FP. Không thay GT hay matcher để tăng điểm. So sánh từng thay đổi riêng trên cùng đầu vào.
3. **Tăng khả năng quan sát**: ablation stride 15/6/3, ROI/high-resolution/slicing và D2/D5. Chốt grid trên development/validation trước; đo candidate recall bằng bbox GT, latency và FP, không chỉ event recall.
4. **Cải thiện action có kiểm soát**: baseline MobileNetV2 temporal consensus, sau đó CLIP embeddings + temporal head, local/global crops. Tách người giữa train/validation/test, kiểm tra phonecall/texting/safe-driving/other; giữ preprocessing và latency đúng với checkpoint.
5. **Học temporal/fusion khi có nhãn phù hợp**: thử transition-aware head/temporal localization, hard negatives (GPS, vật giống điện thoại, tay gần mặt), đánh giá các GT ngắn riêng. Chỉ chọn cấu hình trên validation; không chọn theo holdout.
6. **Kết luận trên holdout mới**: báo cáo event P/R/F1, bbox AP/recall, action macro-F1, seatbelt macro-F1, FP/phút, độ trễ, số người/sự kiện và khoảng bất định theo người. Không gộp các metric khác bản chất thành một “điểm AI”.

Trên đúng 20 GT development, muốn P/R/F1 đều **>0.85** cần **TP≥18, FN≤2 và FP≤3**. So E2 hiện tại, ít nhất phải tăng thêm 6 TP và giảm 10 FP tại TP=18. Đây chỉ là điều kiện đếm trên tập hiện tại; đạt nó chưa chứng minh tổng quát hóa.

## 6. Artifacts, tái lập và trạng thái thực thi

- Chạy `python -m training.dmd.diagnose_v26_gap` để tạo [diagnosis/evidence.json](diagnosis/evidence.json): ledger từng GT, detector sources, training manifest, số liệu raw và SHA-256 đầu vào.
- Chạy `python -m training.dmd.audit_v26_optimization` để cập nhật ledger và báo cáo tổng hợp; log gốc của các thí nghiệm được giữ nguyên. [fn_root_cause_ledger.json](fn_root_cause_ledger.json), [fp_ledger.json](fp_ledger.json), [FINAL_AI_LEAD_DECISION.md](FINAL_AI_LEAD_DECISION.md).
- Test regression bổ sung bảo vệ GT dưới một giây nhưng có đủ hai mẫu, và trường hợp pose không có bbox. Collector có test kéo dài biên và tách qua khoảng mất pose; xem [test_results.json](test_results.json) cho kết quả suite hiện tại.
- Dataset Kaggle private đã nhận phiên bản code đồng bộ với audit hiện tại theo [receipt](kaggle_upload_receipt.json): 23/23 tệp, 277.050 byte, giữ cây `package/...`. Đây không phải bằng chứng đã chạy GPU/train hay đạt >85%. Gói không chứa video, checkpoint, raw prediction hoặc nhãn gốc.
- Governance live cho phép diagnostics và code-only upload; quyền fitting/labels/group split chưa đầy đủ trong record. Đây là giới hạn đối với các bước huấn luyện tiếp theo, không cản việc phân tích code/log đã thực hiện. Không gán trạng thái quyền hiện tại thành lỗi `RIGHTS_ACCEPTED=false` của báo cáo cũ.

**Trạng thái:** đã định vị và lượng hóa nhiều nguyên nhân; chưa có kết quả end-to-end độc lập >85%. Báo cáo này sửa các diễn giải sai về D2/D5, GT_14 và tính tương đương của metric trong tài liệu cũ.
