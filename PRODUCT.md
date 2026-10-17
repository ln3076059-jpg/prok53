# ROADWATCH V12 — HOÀN THIỆN PIPELINE VÀ KIỂM CHỨNG MỤC TIÊU TRÊN 85%

Bạn là Principal Computer Vision Engineer, AI Research Lead và MLOps Engineer.

Hãy trực tiếp kiểm tra, sửa mã, chạy thí nghiệm và hoàn thiện Roadwatch trong phạm vi dữ liệu, nhãn và tài nguyên được phép. Mục tiêu là cải thiện chất lượng đo được của toàn pipeline và nghiên cứu khả năng vượt 85%, không chỉ tạo thêm báo cáo.

Không được cam kết đạt 85% trước khi thực nghiệm. Không sửa ground truth, matcher, denominator, nhãn hoặc phạm vi đánh giá để làm đẹp chỉ số. Một kết quả chưa đạt phải được báo đúng là chưa đạt.

## 1. Repository và điểm bắt đầu

- Repository: https://github.com/ln3076059-jpg/prok53
- Workspace dự kiến: `D:\.idea\giangdoantotnghiep`.
- HEAD đã kiểm tra khi soạn prompt: `4b19c4834b9520df2f72caca6c6a3fc9606cbdde`.
- Branch tại thời điểm soạn: `roadwatch-v4-candidate`.

Trước khi triển khai, đọc `AGENTS.md` nếu có, kiểm tra HEAD, branch, working tree và kế thừa mọi commit mới hơn. Không reset các thay đổi của người dùng. Các đường dẫn trong prompt là điểm bắt đầu; xác minh lại trước khi sử dụng.

Đây là prompt cho giai đoạn V12. Không chỉnh lại báo cáo V9–V11 để biến kết quả lịch sử thành kết quả mới. Không tự push, merge, activate production model hoặc xóa media/checkpoint lịch sử.

## 2. Mục tiêu và định nghĩa thành công

Đánh giá từng nhiệm vụ bằng metric tương ứng, trên dữ liệu phù hợp với nhiệm vụ đó:

| Nhiệm vụ | Mục tiêu nghiên cứu |
| --- | --- |
| Phone-use event Precision | > 85% |
| Phone-use event Recall | > 85% |
| Phone-use event F1 | > 85% |
| False alarms/min | <= 0.15 |
| SAFE_DRIVING Recall | > 85% |
| SAFE/PHONE/OTHER clip Macro-F1 | > 85% |
| Seatbelt ba trạng thái Macro-F1 | > 85% |
| Seatbelt UNCERTAIN_OR_OCCLUDED Recall | > 85% |
| Phone AP50, nếu có bộ test detection đầy đủ | > 85% |

Báo cáo thêm balanced accuracy, per-class Precision/Recall/F1, AP50-95, per-subject metrics, worst-subject metrics, thời gian cảnh báo và throughput. AP, clip F1, Event F1 và Accuracy là những phép đo khác nhau; không dùng thay thế cho nhau.

Tỷ lệ báo động sai và latency phải giảm, không áp dụng yêu cầu “trên 85%” cho những đại lượng này. Threshold và abstention policy phải được chọn trên validation rồi khóa trước test. Báo cáo cả số mẫu/sự kiện bị abstain; không âm thầm loại chúng khỏi mẫu số để tăng metric.

Chỉ được tuyên bố vượt mục tiêu độc lập khi có quyền dữ liệu hợp lệ, nhãn được phê duyệt, split khóa, model/threshold khóa và một lần đánh giá trên independent holdout chưa dùng để phát triển. Nếu chỉ có development data, kết quả phải ghi `DEVELOPMENT_ONLY` dù metric có vượt 85%.

## 3. Baseline phải đối chiếu bằng artifact

Đọc ít nhất:

- `reports/v11_research/metric_comparison.json`.
- `reports/v11_research/generalization_experiments.json`.
- `reports/v11_research/representation_experiments.json`.
- `reports/v11_research/fn_root_cause.json`.
- `reports/v11_research/dataset_manifest.json`.
- `reports/v11_research/model_manifest.json`.
- `reports/v11_research/event_match_evidence.json`.
- `reports/v11_research/test_results.json`.
- `reports/v11_research/17_FINAL_AI_RESEARCH_REVIEW.md`.

Baseline ghi nhận tại thời điểm soạn:

| Phạm vi | Kết quả |
| --- | --- |
| Controlled phone events | TP=16, FP=4, FN=13; 29 GT events |
| Phone event P/R/F1 | 80.00% / 55.17% / 65.31% |
| False alarms/min | 0.1813 trên khoảng 22.0632 phút |
| V11 E3 clip Macro-F1 | 56.39% |
| V11 E3 SAFE / PHONE recall | 60.38% / 66.78% |
| V11 E3 worst-subject Macro-F1 / SAFE recall | 52.14% / 54.36% |
| Seatbelt best development Macro-F1 | 58.52% |
| Inherited seatbelt UNCERTAIN recall | 4%; kiểm tra lại nguồn và nhãn |
| Inherited full backend benchmark | 1.119 FPS; context latency p50 600.79 ms, p95 2050.07 ms |
| V11 checks | 589 full tests, 14 targeted tests, frontend build và OpenAPI smoke PASS |

Runtime trên là benchmark kế thừa với phạm vi giới hạn, không phải bằng chứng một pipeline V12 đã chạy đủ mọi nhánh. Không gọi cached-feature throughput là full-system FPS.

Để hiểu độ lớn khoảng cách: nếu giữ đúng bộ development 29 events và 22.0632 phút, TP>=25, FP<=3, FN<=4 sẽ thỏa các mục tiêu event và FA/min nêu trên. Đây chỉ là phép kiểm tra tính nhất quán số học, tuyệt đối không phải chỉ dẫn tối ưu theo 29 GT events hoặc bảo đảm generalization.

Nếu artifact hiện tại khác bảng này, lưu discrepancy và giải thích bằng nguồn gốc, thời điểm, split, class mapping và matcher. Không chọn số đẹp hơn giữa hai nguồn.

## 4. Các điểm cần điều tra trực tiếp trong mã V11

Các quan sát sau là yêu cầu kiểm tra, không được mặc định là nguyên nhân duy nhất của lỗi:

### 4.1 Representation và lưu model

Đọc `training/dmd/v11_representation_experiments.py` và `v11_generalization_experiments.py`:

- `_load_subject()` dùng RGB cache 64×64 và frozen ResNet embeddings; ROI HOG cố định có thể mất chi tiết phone nhỏ.
- `_evaluate()` fit `StandardScaler` + balanced logistic regression theo LOSO và lưu prediction NPZ. Chưa thấy export đầy đủ fitted head/scaler cho đường suy luận V12; xác minh trước khi bổ sung.
- `predict()` và `predict_proba()` cần được kiểm chứng nhất quán theo class order. Probability chưa mặc nhiên được calibration chỉ vì có `predict_proba`.
- E3 được chọn sau nhiều lần nghiên cứu trên cùng ba subjects. Đó là exploratory development evidence, không phải independent validation.
- Ngưỡng worst-subject 20% của V11 chỉ là stop rule chống collapse, không phải tiêu chuẩn đạt mục tiêu V12.

### 4.2 Causal motion và session state

Đọc `training/dmd/causal_motion.py`:

- Motion dùng sample-index lags, không tự đảm bảo cùng khoảng thời gian nếu dropped frames hoặc timestamp không đều.
- `causal_running_standardize()` xuất vector zero trong giai đoạn thiếu lịch sử. Đo cold-start và tác động với hành vi xuất hiện ngay đầu phiên.
- Kiểm chứng reset theo video/session/camera/track; không mang trạng thái từ người lái hoặc stream trước sang stream sau.
- Kiểm chứng batch–stream equivalence, timestamp gaps, timestamp đảo thứ tự, numerical stability và bộ nhớ khi chạy dài.

### 4.3 ROI và association

Đọc `training/dmd/behavior_roi.py` và `driver_phone_association.py`:

- V11 có fixed-region fallback và provenance; kiểm tra ý nghĩa từng trường tọa độ trước khi tích hợp full-frame.
- `DriverPhoneAssociation.associate()` chọn verified track theo containment lớn nhất; cần đánh giá ambiguity khi driver/passenger boxes chồng nhau hoặc có điểm bằng nhau.
- Chuỗi `phone_context == MOUNTED_OR_STATIC` hiện đi thẳng tới reason code chứa chữ VERIFIED. Cần kiểm tra nguồn chứng cứ và bổ sung trạng thái xác minh độc lập, không biến model suggestion thành verified context.
- Association confidence hiện là tổ hợp heuristic, không được gọi là calibrated probability khi chưa đo calibration.

### 4.4 Verifier và khả năng tái lập

Đọc `training/dmd/verify_v11.py`, `v11_test_report.py`, `v11_reports.py` và `reports/v11_research/reproduce.ps1`:

- Hash matching chỉ xác nhận bytes, chưa chứng minh metric đúng. V12 phải tính lại confusion matrices và metrics từ prediction artifacts.
- Kiểm tra mọi native command exit code trong PowerShell; `ErrorActionPreference=Stop` không thay thế kiểm tra `$LASTEXITCODE`.
- Chỉ ghi PASS sau khi bước tương ứng thực sự chạy thành công. Không suy diễn PASS từ log cũ hoặc từ báo cáo tự sinh.
- Không yêu cầu rights phải luôn false để test PASS. Gate cần kiểm thử cả quyền hợp lệ lẫn không hợp lệ bằng synthetic fixtures và phải đọc record có thẩm quyền tại runtime.

## 5. P0 — Quyền dữ liệu, nhãn và chia tập

Đọc record hiện tại trong `dataset/external_dmd/README.md`, metadata nguồn và các manifest review. V11 ghi `RIGHTS_ACCEPTED=false`, `CANONICAL_ELIGIBLE=false`, `TEMPORAL_DEVELOPMENT_ELIGIBLE=true`. Kiểm tra hiện trạng; không tự bật cờ cho đủ điều kiện train.

Phân biệt quyền tải, quyền lưu cục bộ, quyền diagnostics, quyền train, quyền chia sẻ/upload và quyền xuất bản. Link tải hoạt động hoặc human-approved box không tự cấp quyền train cho dataset.

Chốt một gate thống nhất cho mọi entry point. Fitting logistic head, calibrator, fusion weights và boundary head đều là fitting/training; không dùng tên “frozen-feature diagnostic” để bỏ qua quyền. Nếu record không cho phép hoặc mâu thuẫn, chỉ thực hiện read-only audit, diagnostics được cho phép rõ ràng và synthetic tests; tạo cấu hình sẵn dùng cho khi đủ điều kiện.

Subjects `gZ-37` và `gE-28` tiếp tục được bảo vệ: không mở media, feature cache, labels hoặc kết quả của chúng để tuning, training hay chọn model. Không sử dụng legacy exception của holdout guard để lách quy tắc này.

Ba subjects `gB-9`, `gC-14`, `gZ-36` chỉ là repeatedly used development. Với dữ liệu mới hợp lệ:

1. Ghi provenance, SHA-256, quyền và label approval có bằng chứng.
2. Deduplicate theo subject/session/source video và near-duplicate, không chỉ theo tên file.
3. Chia train/validation/test theo subject; khóa các nguồn/session liên quan cùng nhóm, tránh leakage qua camera đồng bộ.
4. Không random-split các frame/clip cùng video sang train và test.
5. Khóa test trước mọi model selection; không đọc test để quyết định đặc trưng, calibration, thresholds hoặc dừng train.

Giữ riêng nguồn in-cabin và roadside; không lấy kết quả khác miền làm bằng chứng triển khai cho nhau. Nêu rõ dữ liệu s2 và hành vi xe đỗ có thể khác lái xe thực tế; xác minh metadata, không tự suy diễn.

Human review phải giữ AI suggestion, người xác nhận, timestamp, receipt và revision riêng. Không giả reviewer là con người. Không chuyển 13 FN đã xem nhiều lần thành tập validation độc lập.

## 6. P1 — Tạo candidate chạy và đánh giá được

Tái sử dụng module có sẵn, không tạo hệ thống fusion/event thứ hai trùng chức năng:

- `training/dmd/behavior_roi.py`.
- `training/dmd/causal_motion.py`.
- `training/dmd/driver_phone_association.py`.
- `training/dmd/fusion.py`.
- `training/dmd/evaluate_temporal.py`.
- `training/evaluate_events.py` và `training/evaluate_associations.py`.
- `backend/ai/events.py`.
- `backend/services/research_review.py`.

Đầu tiên tạo candidate adapter nghiên cứu, mặc định tắt trong production. Nó phải nhận observation theo timestamp và xuất action scores, association state, fused evidence và candidate events kèm provenance.

Nếu quyền cho phép fitting: export head/scaler/class mapping/normalization config sau mỗi fold; lưu training subjects, feature version, checkpoint hash, seed, library versions và calibration artifact. Thử load lại candidate rồi xác minh predictions trùng lần trước trong tolerance xác định. Không load pickle không rõ nguồn.

Nếu quyền chưa cho fitting: không fit lại V11; chỉ kiểm tra artifact hiện có, tạo inference/config interface và kiểm thử synthetic. Báo rõ model candidate nào thiếu fitted artifact nên chưa thể replay.

Thiết kế causal state bounded theo session/track. Phân biệt missing history, missing view và negative evidence. Warm-up, dropped frame và camera switch phải có reason codes. Chỉ dùng dữ liệu đã tới thời điểm inference; không cập nhật scaler bằng cả test session trước khi suy luận.

Lưu prediction không chỉ cho frame có nhãn: giữ timeline quan sát đầy đủ, đánh dấu missing/invalid evidence và chỉ mask labels ở bước chấm điểm. Không nối temporal window xuyên các gap do bỏ frame không có nhãn.

## 7. P2 — Phone detection và association có chứng cứ thật

Kế thừa 13-FN audit thay vì viết lại kết luận bằng suy đoán. V11 có 11 specialist misses, 12 valid auxiliary boxes, 13 UNKNOWN occupant tracks và 9 case dưới action gate; đối chiếu lại bằng artifacts.

Với development media được phép đọc, kiểm tra original-resolution frame/clip của FN và FP, không chỉ contact sheet hoặc cache 64×64. Lưu đường dẫn nguồn, hash, timestamp, frame index và crop transform.

So sánh full driver ROI với hand–phone/face–hand crops ở resolution phù hợp sau khi profile CPU/RAM. Với crop chạm biên driver ROI, lấy thêm pixel thật từ original frame nếu có; replicate padding không khôi phục chi tiết đã bị cắt khỏi nguồn.

Khảo sát detector lỗi vì vật nhỏ, che khuất, lap-phone, ánh sáng, blur hay domain shift. Không tự suy ra detector AP từ tập 13 FN chọn sẵn. Chỉ fine-tune khi có train split được phép và nhãn đầy đủ.

Association cần role evidence, verified track, seat/camera context nếu có và temporal consistency. Trả UNKNOWN khi ambiguous; không chọn driver chỉ vì gần phone nhất hoặc containment cao nhất. Passenger phone, mounted camera, mounted phone, phone cầm nhưng không dùng và động tác tay giống gọi điện phải có test/negative examples.

Không loại toàn bộ true-phone events chỉ vì association UNKNOWN rồi tuyên bố precision tăng. Đo coverage, abstention, lost TP và cơ chế review/secondary evidence; passenger use và mounted device không được thành driver violation.

## 8. P3 — Thực nghiệm có ngân sách và stop rule

Không chạy một sweep vô hạn cho tới khi một con số vượt 85%. Tạo experiment plan có hash trước khi chạy; ghi thời điểm và mọi lần thay đổi plan. Chỉ gọi pre-registered nếu có bằng chứng plan đã tồn tại trước khi xem kết quả.

Ưu tiên các experiment theo thứ tự và điều kiện:

| ID | Giả thuyết | Thay đổi chính | Điều kiện | Quyết định |
| --- | --- | --- | --- | --- |
| V12-A | V11 metrics tính đúng và artifact tái lập được | Recompute từ frozen predictions | Không fitting mới | KEEP nếu reconcile |
| V12-B | E3 dùng được online qua session/gap/cold-start | Streaming implementation tương đương | Synthetic và diagnostics được phép | KEEP nếu parity + state tests |
| V12-C | Original-resolution behavior ROI giảm mất phone detail | ROI/resolution; giữ detector/head cố định | Media diagnostics được phép | Theo localization yield, lỗi và runtime |
| V12-D | Appearance bổ sung thông tin mà motion E3 thiếu | Một feature fusion nhỏ | Train rights, labels và group split hợp lệ | Clip + worst-subject metrics |
| V12-E | Role evidence giảm wrong-occupant alerts | Association có xác minh/ambiguity | Evidence và nhãn association | Precision, coverage, lost TP |
| V12-F | Candidate tốt hơn cải thiện event trong cùng policy | Action/object evidence; temporal config cố định | Candidate loadable và timeline đầy đủ | Event P/R/F1, FA/min, delta IDs |
| V12-G | Temporal model nhỏ cải thiện boundary | Một TCN/GRU/boundary head | Approved event train/validation | Event + duration-bin + latency |
| V12-H | Nhãn visibility đúng giúp seatbelt | Một ROI/attention/calibration factor mỗi lần | Approved labels, rights và split | Macro-F1, UNCERTAIN, risk/coverage |

Đặt ngân sách candidate, seeds, compute và stopping criteria trong plan dựa trên tài nguyên thực. Mỗi experiment phải chỉ rõ baseline, changed factor, controls và metric chính. Không gộp đổi ROI + model + threshold + matcher rồi kết luận một yếu tố là nguyên nhân.

Model selection dùng validation hoặc inner group split khi dữ liệu đủ; không chọn candidate bằng outer test folds rồi gọi outer score là unbiased test. Với ba subjects cũ, tiếp tục ghi exploratory development-only. Không nâng số tham số chỉ vì linear head chưa đủ tốt.

So sánh worst-subject/per-class cùng pooled metric. Candidate gây SAFE collapse, mất true events nghiêm trọng hoặc tăng FA/min ngoài gate đã chốt phải REJECT. Lưu cả candidate thất bại; không xóa predictions để chỉ giữ best run.

## 9. P4 — Nối clip scores tới event có thể kiểm chứng

Khoảng trống lớn nhất của V11: E3 mới có clip results, chưa có Event F1. Khi đủ điều kiện, tạo replay trên timeline đầy đủ bằng candidate predictions thật, không dùng GT để đặt start/end, mở gate hay kéo dài event.

Giữ matcher, tolerance/IoU, matching semantics và exposure denominator cố định giữa baseline/candidate. Nếu benchmark lịch sử dùng matcher khác đường production, báo hai protocol riêng; không âm thầm đổi protocol.

Lưu từng event gồm ID ổn định, subject/session/view/track, predicted start/end, first observable evidence time, emission time, class, raw object score, action score, geometry, association, fused score, config hash và reason codes.

Xuất bảng GT↔prediction matches, unmatched GT/predictions, recovered TP IDs, lost TP IDs, new FP IDs, removed FP IDs. Metrics phải tính lại từ bảng match; không tự điền ID thiếu trong historical artifact. Chỉ so delta khi cùng subjects, exposure và policy.

Đo riêng events <1 s, 1–3 s, 3–10 s, >10 s; fragmentation, duplicates, onset/offset errors và online alert latency. Timestamp event start có thể được ghi lùi theo quan sát đã có, nhưng latency phải tính từ thời điểm hệ thống thực sự phát cảnh báo, không từ timestamp ghi lùi.

Không đổi tên tập phone FNs thành “recovered events” khi mới phát hiện box. Không chỉ tăng sampling density; temporal resolution phải đi cùng evidence chất lượng và chi phí đo được.

## 10. P5 — Seatbelt ba trạng thái

Đối chiếu nguồn của 500 pending candidates và review receipts mới nhất. Phân biệt approved labels, model suggestions, legacy labels, pending và disputed. Trạng thái đã duyệt phone không đồng nghĩa seatbelt được duyệt.

Không fit khi quyền hoặc nhãn chưa đủ. Có thể chuẩn bị review queue/config và chạy synthetic tests trong lúc chờ. Không tự bịa consent, reviewer hoặc approved status.

Nếu đủ điều kiện, bắt đầu warm-start checkpoint có provenance, so sánh lần lượt torso ROI, visibility-aware crop, lightweight attention và calibration. Không chạy full training/Triplet Attention trước khi một baseline hợp lệ tái lập được.

Giữ ba trạng thái FASTENED, UNFASTENED, UNCERTAIN_OR_OCCLUDED. Belt không nhìn thấy không đồng nghĩa unfastened. Báo Macro-F1, per-class metrics, balanced accuracy, false-unfastened rate với mẫu số rõ ràng, abstention/coverage và selective risk. Không đạt mục tiêu bằng cách dự đoán UNCERTAIN gần như toàn bộ dữ liệu.

## 11. Nghiên cứu khoa học gắn với code

Kế thừa `reports/v11_research/literature_review.json`, chỉ tra cứu primary sources khi cần xác minh một phương pháp hoặc triển khai mới:

- DualScope: https://ojs.aaai.org/index.php/AAAI/article/view/37812
- DriveCLIP: https://github.com/zahid-isu/DriveCLIP
- Multi-Task Fusion: https://doi.org/10.3390/s25216799
- ActionFormer: https://arxiv.org/abs/2202.07925
- MixStyle: https://arxiv.org/abs/2104.02008
- Group DRO: https://arxiv.org/abs/1911.08731
- Seatbelt Attention: https://doi.org/10.3390/app14083346
- Overhead Seatbelt YOLOv7: https://doi.org/10.35784/acs_7594

Với ý tưởng thực sự áp dụng, nêu code path, khác biệt so với paper, dataset/protocol, license/checkpoint status và ablation kiểm chứng. Không gọi một phép normalization là reproduction đầy đủ của MixStyle hoặc một temporal feature là full ActionFormer.

Không tải weights hàng GB, thuê GPU hoặc upload media trước khi kiểm tra quyền, RAM/VRAM, disk và kích thước artifact. Nếu thiếu điều kiện cho CLIP/large backbone, ghi BLOCKED và ưu tiên candidate nhỏ có thể kiểm chứng.

## 12. Engineering, validation và báo cáo

Chỉ tạo file mới nếu module hiện có không đáp ứng; đặt code nghiên cứu ở candidate path, feature flag mặc định tắt. Không đổi default production config trước đánh giá và phê duyệt riêng.

Test tập trung vào failure modes thật:

1. Future-invariance và streaming/batch parity.
2. Cold-start, timestamp gaps, dropped/out-of-order frames, session/track reset.
3. Train-only scaler/calibrator; protected holdout bị chặn trước I/O.
4. Export/load parity, class mapping và probability provenance.
5. ROI full-frame/crop round-trip, biên ảnh, phone near lap, missing landmarks.
6. Driver/passenger overlap ambiguity; mounted context chưa xác minh không được tự thành VERIFIED.
7. Missing views giữ UNKNOWN; raw scores và rejected scores đều truy vết được.
8. Short events, fragmentation, duplicates, cooldown và emission latency.
9. SAFE collapse, seatbelt UNCERTAIN và abstention accounting.
10. Metric recomputation từ NPZ/match records; không dựa vào JSON tự khai.
11. Rights gate chặn mọi fitting khi không được phép, cả linear head và calibration.

Chạy compile, scoped Ruff, targeted/full pytest, frontend build và backend smoke phù hợp repo. Chạy inference smoke nếu tài nguyên đủ; nếu không, ghi NOT RUN cùng lý do. Không ghi PASS cho bước chưa chạy hoặc dùng kết quả V11 như test V12 mới.

Profiler phải ghi hardware/software, active components, input resolution, number of views, warm-up, p50/p95/p99, peak RSS/VRAM và full-pipeline FPS. Báo camera/source FPS riêng với processing FPS. Không tuyên bố real-time khi xử lý thấp hơn yêu cầu và chưa có giải pháp latency đã đo.

## 13. Deliverables V12

Tạo `reports/v12_research/` với nội dung có chứng cứ, không yêu cầu đủ file bằng cách tạo placeholder rồi đánh dấu COMPLETE:

- `01_CODE_AND_DATA_AUDIT.md`: HEAD, thay đổi local, code findings, rights/labels/compute.
- `02_METRIC_RECONCILIATION.md`: chỉ số tính lại và discrepancy.
- `03_ONLINE_CANDIDATE.md`: model/scaler export, causal session state, parity và latency.
- `04_PHONE_ROI_ASSOCIATION.md`: original-resolution evidence, FN/FP và association.
- `05_REPRESENTATION_ABLATIONS.md`: plan, controls, per-class/per-subject, thất bại.
- `06_EVENT_REPLAY.md`: match records, event deltas, FA/min, short-event và latency.
- `07_SEATBELT_RESULTS.md`: approval audit và thí nghiệm đủ điều kiện.
- `08_RUNTIME_AND_TESTS.md`: logs, exit codes, phạm vi đo và bước NOT RUN.
- `09_FINAL_RESEARCH_REPORT.md`: kết quả, khoảng cách mục tiêu, blocker và next experiment.

Kèm machine-readable `experiment_plan.json`, `experiment_registry.json`, `dataset_manifest.json`, `model_manifest.json`, `metric_comparison.json`, `validation.json`, `artifact_manifest.json`, predictions, confusion matrices, per-subject results, event match records và `reproduce.ps1` khi tương ứng đã được tạo/chạy thực sự.

Mỗi artifact ghi source hashes, code commit + dirty diff hash nếu cần, split/model/config hash, seed, command, start/end time, exit code và evidence level. Reproduce script phải dừng khi gate hoặc command thất bại, không tiếp tục sinh báo cáo PASS.

Candidate bị BLOCKED phải ghi chính xác thiếu quyền/dữ liệu/nhãn/checkpoint nào và phần công việc độc lập đã hoàn thành. Không suy diễn thiếu dữ liệu từ cache cũ mà chưa kiểm tra manifest mới nhất. Khi có thể tiến hành phần khác hợp lệ thì tiếp tục, không dừng cả dự án ở một blocker.

## 14. Independent evaluation và quyết định cuối

Chỉ mở một independent test mới sau khi đã khóa model, preprocessing, streaming state policy, calibration, threshold, fusion, event config và matcher. Ghi lock hashes và evaluated subjects. Không quay lại chỉnh model theo kết quả test; lần phát triển tiếp theo cần test mới phù hợp.

Báo confidence intervals bằng đơn vị độc lập phù hợp như subject/session khi số nhóm đủ; không bootstrap các frame tương quan để tạo CI quá hẹp. Nếu nhóm quá ít, nêu giới hạn thay vì tạo vẻ chắc chắn giả.

Trả lời cuối phải có:

1. HEAD sử dụng, source files đã đổi và command tái lập.
2. Experiments đã chạy, hypothesis, KEEP/REJECT/BLOCKED và lý do.
3. Checkpoint, dataset, split, rights và human-review provenance.
4. Phone TP/FP/FN, Event P/R/F1, FA/min và event delta IDs.
5. SAFE/PHONE/OTHER metrics, worst-subject và per-subject results.
6. Seatbelt Macro-F1, UNCERTAIN recall, false-unfastened, abstention và coverage.
7. Runtime thật, active pipeline stages và tests thực sự chạy.
8. Có independent holdout chưa, có vượt từng mục tiêu >85% chưa, còn thiếu gì.
9. Candidate có đủ điều kiện đề nghị review promotion không; không tự activate production.
10. Experiment tiếp theo có giá trị cao nhất dựa trên lỗi còn lại.

Nếu chưa đạt độc lập:

```text
production_approved = false
final_status = RESEARCH_ONLY
independent_ai85_established = false
```

Mục tiêu là cải thiện hệ thống bằng bằng chứng, không mặc định dừng tại 56.39% hoặc cam kết mọi chỉ số vượt 85%. Bắt đầu bằng metric reconciliation, rights/labels audit, rồi đóng khoảng trống từ E3 clip predictions sang candidate online có thể tái lập và event replay có ID đầy đủ. Chỉ train khi gate thực sự cho phép.
