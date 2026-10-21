# Nghiên cứu cách sửa ROADWATCH để hướng tới mọi chỉ số bắt buộc >85%

Rà soát ngày 21/10/2026. Phạm vi: code hiện tại, artifacts V26 và nguồn khoa học chính chủ. Đây là **nghiên cứu + chẩn đoán chạy thật trên artifacts**, chưa huấn luyện model, chưa sửa inference pipeline và chưa có metric candidate mới.

Kết quả phân tích tái lập: [fix_feasibility.json](fix_feasibility.json). Script: [analyze_fix_feasibility.py](analyze_fix_feasibility.py). Phiên bản chính xác của các nguồn bổ sung: [source_register.json](source_register.json). Script chỉ dùng thư viện chuẩn, kiểm tra counts, hash đầu vào, timing trace và khả năng lấy mẫu; không gọi detector và không truy cập holdout.

## 1. Những kết luận cần sửa so với hướng dẫn trước

1. **Lỗi candidate identity ở raw runner không giải thích trực tiếp F1=53,33% của E2.** `experiments/E2/metrics.json` ghi `action_mode=NONE`. Cached runner đã chọn `best_associated` trước khi tính action. Regression phải đi qua raw runner có action; chạy lại E2 không kiểm chứng được việc sửa nhánh action.
2. **Ghi timestamp chưa đủ để bảo đảm fusion đúng thời gian.** Trong `replay_v24_record`, `predict_from_sequence()` trả score cho tâm cửa sổ, nhưng pose/candidate/rule lấy từ record hiện tại và fusion được ghi vào observation hiện tại. Vòng ngoài đưa observation thẳng vào reporter. Không thấy queue căn chỉnh target timestamp ở đường này.
3. **Không được thay resize hoặc cadence riêng ở inference.** `FEATURE_CONTRACT` khóa resize 64→112, label `CENTER`, sequence 9; checkpoint khai báo chu kỳ 0,5 giây. Resize/cadence/label mới cần cache và checkpoint tương ứng. Đổi chuỗi manifest để vượt guard không phải sửa model.
4. **Raw runner có khả năng fallback dù yêu cầu deep model.** Constructor không truyền `causal_mode` hoặc `missing_checkpoint_policy`; adapter mặc định strict causal + rule fallback. Checkpoint centered bị từ chối trong strict causal. Đây là kết luận từ code path, chưa phải khẳng định một raw run lịch sử đã chạy deep/fallback. Phải xuất requested/actual mode, reason và forward count trong mỗi run.
5. **F1 >85% không đồng nghĩa cả precision và recall >85%.** Ví dụ TP/FP/FN=15/0/5 cho F1=85,71% nhưng recall=75%. Điều kiện 18 TP và ≤3 FP áp dụng khi yêu cầu đồng thời P/R/F1 >85% trên 20 GT.

## 2. Lỗi thời gian đã đo được từ raw trace E4/E5

| Run | Score centered hợp lệ | Score target khác observation | Observation có fusion >0 và lệch target | Khoảng lệch |
|---|---:|---:|---:|---|
| E4 | 1.803 | 1.803 | 563 | 2,016077–2,016181 giây |
| E5 | 1.803 | 1.803 | 553 | 2,016077–2,016181 giây |

Ví dụ E4 gC-14 frame 390: observation=13,104839s, score target=11,088700s, score available=13,105s. Score cũ được kết hợp với detector của thời điểm mới. Đây là sai lệch ý nghĩa thời gian, không tự chứng minh có đọc feature tương lai trước availability. Số event bị ảnh hưởng và mức tăng F1 sau sửa vẫn **chưa đo**.

**Thiết kế sửa có thể triển khai ngay mà không train:**

- Buffer record theo `(session, camera, occupant/seat stream, source_frame_id)` cùng detector/pose/candidate ở thời điểm target.
- Mỗi tick chỉ append một embedding cho mỗi stream. Khi đủ 9 mẫu, tìm record tâm bằng source identity, không làm tròn timestamp rồi tìm nearest tùy ý.
- Tính deep score của cửa sổ; tính rule/phone-conditioned features bằng candidate và pose của **record tâm**, rồi fusion cùng record đó.
- Đưa observation target vào event engine theo thứ tự target time; giữ riêng `available_at` và thời gian emission thực. Boundary event có thể mô tả quá khứ, nhưng cảnh báo chỉ xuất khi có đủ dữ liệu.
- Tất cả stream, NONE/RULE và deep, phải có chính sách thời gian rõ ràng khi so sánh; không thay timestamp của rule/current để giả lập deep cùng thời điểm.
- EOF thiếu future context: đánh dấu score không có; fallback theo policy đã khai báo và báo coverage, không giả frame tương lai.
- Test transition PHONE→SAFE và SAFE→PHONE: thay đổi frame sau target không được làm đảo identity/candidate của target; kiểm tra emission không sớm hơn availability.

**Phương án dài hạn:** huấn luyện head causal với label tại cuối cửa sổ; hoặc online TSM/causal temporal convolution. Cần checkpoint riêng. Đổi `CENTER` thành `LAST` trên checkpoint cũ là sai.

Vị trí cần sửa: `training/dmd/run_verified_v24.py::replay_v24_record`, vòng reporter; `action_adapter.py::predict_from_sequence`; raw runner và shared backend adapter.

## 3. Cần cải thiện bao nhiêu mới đạt?

Các hàng sau chỉ là **kịch bản số học**, không phải kết quả model. Giả định TP/FP/FN khác giữ nguyên theo mô tả; thay inference thật có thể làm tất cả counts thay đổi.

| Kịch bản | TP/FP/FN | P | R | F1 |
|---|---|---:|---:|---:|
| V26 E2 đo được | 12/13/8 | 48,00% | 60,00% | 53,33% |
| Loại đúng 6 duplicate FP, giữ counts khác | 12/7/8 | 63,16% | 60,00% | 61,54% |
| Loại tất cả FP, không phục hồi FN | 12/0/8 | 100% | 60,00% | 75,00% |
| Phục hồi tất cả FN, giữ 13 FP | 20/13/0 | 60,61% | 100% | 75,47% |
| Phục hồi 3 FN + loại 6 FP | 15/7/5 | 68,18% | 75,00% | 71,43% |
| Một tổ hợp thỏa đồng thời >85% | 18/3/2 | 85,71% | 90,00% | 87,80% |

Trên cohort này, chỉ có TP∈{18,19,20}, FP≤3, FN=20−TP thỏa cả ba gate. Phải phục hồi ít nhất 6 trong 8 FN và loại ít nhất 10 trong 13 FP **nếu so counts cuối với baseline**. Không có một threshold hay một sửa duplicate đơn lẻ bảo đảm điều đó.

Trong một cascade bắt buộc qua bốn gate, nếu mỗi gate giữ lại 85% GT đã sống qua gate trước thì retention cuối chỉ 0,85⁴=52,20%. Muốn tích đạt 85% với bốn retention bằng nhau, mỗi gate khoảng 96,02%. Đây là chain rule cho cùng quần thể event và retention có điều kiện, không phải phép nhân AP, accuracy và F1 của các dataset khác nhau. Do đó cần đo coverage theo tầng, không chỉ đặt mỗi component một mục tiêu 85% rồi kỳ vọng event tự đạt.

## 4. Sự kiện ngắn: tăng FPS vẫn chưa đủ

Phân tích dùng FPS và toàn bộ frame grid từ metrics V26, phase bắt đầu tại frame 0. Chưa chạy detector ở các frame mới. Các số chỉ thể hiện sampling support bên trong GT.

| FN | Độ dài | Số mẫu stride 15 | Số mẫu stride 5 | Span của stride 5 | Kết luận |
|---|---:|---:|---:|---:|---|
| GT_05 | 0,4032s | 1 | 3 | 0,3360s | Dù tăng cadence vẫn không thể có span 0,5s hoàn toàn trong GT |
| GT_10 | 0,8064s | 1 | 4 | 0,5040s | Tăng cadence tạo đủ temporal support; detector/association vẫn cần kiểm chứng |
| GT_12 | 0,6048s | 1 | 3 | 0,3360s | Stride 5 vẫn thiếu span 0,5s; stride 3 mới có span khoảng 0,5040s ở phase hiện tại |
| GT_14 | 0,9072s | 2 | 5 | 0,6720s | Baseline đã có hai mẫu; nguyên nhân chính hiện tại là không có exported proposal |

Thiết kế thử: detector/pose cadence `{15,10,5,3}` frames ở FPS nguồn; duration gate `{0.2,0.35,0.5}` giây, min observations vẫn ≥2 cho thử nghiệm đầu. Từng biến thử riêng trước khi phối hợp hai biến có tương tác. Grid là đề xuất kỹ thuật cần khóa trước chạy, không phải giá trị tối ưu đã được paper chứng minh.

Action cadence phải là scheduler riêng theo checkpoint. Không đẩy 9 frame ở 10Hz vào GRU học 9 frame ở 2Hz rồi gọi đó là cùng model. Adaptive sampling từ 2Hz tăng lên khi có cue có thể bỏ lỡ phần đầu event; dùng dense fixed cadence làm đối chứng và chỉ dùng cue inference, không dùng GT.

## 5. Ma trận lỗi → phương án sửa → phép kiểm chứng

| Ưu tiên | Bằng chứng / vị trí | Cách sửa cụ thể | Điều kiện chứng minh |
|---|---|---|---|
| P0 | Lệch target ~2,016s, cached E4/E5 | Queue record theo target, rule và deep cùng record; log emit time riêng | Replay cùng feature/input, timing invariant pass, báo P/R/F1 trước/sau |
| P0 | Raw action lấy detection[0], event lấy best candidate | Chọn driver candidate trước; shared pipeline; state update một lần | Test A/B qua raw runner, candidate/frame/occupant ID trùng; action-enabled raw ablation |
| P0 | Raw adapter default strict + fallback | Expose causal/missing-checkpoint policy; research deep run dùng ERROR; preflight forward | requested=actual khi thí nghiệm deep, count>0 sau warm-up, fallback có lý do |
| P0 | Cache D5 và raw D2 khác input | Chạy raw D5 cả gC/gZ cùng config; provenance theo source | Full session, shared backend policy, ledger đủ 20 GT |
| P1 | GT_02: 20 candidate frames, 1 associated | Role GT; track ổn định; distances chuẩn hóa theo người/ghế | Driver-retention tăng, passenger FP và identity switch không tăng ngoài tolerance khóa trước |
| P1 | GT_14/16 không có exported candidate | Capture đầu ra trước project filters; hạ detector floor có giới hạn; direct high-res/SAHI | Bbox recall/AP đo trên nhãn đầy đủ, biết miss ở model/ROI/NMS nào |
| P1 | GT_17/19 sparse evidence | Low-score continuation kiểu ByteTrack, timeout; local hand/face ROI | Chỉ nối track đã xác minh; đo drift/false continuation; không coi bbox dự đoán là detection quan sát |
| P1 | 6 duplicate FP, bridge/cooldown heuristics | Stable event ID; update interval hiện có; merge dựa track + evidence | Không xóa mù event gần nhau; kiểm tra hai GT thật cách ngắn vẫn tách được |
| P1 | Hai model confidence khác thang | Calibration từng nguồn bằng correctness labels; source-aware fusion | Brier/NLL/reliability + bbox/event metrics; không tự coi raw confidence là probability |
| P1 | Rule SAFE hard veto, action uncalibrated | NONE/RULE/DEEP/HYBRID; giữ hard gate identity, thử soft fusion action | Component score đúng mà event bị loại được truy nguyên; chọn bằng validation |
| P1 | Feature bị giảm 64×64 | Preprocess contract mới 112 trực tiếp/224; re-extract train/val và retrain head | Cache hash mới, checkpoint phù hợp; baseline cũ vẫn giữ contract cũ |
| P1 | Chỉ ba subject development trong allowlist | Data manifest mở rộng subject có quyền, group split + duplicate audit | Không tái dùng gZ-37/gE-28 đã consumed; quyền và split có bằng chứng |
| P1 | Seatbelt uncertain yếu | Visibility stage + belt stage; hard examples; model 3-class end-to-end evaluation | Không loại uncertain khỏi denominator; macro/per-class metrics đủ support |
| P2 | Detector hardcode CPU | Tham số device xuyên detector/pose/action, preflight cuda; profile bottleneck | Log device thực, GPU memory, numerical parity, throughput cả decode |

Các “raw proposals” hiện lấy từ `res.boxes` sau `YOLO.predict(conf=...)`, đã chịu filter/NMS trong thư viện. Đặt tên pre-project-filter chính xác; muốn tuyên bố pre-NMS phải capture tensor trước postprocess. Hạ floor rồi lưu `res.boxes` không khôi phục proposals đã bị internal NMS loại.

## 6. Nghiên cứu nên dùng như thế nào?

Các lựa chọn dưới đây có nguồn gốc rõ, còn hiệu quả trên ROADWATCH là **giả thuyết cần thí nghiệm**.

| Công trình | Tác dụng liên quan | GitHub chính chủ / tác giả dẫn | Giới hạn khi áp dụng |
|---|---|---|---|
| [SAHI, 2022](https://arxiv.org/abs/2202.06934) | Sliced inference làm vật nhỏ chiếm nhiều pixel; thử cho phone ROI | [obss/sahi](https://github.com/obss/sahi) | Bài gốc đo aerial AP, không chứng minh phone event F1; tốn thêm compute và có duplicate |
| [ByteTrack, ECCV 2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136820001.pdf) | Dùng low-score detection để duy trì track đã có thay vì bỏ ngay | [FoundationVision/ByteTrack](https://github.com/FoundationVision/ByteTrack) | Phone nhỏ/đổi hướng khác benchmark tracking người; phải có detection và role evidence |
| [Temperature scaling, ICML 2017](https://proceedings.mlr.press/v70/guo17a.html) | Fit T trên validation logits; giảm sai lệch confidence | [gpleiss/temperature_scaling](https://github.com/gpleiss/temperature_scaling) | Repo tự báo không còn duy trì; scalar T không đổi argmax, không tự chữa classification errors |
| [TSM, ICCV 2019](https://arxiv.org/abs/1811.08383) | Online temporal representation phù hợp causal head nhẹ | [mit-han-lab/temporal-shift-module](https://github.com/mit-han-lab/temporal-shift-module) | Chọn đúng online variant; demo gốc nhận dạng cử chỉ, cần train/map action của ROADWATCH |
| [DriveCLIP, 2024](https://arxiv.org/abs/2306.10159) | Pretrained visual embedding, linear probe rồi fine-tune có kiểm soát | [zahid-isu/DriveCLIP](https://github.com/zahid-isu/DriveCLIP), [DeepInsigth được README dẫn cho VideoCLIP](https://github.com/jiajingchen113322/DeepInsigth) | Đã tìm được repo VideoCLIP phụ: có source nhưng chưa xác minh checkpoint/config/evaluator tái lập headline; root repo phụ không có README/license rõ ở trang đã mở |
| [VideoMAE, NeurIPS 2022](https://arxiv.org/abs/2203.12602) | Pretrained video encoder; thử khi head nhẹ vẫn yếu và có data/compute | [MCG-NJU/VideoMAE](https://github.com/MCG-NJU/VideoMAE) | “Ít dữ liệu” trong paper vẫn khoảng 3–4 nghìn video; không có cơ sở cho rằng hai người là đủ. Repo chủ yếu CC-BY-NC 4.0; offline model không mặc nhiên causal |
| [ActionFormer, ECCV 2022](https://arxiv.org/abs/2202.07925) | Temporal multiscale + regression boundary | [happyharrycn/actionformer_release](https://github.com/happyharrycn/actionformer_release) | Cần train trên feature/interval nhãn đúng; code offline cần causal adaptation hoặc khai báo lookahead nếu dùng online |

Không cần thay tất cả thành Transformer. Thử action theo thứ tự chi phí: frozen encoder + linear probe → temporal consensus/causal TCN hoặc online TSM → local/global fine-tuning → VideoMAE. Mọi model mới phải dùng cùng subject split và decoder trước khi so hiệu quả representation.

Với calibration, softmax(z/T), T>0 giữ nguyên lớp argmax; lợi ích có thể nằm ở threshold/fusion, không phải tự động tăng Top-1. Detector confidence cần calibration riêng theo định nghĩa bbox đúng/sai; không áp dụng mù softmax-temperature cho score sau NMS.

## 7. Bổ sung dữ liệu mới là một phương án kỹ thuật chính

- Tăng đa dạng subject/session/camera thay vì chỉ lấy thêm frame gần nhau từ hai người. Chọn quy mô từ learning curve theo số subject; không hứa một con số subject đủ đạt 85%.
- Thu thập hard negatives từ train/development: chạm tai, uống nước, che mặt, nội thất giống phone, passenger phone, phone đặt giá đỡ.
- Phân bố bbox validation phải có frame âm tính và sampling đại diện toàn session. Có thể thêm error-enriched subset để chẩn đoán nhưng phải báo riêng; không chỉ đo trên frame dễ hoặc frame quanh FN.
- Nhãn độc lập: phone bbox, occupant role, visibility, action và event start/end. Model hỗ trợ gợi ý nhãn chỉ tạo pseudo-label; cần người kiểm tra trước khi thành GT đánh giá.
- Version hóa data manifest để mở rộng development pool đã được cấp quyền; kiểm tra đồng thời `research_permissions.DEVELOPMENT`, checkpoint validator và holdout guard. Không bỏ allowlist hoặc giải phóng consumed holdouts để có thêm mẫu.
- Augmentation bảo toàn ngữ nghĩa: horizontal flip chỉ dùng khi sửa mapping trái/phải/ghế/calibration tương ứng; không copy-paste phone vào ghế phụ nhưng giữ nhãn driver use.
- Active learning chọn dữ liệu train chưa nhãn bằng uncertainty/disagreement + diversity. Không lấy hard negatives từ holdout.
- Có thể cải thiện góc camera/độ phân giải hoặc thêm modality nếu hệ thống thực tế cung cấp được. Khi đó là cấu hình benchmark mới, cần baseline cùng input; không so với single-camera cũ như một ablation cùng điều kiện.

## 8. Thứ tự thực nghiệm và cách chọn candidate

### Giai đoạn A — correctness, chưa search threshold

1. Tái lập E2 và lưu input/hash.
2. Sửa raw identity, thời gian fusion và actual-mode preflight bằng regression tests qua đúng call path.
3. Tái chạy E4/E5 với feature cũ và queue aligned, giữ các biến khác; đo mode coverage, timestamp invariants, recovered/lost TP, new/removed FP. E2 làm đối chứng không action.
4. Chạy full raw D5 cả hai người, action NONE rồi action mode tương ứng; kiểm tra backend parity.

### Giai đoạn B — search hữu hạn theo bottleneck

Khóa search plan trước mỗi phase trên development. Các giá trị dưới là điểm xuất phát do rà soát code, chưa phải kết quả tối ưu. Mỗi phase đổi một yếu tố, giữ winner của phase trước; cuối cùng thêm paired ablation để đo tương tác.

| Phase | Các giá trị thử | Điều kiện đầu vào |
|---|---|---|
| Detector resolution | 640, 960, 1280 | Phone bbox validation + hardware profiling |
| Slicing | off; tile 384 overlap 0,2; tile 512 overlap 0,2 | Giữ full-image pass, tọa độ/duplicate kiểm chứng |
| Detector/pose cadence | stride 15, 10, 5, 3 | Action scheduler giữ đúng contract riêng |
| Temporal duration | 0,2; 0,35; 0,5 giây | Min observations ≥2 và high-rate input đã capture |
| Bridge duration | 0; 0,5; 1; 2; baseline 4 giây nếu config xác nhận | Evidence gap/track timeout rõ, không kéo bridge theo GT |
| Action/fusion | NONE, RULE_ONLY, DEEP_ONLY, HYBRID | Cùng proposal/association, timestamp target khớp |
| Action representation | baseline64→112; direct112; direct224 | Mỗi contract mới có cache/train/checkpoint đúng |
| Seatbelt | 3-class baseline; visibility+binary | Nhãn và group split giống nhau, giữ đủ ba lớp ở output |

Không Cartesian product tất cả grid. Đặt giới hạn GPU-hours/disk theo tài nguyên đang có trước train; lưu checkpoint/resume. Các search boundary/calibration/fusion vẫn là fitting theo governance repo, dù model weight không đổi.

Candidate selection: nếu cả P,R>0,85 thì chọn F1 cao nhất trong runtime budget; nếu chưa có candidate thỏa, báo Pareto precision/recall/runtime, dùng tiêu chí khóa trước (ví dụ max min(P,R), rồi F1) để chọn hướng thử tiếp. Không chọn bằng số lần benchmark may mắn. Với training, dùng một seed để sàng lọc rồi ba seed khóa trước cho finalists nếu compute cho phép, báo variation.

### Giai đoạn C — xác nhận và tổng quát hóa

- Đo component và event bằng đúng candidate bundle, gồm tất cả mode/model/input dependencies.
- Report micro aggregate + per subject + worst subject + coverage; confidence interval dùng sampling unit subject/session khi đủ số nhóm, không coi hàng nghìn frame tương quan là hàng nghìn người độc lập.
- Với chỉ hai subject, báo rõ CI tổng quát hóa yếu. Point estimate >85% và lower confidence bound >85% là hai claim khác nhau; không tự chuyển qua lại sau kết quả.
- Candidate freeze trước independent evaluation. Thêm subject test chưa dùng phát triển; không “mở lại” holdout đã consumed.
- Chỉ toàn hệ thống MET khi **tất cả gate bắt buộc có ground truth và đều pass**. Một gate chưa đo đủ không được bỏ qua khỏi kết luận tổng.

## 9. Bắt đầu triển khai tại đâu?

Ưu tiên `run_verified_v24.py`/`action_adapter.py` cho lỗi thời gian đã đo, `run_autonomous_v21.py` cho candidate identity/actual mode/device và shared runner. Sau đó mới tới detector cadence, tracker, preprocessing contract mới, dataset mở rộng và fine-tuning.

Kaggle có thể giúp chạy inference và training khi có quyền, nhưng code đang hardcode CPU nên upload gói không tự tạo GPU acceleration. `requirements-windows-cu128.txt` hiện chỉ khóa torch/torchvision; `requirements-kaggle.txt` chỉ khóa Ultralytics/PyYAML/Pillow, còn NumPy/OpenCV và các thư viện nền phụ thuộc base image. Phải lưu base image + `pip freeze`, rồi chạy preflight kiểm tra device thực của detector, pose và action, checkpoint hashes, paths và artifact output trước full job.

Tình trạng cuối nghiên cứu: **chưa đạt >85%**; đã có thêm bằng chứng lỗi và phương án kiểm chứng cụ thể. Không có cơ sở để cam kết bất kỳ model nào chắc chắn đưa tất cả metrics vượt 85% trên dữ liệu độc lập.
