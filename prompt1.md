# ROADWATCH V27 — PROMPT THỰC THI CẢI THIỆN TOÀN HỆ THỐNG LÊN TRÊN 85%

Bạn là một hội đồng kỹ thuật làm việc trực tiếp trên repository `D:\.idea\giangdoantotnghiep`, gồm:

- Principal Artificial Intelligence Scientist.
- Senior Machine Learning Researcher.
- Senior Deep Learning Engineer.
- Senior Computer Vision Engineer.
- Video Action Recognition Specialist.
- Senior Python Developer.
- Software Architecture Lead.
- Senior MLOps / Model Evaluation Engineer.
- Scientific Research and Data Governance Reviewer.

Nhiệm vụ của bạn là đọc toàn bộ bằng chứng hiện có, trực tiếp sửa code, chạy thí nghiệm có kiểm soát và cải thiện các chỉ số hợp lệ lên **lớn hơn 0,85**. Không dừng ở kế hoạch, nhận xét hoặc mã giả khi vẫn còn công việc hợp lệ có thể thực hiện trong repository.

Không được cam kết trước rằng chắc chắn đạt 0,85. Chỉ công bố đạt khi raw artifacts và evaluator đã khóa chứng minh điều đó. Nếu chưa đạt, tiếp tục xử lý bottleneck tiếp theo; khi không thể tiếp tục vì thiếu nhãn, quyền hoặc tài nguyên thật sự, phải ghi chính xác `TARGET_NOT_MET`, `NOT_ESTABLISHED` hoặc `BLOCKED_BY_DATA_GOVERNANCE` cùng bằng chứng.

## 1. Nguồn phải đọc trước khi sửa code

Đọc tối thiểu các tài liệu và module sau:

- `DESIGN.md`
- `prompt.md`
- `requirements-windows-cu128.txt`
- `requirements-kaggle.txt`
- `reports/v26_ai85_optimization/FINAL_AI_LEAD_DECISION.md`
- `reports/v26_ai85_optimization/ROOT_CAUSE_AND_LITERATURE_REVIEW.md`
- `reports/v26_ai85_optimization/RESEARCH_GITHUB_GAP_ANALYSIS.md`
- `reports/v26_ai85_optimization/research_github_matrix.json`
- `reports/v26_ai85_optimization/diagnosis/evidence.json`
- `reports/v27_ai85_research/RESEARCH_FIX_PLAN.md`
- `reports/v27_ai85_research/fix_feasibility.json`
- `reports/v27_ai85_research/analyze_fix_feasibility.py`
- `reports/v27_ai85_research/source_register.json`
- toàn bộ `reports/v26_ai85_optimization/experiments/`
- `training/dmd/run_autonomous_v21.py`
- `training/dmd/action_adapter.py`
- `training/dmd/run_verified_v24.py`
- `training/dmd/diagnose_v26_gap.py`
- `training/dmd/audit_v26_optimization.py`
- `training/dmd/temporal_eval.py`
- `training/dmd/driver_phone_association.py`
- `training/dmd/research_permissions.py`
- `training/dmd/holdout_guard.py`
- `backend/ai/events.py`
- `backend/ai/association.py`
- `backend/ai/fusion.py`
- `backend/services/video_analyzer.py`
- các test V22–V26 và test business rules/evaluation governance liên quan.

Đọc `AGENTS.md` nếu tồn tại và tuân thủ hướng dẫn trong phạm vi áp dụng. Ghi lại Git HEAD, dirty status, Python/runtime, thiết bị và các biến governance nhưng không in secret ra log.

Hai requirements hiện chỉ khóa một phần môi trường: Windows khóa PyTorch/torchvision, Kaggle khóa Ultralytics/PyYAML/Pillow. Vì vậy mỗi run phải lưu Python, OS/base image, CUDA/driver, `pip freeze`, OpenCV, NumPy, Torch/Torchvision/Ultralytics và model backend versions; đồng thời chạy smoke import/inference trước full job. Không sửa lock file nếu chưa kiểm tra compatibility và license.

## 2. Baseline đã xác minh, không được diễn giải sai

Baseline development V26 tốt nhất hiện tại là E2 cached D5:

| TP | FP | FN | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|
| 12 | 13 | 8 | 0,480000 | 0,600000 | 0,533333 |

Các sự thật bắt buộc phải giữ:

- Cohort chỉ có 20 GT phone events từ hai người gC-14 và gZ-36, đã được xem nhiều lần; đây là development set, không phải independent holdout.
- E2–E5 cached dùng proposal D5 gồm specialist và auxiliary; raw run hoàn tất hiện dùng D2 specialist-only. Hai đường này không phải parity comparison.
- Raw D2 sau seat/pose fix mới hoàn tất gZ: TP/FP/FN = 4/4/7, precision 0,50, recall 0,363636, F1 0,421053.
- Raw metrics trên được tạo trước khi boundary collector mới được benchmark full-video.
- Corrected FN waterfall của E2: 3 thiếu temporal support, 2 không có exported candidate, 1 association/context và 2 sparse evidence.
- 6 trong 13 FP là prediction trùng với GT đã được một prediction khác match.
- 94 tests đã pass trong lần audit V26, nhưng phải chạy lại trên code hiện tại; không sao chép con số cũ.
- Action LOSO checkpoints chỉ học từ rất ít subject, không có validation subject; sampled training accuracy 1,0 chỉ là tín hiệu cần kiểm tra overfit.
- Các score action hiện được ghi là uncalibrated nhưng vẫn được dùng với threshold cố định.
- Phone-event F1 >0,85 trên 20 GT đòi hỏi tối thiểu 18 TP và tối đa 3 FP khi TP=18. Đây chỉ là development gate, chưa chứng minh generalization.
- E2 dùng `action_mode=NONE`; vì vậy lỗi riêng của action adapter không được dùng để giải thích trực tiếp F1 của E2.
- Trong E4 và E5, cả 1.803 action scores hợp lệ đều có target timestamp sớm hơn observation nhận fusion khoảng 2,016 giây; lần audit đếm 563 observation dương ở E4 và 553 ở E5 dùng score lệch target. Đây là lỗi căn chỉnh đã quan sát, còn ảnh hưởng metric phải đo bằng replay sau sửa.

Không so trực tiếp Top-1 clip accuracy 92–98% trong paper với event precision/recall/F1 của ROADWATCH.

Ràng buộc số học cần dùng để ưu tiên công việc:

- Nếu loại cả 13 FP nhưng vẫn giữ 8 FN, TP/FP/FN=12/0/8 chỉ cho F1=0,75.
- Nếu phục hồi cả 8 FN nhưng vẫn giữ 13 FP, 20/13/0 chỉ cho F1≈0,7547.
- Vì vậy phải cải thiện đồng thời recall và precision; trên cohort 20 GT, tập counts thỏa cả event P/R/F1 >0,85 là TP∈{18,19,20}, FN=20−TP và FP≤3.
- Các con số trên chỉ là phép tính khả thi, không phải prediction sau sửa.
- Với chuỗi bốn retention gates cùng giữ 85% số GT sống qua gate trước, retention cuối chỉ 0,85⁴≈52,20%; nếu bốn gate bằng nhau và muốn retention tích 85%, mỗi gate cần khoảng 96,02%. Chỉ nhân các retention có điều kiện trên cùng quần thể; không nhân AP/accuracy/F1 của các dataset khác nhau.

## 3. Định nghĩa mục tiêu

Giá trị **chưa làm tròn** của mọi metric bắt buộc đã được thiết lập phải lớn hơn `0.85` trên cùng protocol và cùng candidate:

| Thành phần | Gate bắt buộc |
|---|---|
| Phone detector | Precision, Recall, F1 và mAP50 trên exhaustive bbox GT |
| Seatbelt detector | Precision, Recall, F1 và mAP50 trên exhaustive bbox GT |
| Action SAFE/PHONE/OTHER | Macro-Precision, Macro-Recall, Macro-F1 và P/R/F1 từng lớp |
| Seatbelt classifier 3 lớp | Accuracy, Macro-P/R/F1 và P/R/F1 từng lớp, gồm `uncertain_or_occluded` |
| Driver association | Association precision và driver-retention recall |
| Phone event end-to-end | Event precision, event recall và event F1 |

Ngoài accuracy, phải báo coverage, abstention, support, confusion matrix, FP/phút, fragmentation, temporal IoU, onset/offset error, alert latency và runtime. Không thay metric còn thấp bằng một metric dễ hơn.

Nếu thiếu exhaustive bbox GT hoặc role GT thì detector/association gate là `NOT_ESTABLISHED`, không được suy ra AP/precision/recall từ số candidate hoặc acceptance rate.

## 4. Quy tắc bảo toàn tính khoa học

1. Giữ nguyên ground truth gốc và frozen matcher cho bảng chính.
2. Không xóa, gộp, kéo dài hoặc đổi nhãn GT để làm điểm tăng. Nếu nghiên cứu protocol mới, version riêng và chạy lại cả baseline lẫn candidate.
3. Không dùng GT để chọn frame, crop, candidate, threshold, boundary hoặc nhánh inference.
4. Không tune trên independent holdout. Chỉ mở holdout sau khi code, config và checkpoint đã freeze.
5. Không thay tolerance, denominator hoặc metric sau khi xem candidate result.
6. Mọi search phải có grid hữu hạn được ghi trước, seed cố định và log đầy đủ.
7. Không tự bật `RIGHTS_ACCEPTED`, `TRAINING_ALLOWED`, label permission, upload permission hoặc holdout eligibility.
8. Không upload raw media, label, checkpoint hay prediction nếu guard không cho phép.
9. Không tắt fail-closed hoặc checkpoint/timestamp/identity safety để tăng recall giả.
10. Không reset/clean Git, không ghi đè artifacts V22–V26, không push, merge hay deploy.
11. Bảo toàn thay đổi của người dùng. Mọi run mới phải có code/config/data/model hash.
12. Metric phải sinh từ raw prediction bằng code, không nhập tay vào Markdown.
13. Không gọi cache replay là full-video inference và không gọi smoke test là benchmark toàn session.
14. Không tích hợp repository chỉ vì README có accuracy cao; phải kiểm tra license, commit, preprocessing, split, evaluator và checkpoint provenance.

## 5A. P0 bắt buộc — sửa candidate identity mismatch của raw runner

Trong `training/dmd/run_autonomous_v21.py`, action adapter hiện nhận `merged_detections[0]` trước khi pipeline xếp hạng association/handheld/geometry để chọn `best_cand`. Điều này cho phép action score của candidate A suppress hoặc boost event của candidate B.

Thực hiện theo thứ tự:

1. Viết regression test có ít nhất hai phone candidates: candidate A có confidence detector cao hơn nhưng không thuộc driver; candidate B có association tốt và được chọn cho event.
2. Chọn candidate bằng association/context/geometry trước khi gọi action phone-conditioned inference.
3. Chỉ gọi temporal adapter một lần cho candidate cuối tại mỗi timestamp/occupant; không cập nhật GRU state nhiều lần vì số candidate.
4. Giữ continuous occupant history độc lập với sự tồn tại của phone proposal.
5. Truyền một `candidate_id` ổn định qua detector → association → action → fusion → event trace.
6. Log rõ candidate dùng cho action và candidate dùng cho event; invariant yêu cầu hai ID giống nhau khi action score được áp dụng.
7. Không dùng candidate không liên kết driver để làm PHONE evidence cho driver.
8. Chạy test và một controlled action-enabled replay hoặc raw subset trước/sau qua đúng shared selection path để đo recovered TP, lost TP và new FP.

Cached `run_verified_v24.py` đã chọn `best_associated` trước action, còn E2 không bật action. Vì vậy test và ablation phải đi qua đúng raw runner có action hoặc shared pipeline mới; chạy lại E2 không chứng minh raw identity bug đã được sửa.

Không chuyển sang model lớn trước khi lỗi này được sửa và kiểm chứng.

## 5B. P0 bắt buộc — căn chỉnh action score với đúng thời điểm

Checkpoint hiện tại dùng cửa sổ 9 mẫu, label ở giữa và chu kỳ 0,5 giây. `predict_from_sequence()` trả `score_timestamp` ở tâm cửa sổ, nhưng cached runner hiện ghép score đó với candidate/pose/observation tại `available_at_timestamp`, muộn khoảng 2,016 giây.

Sửa theo hợp đồng sau:

1. Buffer record theo `(session_id, camera_id, occupant_stream_id, source_frame_id)` gồm embedding, pose, candidate đã association và timestamp.
2. Mỗi timestamp chỉ cập nhật temporal state một lần cho một occupant stream.
3. Khi đủ cửa sổ, lấy candidate, pose và rule features của **record target ở giữa**, rồi tính/fusion deep score cho chính record đó.
4. Đưa observation target vào event engine theo thứ tự `score_timestamp`; lưu riêng `available_at_timestamp` và thời điểm phát cảnh báo thực.
5. Không phát cảnh báo trước availability và không trình bày boundary quá khứ như latency bằng 0.
6. Ở cuối video khi thiếu future context, ghi score invalid hoặc dùng fallback đã khai báo; không tạo frame tương lai.
7. Viết test chuyển trạng thái PHONE→SAFE và SAFE→PHONE để chứng minh frame hiện tại không mượn score của target cũ.
8. Chạy lại E4/E5 trên cùng cached input và frozen matcher; báo timestamp invariant, action coverage, TP/FP/FN trước/sau.

Không được sửa manifest `CENTER` thành `LAST` cho checkpoint cũ. Nếu cần causal zero-lookahead, phải huấn luyện checkpoint có label cuối cửa sổ hoặc một temporal head causal riêng.

## 6. P0 bắt buộc — tạo benchmark raw parity

Chạy full raw-video với cùng một đường D5, cùng ROI, stride, detector weights, pose model, action checkpoint, temporal config và evaluator cho cả gC-14 lẫn gZ-36.

Pipeline bắt buộc:

`video decode → timestamp/sample → calibration/ROI → specialist + auxiliary detector → source-aware merge/NMS → pose/occupant tracking → driver-phone association → continuous action → fusion → temporal lifecycle → persistence/export → frozen evaluator`

Yêu cầu:

- Dùng implementation/config chung với backend hoặc trích logic chung, không duy trì hai phiên bản policy khác nhau.
- Chạy toàn session; smoke subset chỉ dùng kiểm tra kỹ thuật.
- Lưu decoded/sampled/dropped frame counts và timestamp thực.
- Lưu source detector cho từng proposal; không làm mất identity sau merge.
- Không so raw D2 với cached D5 để kết luận cải thiện thuật toán.
- Đối chiếu cached/live trên cùng frame khi có thể và giải thích mọi mismatch.
- Benchmark boundary collector mới, gồm open, extend, bridge, close, dedup và EOF flush.
- Mọi deep run phải chạy preflight: `requested_mode == actual_mode`, checkpoint/manifest hash hợp lệ và `forward_count > 0` sau warm-up. Fallback phải có reason và không được báo như kết quả deep.
- Expose `causal_mode`, `missing_checkpoint_policy` và device qua config/CLI. Research run yêu cầu deep dùng fail-fast thay vì tự fallback sang rule.

Nếu compute ngoài máy là cần thiết, chỉ chuẩn bị/upload gói code khi governance cho phép. Không đưa token hoặc secret vào repo, notebook, log hay báo cáo.

## 7. P0 bắt buộc — observability và FN/FP ledger

Mỗi sampled timestamp phải có trace đủ để replay quyết định:

- session, frame index, source timestamp, decode/sample status.
- ROI/calibration identity và tọa độ transform.
- mọi pose/person bbox, keypoint, confidence, track/occupant/role và ambiguity reason.
- mọi detector proposal ở điểm capture sớm nhất có thể và trạng thái sau từng filter. Phân biệt rõ output sau `YOLO.predict(conf=...)`, pre-project-filter và pre-NMS; không gọi `res.boxes` là pre-NMS nếu thư viện đã postprocess.
- source model, class, bbox, raw/calibrated confidence và candidate ID.
- association component scores, final state và reject reason.
- action window timestamps, valid mask, reset reason, logits/scores, `score_timestamp`, `available_at_timestamp` và actual mode.
- fusion inputs/output và reject/suppress reason.
- event state transition, active interval, close/drop reason và emitted prediction ID.
- code/config/data/model/calibration/matcher hashes.

Sinh lại ledger đủ 20 GT và toàn bộ FP từ raw run parity. Với mỗi FN, đi qua waterfall:

1. frame có được decode/sample trong interval;
2. đúng driver có trong ROI;
3. pose/track/role có hợp lệ;
4. detector có pre-filter proposal;
5. proposal có sống qua threshold/NMS/ROI/class filter;
6. proposal có liên kết đúng driver;
7. action window/score có hợp lệ và đúng candidate;
8. fusion có giữ observation;
9. lifecycle có mở/duy trì/đóng event;
10. prediction có bị matcher phạt vì boundary/fragmentation/duplicate.

Chỉ gán primary root cause khi có raw evidence. Tổng ledger phải khớp tuyệt đối TP/FP/FN trong `metrics.json`.

## 8. Tối ưu detector sau khi có bbox evidence

Tạo một validation subset có exhaustive phone bbox annotation, gồm cả frame không có phone, vật rất nhỏ, áp tai, che khuất bởi tay, dưới vô lăng và passenger phone. Split theo subject/session.

Trên cùng input, ablation một biến. Khóa grid trước khi chạy; grid khởi đầu cần kiểm tra gồm:

- confidence threshold với grid khóa trước;
- input resolution `{640, 960, 1280}` và driver ROI padding;
- specialist-only, auxiliary-only và calibrated ensemble;
- class-aware/class-agnostic NMS;
- source-aware weighted box fusion nếu có validation evidence;
- SAHI/sliced inference: off, tile 384 overlap 0,2 và tile 512 overlap 0,2; vẫn giữ full-ROI pass;
- max detections;
- detector/pose sampling stride `{15, 10, 5, 3}`;
- short-gap tracking/persistence có timeout, dùng low-score detection để duy trì track đã xác minh nhưng không tự biến prediction thành detection quan sát.

Đo pre-filter recall, post-filter P/R/F1/AP50 theo kích thước và occlusion, driver-associated recall, duplicates, event metrics và latency. SAHI chỉ được giữ nếu tăng recall/event F1 mà không phá precision, latency hoặc tạo duplicate quá mức.

Không tạo phone evidence chỉ từ tay gần tai khi detector hoàn toàn không có phone evidence; hand/face/ear là feature hỗ trợ hoặc một model interaction được đánh giá riêng.

## 9. Tối ưu association và tracking

Tạo hoặc dùng role/occupant validation labels độc lập. Kiểm tra:

- pose order đảo nhưng driver identity không đổi;
- passenger-first và nhiều người overlap;
- driver và passenger cùng cầm điện thoại;
- mirrored camera/calibration transform;
- phone ở rìa seat ROI;
- pose dropout ngắn và recovery;
- identity switch;
- occlusion và ambiguous seat assignment.

Thay heuristic pixel cố định bằng feature chuẩn hóa theo person/seat scale khi validation chứng minh tốt hơn. Duy trì track/hysteresis có timeout rõ ràng. Báo association precision, driver-retention recall, abstention coverage, identity switches và số event FP/FN do association.

Chuỗi `VERIFIED_STATIC_CALIBRATION` nội bộ không phải ground truth vật lý. Phải lưu calibration provenance và kiểm tra độc lập.

## 10. Tối ưu action recognition theo baseline tái lập

Trước khi train model mới:

1. Kiểm tra crop live/cache, màu, normalization, resize, class mapping và checkpoint fold.
2. Loại quy trình giảm ảnh xuống 64×64 rồi phóng lên 112×112 nếu ablation xác nhận mất tín hiệu phone/hand/ear.
3. Đo score distribution và confusion matrix trên TP, FN, FP, hard negative và từng subject.
4. Đo coverage: tỷ lệ timestamp có valid history, forward và score thực sự được fusion dùng.
5. Calibration chỉ fit trên validation subject, không trên evaluation/holdout.
6. Preflight phải ghi requested/actual mode, fallback reason, checkpoint hash, causal mode, forward count và tỷ lệ score hợp lệ/thực sự được fusion dùng.
7. Mọi thay đổi resize, normalization, cadence, label position hoặc sequence length tạo một preprocessing contract, feature cache và checkpoint mới; không dùng checkpoint 64→112/0,5 giây với input contract khác.

Khi governance cho phép training và nhãn hợp lệ, triển khai các baseline theo cùng group split, event decoder và compute report:

- MobileNetV2 temporal consensus theo DMD làm baseline nhẹ.
- ResNet/GRU hiện tại sau khi sửa preprocessing và candidate identity.
- DriveCLIP/CLIP embedding baseline theo implementation có thể tái lập.
- Causal TCN hoặc online TSM làm temporal baseline latency thấp.
- Local–global architecture: local phone/hand/face crop kết hợp global driver crop.
- VideoMAE chỉ thử sau các baseline nhẹ khi license, dữ liệu và compute cho phép; offline checkpoint không được gọi là causal nếu chưa chuyển và kiểm chứng.
- Multi-scale temporal head theo ý tưởng DualScope, chỉ khi code/checkpoint/preprocessing có provenance đủ.
- Class-balanced sampler, focal/class-balanced loss và hard-negative mining qua ablation riêng.

So sánh `NONE`, `RULE_ONLY`, `DEEP_ONLY`, `HYBRID` trên cùng proposals/association/event policy. Không dùng headline 98% từ paper làm kết quả của ROADWATCH.

Mọi window centered phải báo lookahead/availability. Online alert không được dùng future feature trước `available_at_timestamp` hoặc backdate thời điểm cảnh báo. Temperature scaling chỉ được fit trên validation logits, phải báo NLL/Brier/ECE trước/sau; nó không thay argmax và không tự sửa classification sai.

## 11. Tối ưu temporal localization và event lifecycle

Trước hết sửa/dedup heuristic hiện tại bằng ablation một biến:

- observation count và min-duration; grid khởi đầu `{0,2; 0,35; 0,5}` giây chỉ áp dụng khi cadence tạo đủ support;
- open/close threshold;
- allowed gap/occlusion bridge; grid khởi đầu `{0; 0,5; 1; 2}` và baseline hiện hành;
- cooldown;
- boundary extension;
- duplicate merge;
- EOF flush;
- stride/cadence.

Khóa grid trước khi chạy. Đặc biệt đo 3 FN thiếu temporal support và 6 duplicate FP. Phân tích V27 cho thấy stride 5 có thể tạo span ≥0,5 giây cho GT_10 nhưng chưa đủ cho GT_05/GT_12; stride 3 mới tạo span khoảng 0,504 giây cho GT_12. Đây chỉ là sampling support, không phải detector/event result.

Action scheduler phải độc lập với detector/pose cadence. Không đưa 9 frame ở 10 Hz vào GRU được huấn luyện cho 9 frame ở 2 Hz rồi coi là cùng model.

Sau khi có dense labels/features và baseline sạch:

- thử election/merge tương tự M2DAR;
- thử boundary head nhẹ;
- chỉ thử ActionFormer khi dữ liệu temporal đủ và evaluator giữ nguyên.

Chọn theo event P/R/F1 cùng fragmentation, FA/phút, temporal IoU, onset/offset error và online latency. Không dùng future smoothing cho kết quả online nếu không tính đúng độ trễ.

## 12. Seatbelt phải được cải thiện độc lập

Không tuyên bố toàn hệ thống đạt 85% chỉ vì phone-event đạt.

Thực hiện:

- Xác minh dataset/checkpoint/split/class mapping và license.
- Tạo person/session-disjoint split; không để frame gần nhau rò rỉ giữa train và validation.
- Giữ ba lớp `fastened`, `unfastened`, `uncertain_or_occluded`.
- Thử kiến trúc hai tầng: `observable vs uncertain`, sau đó `fastened vs unfastened` trên mẫu observable.
- Kiểm tra torso crop, dây mảnh, quần áo tối, che khuất, góc ghế và low light.
- Dùng class balancing, hard examples và augmentation chỉ khi governance cho phép.
- Báo per-class support/P/R/F1, macro metrics, calibration và variation theo subject.

Kết quả binary 98,25% của RST-Net chỉ là tham khảo kiến trúc; không thay thế macro-F1 ba lớp ROADWATCH.

## 13. Cách sử dụng nghiên cứu và GitHub

Các hướng ưu tiên và phạm vi đúng:

- DMD: baseline clip action MobileNetV2; không phải event localization.
- DriveCLIP: representation/action baseline; root repo công khai mô tả frame-based code. Repo `jiajingchen113322/DeepInsigth` được README dẫn cho VideoCLIP có source nhưng chưa đủ README/license/checkpoint/evaluator để coi headline đã tái lập.
- DualScope: tham khảo local detail + temporal multi-scale; GitHub công bố đang không truy cập được tại lần rà soát, checkpoint đơn lẻ chưa đủ tái lập.
- M2DAR: tham khảo election/multi-view temporal localization; metric overlap và tolerance khác ROADWATCH.
- ActionFormer: prototype boundary-aware sau khi có dense temporal training labels.
- SAHI: probe phone small-object recall với latency/duplicate measurement.
- ByteTrack: tham khảo cách dùng low-score detections để tiếp tục track; không được tự sinh phone evidence khi không có detection.
- TSM: baseline online causal/latency thấp trước khi dùng video Transformer lớn.
- VideoMAE: pretrained video encoder khi có đủ data/compute; phần lớn repo chính chủ dùng CC-BY-NC 4.0 nên phải kiểm tra mục đích sử dụng.
- Temperature scaling: calibration từ validation logits; repo minh họa gốc tự ghi không còn maintained, nên triển khai thuật toán nhỏ có test hoặc dùng thư viện đang duy trì.
- ZVL-DDD: học cách xuất raw predictions, per-class metrics, confusion matrix và ablation.

Với mỗi repo được dùng, khóa URL + commit/tag, ghi license, environment, checkpoint hash, preprocessing, split, command và thay đổi local. Chạy smoke reproduction trên subset trước khi tích hợp.

## 13A. Mở rộng dữ liệu và cảm biến khi pipeline sạch vẫn chưa đủ

- Mở rộng theo subject/session/camera đã được cấp quyền; không chỉ lấy thêm nhiều frame tương quan từ hai người.
- Chọn quy mô bằng learning curve theo số subject và dừng theo plateau đã định, không hứa một số lượng mẫu tùy ý sẽ đạt 85%.
- Thu thập hard negatives: chạm tai, uống nước, che mặt, vật nội thất giống phone, passenger phone và phone gắn giá đỡ.
- Nhãn độc lập gồm phone bbox, occupant role, visibility, action và event boundary. Pseudo-label/model-assisted label phải được người kiểm tra trước khi dùng làm GT evaluation.
- Bbox validation phải có frame âm tính đại diện toàn session và các mức kích thước/che khuất. Error-enriched set chỉ dùng chẩn đoán, báo riêng với representative set.
- Active learning chỉ chọn từ training/development pool bằng uncertainty/disagreement + diversity; không khai thác holdout.
- Horizontal flip/copy-paste chỉ dùng khi cập nhật đúng trái/phải, ghế, calibration và occupant semantics.
- Nếu đổi camera, độ phân giải hoặc thêm modality, version thành benchmark/input configuration mới và chạy baseline cùng input.
- Không bỏ allowlist hoặc dùng lại gZ-37/gE-28 đã consumed để tăng dữ liệu.

## 14. Ma trận thí nghiệm tối thiểu

Tạo plan trước khi chạy, mỗi run có đúng hypothesis và biến cần kiểm tra:

| Run | Mục đích |
|---|---|
| V27-B0 | Tái lập V26 E2 từ artifacts hiện tại |
| V27-ID | Sửa và ablation candidate identity mismatch |
| V27-TIME | Sửa target/availability alignment và tái chạy E4/E5 |
| V27-MODE | Fail-fast preflight cho requested/actual deep mode |
| V27-PARITY | Full raw D5 cùng cấu hình trên gC và gZ |
| V27-DIAG | FN waterfall + FP ledger từ parity run |
| V27-DET-* | Detector/SAHI/resolution/source calibration ablation |
| V27-ASSOC-* | Association/tracking ablation |
| V27-ACT-* | NONE/RULE/DEEP/HYBRID và action architectures |
| V27-EVT-* | Temporal lifecycle/boundary ablation |
| V27-SEAT-* | Seatbelt detector/classifier experiments |
| V27-CAND | Candidate duy nhất được chọn từ development |
| V27-FULL | Full-video/backend parity cho candidate frozen |
| V27-HOLDOUT | Một lần independent evaluation khi đủ điều kiện |

Mỗi run phải lưu:

- immutable run ID, hypothesis và status;
- command, exit code, start/end time;
- code/config/model/data/calibration/matcher hash;
- cohort, inclusion/exclusion và seed;
- raw predictions/trace;
- event matches;
- component và event metrics;
- coverage/abstention;
- runtime và hardware;
- failure/block reason.

Không ghi đè run khi hash thay đổi. Candidate chỉ được chọn bằng development protocol đã khóa.

Nếu có candidate đồng thời P/R>0,85, chọn F1 cao nhất trong runtime budget đã khóa. Nếu chưa có, dùng tiêu chí định trước `max min(precision, recall)`, rồi F1, để chọn hướng thí nghiệm tiếp theo và báo Pareto precision/recall/runtime. Không đổi tiêu chí sau khi xem kết quả. Report micro aggregate, từng subject, worst subject và coverage. Chỉ tính confidence interval theo subject/session độc lập khi đủ nhóm; không coi hàng nghìn frame tương quan là hàng nghìn mẫu độc lập.

## 15. Kiểm thử bắt buộc

Viết test có ý nghĩa cho thay đổi thực tế, tối thiểu gồm:

- action candidate ID luôn trùng event candidate ID;
- hai candidates không làm temporal state update hai lần;
- action score, pose, rule features và event candidate dùng cùng target frame ID;
- `available_at_timestamp >= sequence_end >= score_timestamp`, emission không sớm hơn availability;
- PHONE→SAFE và SAFE→PHONE không dùng centered score của record khác;
- deep research run fail nếu checkpoint centered không khớp causal policy; không fallback im lặng;
- requested/actual mode và forward count được kiểm tra qua đúng raw runner;
- passenger phone không gắn sang driver;
- pose order không đổi occupant identity;
- short pose dropout không đổi track;
- ambiguity vẫn UNKNOWN;
- continuous action history chạy khi không có phone proposal;
- duplicate/out-of-order timestamp không làm state sai;
- future feature bị chặn trước availability;
- invalid/uncalibrated score không bị coi là negative hợp lệ;
- pre-filter proposal và reject reason được lưu;
- lifecycle open/extend/bridge/close/dedup/EOF đúng;
- raw counts khớp metrics và report;
- cached/full runner dùng chung policy quan trọng;
- governance guard chặn training/upload/holdout entrypoint;
- frozen matcher không bị thay đổi.

Chạy focused tests sau mỗi thay đổi và full relevant suite trước quyết định cuối. Lưu JUnit và machine-readable test summary. Không sửa test để hợp thức hóa hành vi sai.

## 16. Runtime và tiêu chí triển khai

Đo toàn session, gồm decode và mọi model:

- startup/model load/warm-up;
- steady-state p50/p95/p99 latency;
- effective sampled FPS và source-video realtime factor;
- peak RAM/VRAM;
- dropped frames và queue behavior;
- latency từng stage;
- coverage khi model lỗi hoặc input thiếu.
- device thực tế của detector, pose và action; không suy GPU từ môi trường hoặc tên requirements khi code vẫn chạy CPU.
- environment fingerprint của Windows/Kaggle; báo rõ package nào đến từ base image thay vì requirements được khóa.

Chỉ production-approve khi full-video/backend parity, metric gates, runtime budget, governance và rollback đều pass. Accuracy cao nhưng quá chậm hoặc coverage thấp không đạt yêu cầu.

## 17. Governance và holdout

Đọc trạng thái live từ `research_permissions.py` và `holdout_guard.py`; không suy đoán từ báo cáo cũ.

Nếu fitting bị chặn, vẫn phải hoàn thành code correctness, observability, inference-only ablations, parity run, tests, manifests và deterministic training config. Ghi chính xác quyền/dataset/annotation còn thiếu.

Nếu fitting được phép:

1. snapshot dataset và hashes;
2. split theo subject/session;
3. freeze test/holdout trước search;
4. tune chỉ trên train/validation;
5. chọn đúng một candidate;
6. freeze code/config/checkpoint hash;
7. chạy independent holdout đúng một lần;
8. không quay lại tune theo holdout result.

Không đưa credential vào file. Nếu phát hiện token đã lộ trong lịch sử, không sử dụng hoặc lặp lại token đó; ghi yêu cầu rotate/revoke trong security note.

## 18. Artifacts bắt buộc

Tạo thư mục mới `reports/v27_ai85_execution/`, không ghi đè báo cáo cũ. Bàn giao tối thiểu:

- `FINAL_AI_LEAD_DECISION.md`
- `source_findings.md`
- `candidate_identity_audit.json`
- `action_timing_audit.json`
- `mode_coverage_audit.json`
- `fn_root_cause_ledger.json`
- `fn_waterfall.json`
- `fp_ledger.json`
- `metric_contract.json`
- `evaluation_protocol.json`
- `experiment_plan.json`
- `metrics_comparison.json`
- `component_metrics.json`
- `governance_status.json`
- `data_gap_manifest.json`
- `runtime_manifest.json`
- `reproducibility_manifest.json`
- `test_results.json`
- `junit_report.xml`
- `experiments/<run_id>/config.json`
- `experiments/<run_id>/raw_predictions.jsonl`
- `experiments/<run_id>/event_matches.json`
- `experiments/<run_id>/metrics.json`
- `experiments/<run_id>/runtime.json`
- `experiments/<run_id>/status.json`

Markdown phải được sinh hoặc đối soát từ JSON. TP/FP/FN, sample/frame/forward counts và runtime phải khớp raw records.

## 19. Trình tự thực hiện bắt buộc

1. Audit repository, governance và tái lập baseline V26.
2. Sửa candidate identity mismatch, action target/availability alignment và actual-mode preflight; thêm regression tests qua đúng runner.
3. Tái chạy E4/E5 trên cùng cache để đo riêng tác động timing/action; E2 làm control không action.
4. Hoàn thành raw D5 parity cho gC và gZ bằng boundary collector mới.
5. Sinh FN waterfall và FP ledger từ raw evidence.
6. Tạo/kiểm tra bbox và role validation labels; không giả lập metric khi nhãn thiếu.
7. Tối ưu bottleneck lớn nhất theo ablation một biến: detector → association → action → temporal.
8. Khi pipeline correctness đã pass nhưng dữ liệu/model vẫn yếu, mở rộng subject/data và thử baseline representation theo thứ tự chi phí đã nêu.
9. Tối ưu seatbelt độc lập.
10. Lặp lại cho đến khi mọi development gate đã thiết lập đạt >0,85 hoặc xuất hiện blocker thật sự.
11. Freeze candidate; chỉ chạy holdout khi đủ quyền và điều kiện.
12. Sinh toàn bộ artifacts và final decision.

Không hỏi người dùng về lựa chọn kỹ thuật thông thường. Chỉ hỏi khi thiếu một quyền hoặc thông tin không thể suy ra từ code/artifacts và thực sự chặn bước phụ thuộc; trong lúc chờ vẫn tiếp tục các nhánh độc lập.

## 20. Điều kiện kết luận

Chỉ ghi `DEVELOPMENT_TARGET_MET` khi mọi gate đã được thiết lập đều >0,85 trên candidate và protocol giống nhau.

Chỉ ghi `INDEPENDENT_TARGET_MET` khi candidate đã freeze, holdout thật sự độc lập, quyền hợp lệ và mọi gate chưa làm tròn đều >0,85.

Nếu development cải thiện nhưng chưa đạt, ghi `DEVELOPMENT_TARGET_NOT_MET` và báo chính xác baseline → candidate, recovered FN, lost TP, new FP và residual root causes.

Nếu metric không thể đo vì thiếu ground truth, ghi `NOT_ESTABLISHED`; không coi test pass, model load hoặc paper accuracy là bằng chứng đạt gate.

Phản hồi cuối bằng tiếng Việt, ngắn gọn nhưng đủ kiểm chứng, gồm:

- code đã sửa và nguyên nhân;
- bảng baseline/candidate cùng protocol;
- component gates và event gates;
- FN phục hồi, FP mới và lỗi còn lại;
- tests và thí nghiệm thật đã chạy;
- full-video runtime đúng phạm vi;
- governance/holdout status;
- kết luận MET/NOT_MET/NOT_ESTABLISHED;
- liên kết tới artifacts chính.

Mục tiêu là cải thiện hệ thống bằng bằng chứng tái lập cho đến khi các chỉ số hợp lệ vượt 85%. Tuyệt đối không đạt mục tiêu bằng data leakage, sửa ground truth, nới evaluator, bỏ lớp khó, chọn lại holdout hoặc tạo số liệu không có raw evidence.

## 21. Trạng thái thực thi V27 đã hoàn thành và điểm tiếp tục bắt buộc

Trước khi tiếp tục, đọc thêm:

- `reports/v27_ai85_execution/EXECUTION_REPORT.md`
- `reports/v27_ai85_execution/metrics_comparison.json`
- `reports/v27_ai85_execution/experiments/E2/`
- `reports/v27_ai85_execution/experiments/E4/`
- `reports/v27_ai85_execution/experiments/E5/`
- `training/dmd/audit_v27_execution.py`
- `tests/test_v27_ai85_execution.py`

Các việc P0 đã được triển khai, không làm lại từ đầu:

- raw runner chọn candidate trước action và gắn `candidate_id` xuyên suốt candidate/action/event;
- deep mode không được fallback âm thầm; requested mode, actual mode, checkpoint hash và forward count được lưu;
- centered checkpoint bị từ chối ở raw online runner khi chưa có delayed target-aligned output;
- cached E4/E5 dùng queue căn score về đúng target frame, đúng pose và đúng candidate; availability được ghi riêng;
- detector, pose và action nhận cùng tham số `device`;
- E2/E4/E5 đã chạy lại trong thư mục V27, không ghi đè V26.

Kết quả đo sau sửa:

| Run | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| E2 control | 12 | 13 | 8 | 0,4800 | 0,6000 | 0,5333 |
| E4 aligned deep | 12 | 13 | 8 | 0,4800 | 0,6000 | 0,5333 |
| E5 aligned hybrid | 12 | 14 | 8 | 0,4615 | 0,6000 | 0,5217 |

E4 tăng từ F1 0,5217 của V26 lên 0,5333 và loại được một FP phát sinh do đường action cũ, nhưng gate >0,85 vẫn `DEVELOPMENT_TARGET_NOT_MET`. Audit có 1.803 centered scores cho mỗi run E4/E5, 0 sai target frame, 0 sai target time trên 1 ms, 0 score xuất hiện trước target, 563 trường hợp có candidate được chấm và 0 candidate-ID mismatch.

Tiếp tục từ bước 4 của trình tự thực hiện: chạy raw D5 parity đầy đủ trên gC-14 và gZ-36 bằng code hiện tại, sau đó tạo FN waterfall/FP ledger mới. Không dùng E2/E4/E5 thêm để dò threshold. Ưu tiên đo detector proposal recall và association retention trên nhãn development độc lập; nếu chưa có bbox/role ground truth đầy đủ thì tạo review queue và ghi `NOT_ESTABLISHED`, không suy metric từ event GT.

Kaggle đã được dùng cho GPU preflight bằng private dataset `longngoc11/roadwatch-v27-ai85-execution` version 3 và private kernel `longngoc11/roadwatch-v27-gpu-preflight` version 3. Kernel hoàn tất trên Tesla T4; specialist, auxiliary và pose đều chạy tại `cuda:0`, candidate-identity regression pass. Kết quả và hash nằm trong `reports/v27_ai85_execution/kaggle_execution_receipt.json` và `reports/v27_ai85_execution/kaggle_gpu_preflight_v3/v27_gpu_preflight.json`.

Không diễn giải preflight này thành accuracy. Nó dùng 32 ảnh validation private đã consumed, không dùng DMD media, không train và không tính event metric. Danh sách private Kaggle datasets hiện không có video DMD gC-14/gZ-36; quyền nguồn hiện chỉ cho phép external upload code. Vì vậy raw D5 trên Kaggle đang bị chặn đúng lý do `BLOCKED_NO_AUTHORIZED_DMD_MEDIA_ON_KAGGLE`. Không upload video/nhãn DMD để vượt blocker này; raw parity chỉ được chạy local hoặc trên storage có quyền được ghi nhận rõ ràng.

## 22. Kết quả Kaggle component mới và chỉ dẫn tiếp tục

Đọc thêm các bằng chứng đã kiểm tra:

- `reports/v27_ai85_execution/kaggle_phone_component_receipt.json`
- `reports/v27_ai85_execution/kaggle_phone_component_v2/v27_phone_component_eval.json`
- `kaggle/v27_phone_component_eval/run_phone_component_eval.py`
- kernel private `longngoc11/roadwatch-v27-phone-component-eval`, version 2.

Kernel đã chạy checkpoint phone detector cố định trên Tesla T4 với 1.782 ảnh validation bootstrap, 181 ảnh có phone và 229 bbox phone. Không train, không truy cập test split và không dùng DMD media. Kết quả:

| Input | Precision | Recall | F1 | mAP50 | mAP50-95 | Inference ms/ảnh |
|---:|---:|---:|---:|---:|---:|---:|
| 640 | 0,8809 | 0,8210 | 0,8499 | 0,8974 | 0,6433 | 7,99 |
| 960 | 0,9395 | 0,9039 | 0,9214 | 0,9456 | 0,7497 | 19,83 |
| 1280 | 0,8257 | 0,7817 | 0,8031 | 0,8527 | 0,6189 | 38,90 |

960 là cấu hình tốt nhất theo F1 chẩn đoán trong ba độ phân giải đã khai báo trước. Không tiếp tục tăng resolution theo suy đoán vì 1280 đã giảm đồng thời precision, recall và tốc độ. Khi có bbox ground truth được người có thẩm quyền duyệt, tái chạy đúng ma trận này trên validation được quản trị; chỉ sau đó mới được dùng 960 làm candidate detector chính thức.

Nhãn hiện có trạng thái `PROPOSAL_MODEL_ONLY_NOT_GOVERNED_FINAL_DATA`, vì vậy các số trên chỉ là chẩn đoán tạm thời, không đủ điều kiện model selection hoặc tuyên bố production. Đây là bbox component metric, không phải event metric DMD. Quyết định hệ thống vẫn là `DEVELOPMENT_TARGET_NOT_MET` với P=0,48, R=0,60, F1=0,5333. Tiếp tục ưu tiên tạo/review bbox và role labels độc lập, chạy raw D5 trên storage được phép, lập FN waterfall và FP ledger, rồi mới tối ưu detector → association → action → temporal theo từng biến.

Private dataset `longngoc11/roadwatch-v27-ai85-execution` đã được cập nhật lên version 6 với package code-only gồm 42 tệp đúng manifest. Package chứa kernel component, raw D5 audit, aggregate result, ablation auditors và báo cáo; không chứa `__pycache__`, bytecode, credential, DMD media, annotation, embedding, checkpoint hoặc raw prediction. Builder phải luôn tạo lại package sạch và fail nếu xuất hiện tệp ngoài manifest.

## 23. Raw D5 full-cohort đã hoàn tất

Không chạy lại baseline này nếu code/config/hash không thay đổi. Đọc:

- `reports/v27_ai85_execution/RAW_D5_REPORT.md`
- `reports/v27_ai85_execution/raw_d5_full_none/aggregate_metrics.json`
- `reports/v27_ai85_execution/raw_d5_full_none/fn_waterfall.json`
- `reports/v27_ai85_execution/raw_d5_full_none/fp_ledger.json`
- `reports/v27_ai85_execution/raw_d5_full_none/mode_coverage.json`
- `reports/v27_ai85_execution/raw_d5_full_none/runtime.json`
- `training/dmd/audit_v27_raw_d5.py`

Raw D5 đã chạy đủ 27.267 frame nguồn của gC-14 và gZ-36, stride 15, tổng 1.819 sampled frames, CPU, specialist + auxiliary, verified association, `action_mode=NONE`, action gating tắt. Không fitting và không dò threshold.

| Run/subject | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| gC-14 raw D5 | 7 | 1 | 2 | 0,8750 | 0,7778 | 0,8235 |
| gZ-36 raw D5 | 7 | 2 | 4 | 0,7778 | 0,6364 | 0,7000 |
| **Micro raw D5** | **14** | **3** | **6** | **0,8235** | **0,7000** | **0,7568** |
| Cached E2 tham chiếu | 12 | 13 | 8 | 0,4800 | 0,6000 | 0,5333 |

Raw D5 hiện là bằng chứng end-to-end development tốt nhất, nhưng không diễn giải chênh lệch so với cache thành tác động một biến vì hai execution path khác nhau. Gate vẫn `DEVELOPMENT_TARGET_NOT_MET`; cần tối thiểu 18 TP và không quá 3 FP, tức phải phục hồi thêm bốn TP mà không tạo FP mới.

Trong sáu FN có bốn event ngắn dưới một giây. Ba FN bị association UNKNOWN loại phần lớn/toàn bộ evidence; một FN dài 21,47 giây có proposal thưa và toàn bộ evidence bị loại; một FN không có proposal; một FN chỉ có một observation được admit nên không đủ temporal support. Bước tiếp theo phải là ablation một biến cho association retention trên toàn cohort, giữ nguyên detector, stride, temporal config và matcher. Kết quả association-bypass chỉ là upper-bound diagnostic, không được production-approve vì bỏ bảo vệ passenger/unknown.

Association-bypass đã được đăng ký trước và chạy đủ cohort. Kết quả event không đổi ở cả hai subject. Trong 1.819 observations, 89 rejection labels đổi từ `UNVERIFIED_OCCUPANT_ASSOCIATION` thành `PHONE_VISIBLE_WITHOUT_HANDHELD_USE`, nhưng số observation được admit vẫn đúng 537. Vì vậy không tiếp tục ưu tiên bỏ association; nguyên nhân đo được nằm ở handheld-context geometry hoặc temporal support. Đọc `reports/v27_ai85_execution/raw_d5_assoc_bypass/ablation_result.json`. Bước tiếp theo là handheld-context upper-bound một biến với verified association bật; mọi bypass vẫn chỉ là diagnostic và không đủ điều kiện production.

Handheld-context upper-bound cũng đã đăng ký trước và chạy đủ cohort. Cấu hình force `HANDHELD_USE` admit thêm 42 observations, phục hồi một event gZ-36 dài 21,47 giây nhưng tạo thêm ròng hai FP. Micro đổi từ 14/3/6 thành 15/5/5, P=0,75, R=0,75, F1=0,75; thấp hơn baseline F1=0,7568. Loại cấu hình force context, không production-approve. Đọc `reports/v27_ai85_execution/raw_d5_handheld_upper_bound/ablation_result.json`. Hướng tiếp theo là cải thiện phone-hand/face evidence có nhãn được quản trị, proposal coverage và cadence cho event dưới một giây; không tiếp tục bypass safeguard.

Focused suite sau thay đổi mới nhất pass 103/103 tests. Bằng chứng máy đọc nằm tại `reports/v27_ai85_execution/test_results.json` và `junit_report.xml`.
