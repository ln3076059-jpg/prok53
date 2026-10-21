# ROADWATCH — đối chiếu nghiên cứu, GitHub và nhược điểm hệ thống

Ngày rà soát: 21/10/2026. Mục tiêu của tài liệu này là chọn hướng kỹ thuật có thể tái lập, không cộng hoặc so trực tiếp các metric khác bản chất. Số tham chiếu ROADWATCH hiện tại là E2 cached D5 trên development: TP/FP/FN = 12/13/8, precision 0,48, recall 0,60, F1 0,533333.

## 1. Những công trình có điểm cao và mức độ tương ứng

| Công trình | Kết quả được báo cáo | Code/checkpoint chính chủ | Mức tương ứng với ROADWATCH |
|---|---|---|---|
| DMD, Ortega et al. 2020 | 92,0% test Top-1, RGB MobileNetV2 MaxP-MLP, 4 segments; split xấp xỉ 24/7/6 người | [DMD GitHub](https://github.com/Vicomtech/DMD-Driver-Monitoring-Dataset) | Cùng miền DMD và action semantics; vẫn là clip classification, không có event P/R/F1 hay driver-phone association |
| DriveCLIP, Hasan et al. 2024 | VideoCLIP 98,44% Top-1 DMD; Multi-frameCLIP 93,39%; Single-frame 82,65%; zero-shot 48,12% | [DriveCLIP GitHub](https://github.com/zahid-isu/DriveCLIP) | Phù hợp để thử representation/action head. README nói codebase frame-based, nên nhánh VideoCLIP 98,44% chưa được repo hướng dẫn đầy đủ như headline paper |
| DualScope, Qiu et al. 2026 | 98,63% DMD Top-1; 98,56% SAM-DD; 89,66% SynDD1 | [AAAI paper](https://ojs.aaai.org/index.php/AAAI/article/view/37812), [checkpoint](https://huggingface.co/Qqqzzj/DualScope) | Spatial local detail + temporal multi-scale rất sát nhược điểm action branch; GitHub tác giả được model card dẫn tới hiện trả 404, nên chưa có đường tái lập code đầy đủ |
| M2DAR, Ma et al. 2023 | Average activity overlap 0,5921 trên AI City A2 test | [M2DAR GitHub](https://github.com/PurdueDigitalTwin/M2DAR) | Gần ROADWATCH nhất về untrimmed localization. Dùng nhiều camera và protocol cho phép start/end lệch ±10 giây, nên số 0,5921 không so trực tiếp với frozen matcher của ROADWATCH |
| ActionFormer, Zhang et al. 2022 | 71,0% mAP@tIoU 0,5 trên THUMOS14 | [ActionFormer GitHub](https://github.com/happyharrycn/actionformer_release) | Có classification + boundary regression, phù hợp thay heuristic event sau khi có feature/labels; score là benchmark TAL khác miền |
| SAHI, Akyon et al. 2022 | Inference slicing tăng AP 6,8/5,1/5,3 điểm; cộng dồn với fine-tuning 12,7/13,4/14,5 | [SAHI GitHub](https://github.com/obss/sahi), [benchmark GitHub](https://github.com/fcakyon/small-object-detection-benchmark) | Có thể tăng recall điện thoại nhỏ; kết quả gốc là ảnh aerial, có rủi ro tăng duplicate và latency, bắt buộc đo bằng phone bbox GT |
| ZVL-DDD, Miyata et al. 2026 | 76,0% multiclass Top-1 SAM-DD nhưng 95,8% binary AUPRC | [ZVL-DDD GitHub](https://github.com/mtakamichi/ZVL-DDD) | Không phải ứng viên >85 multiclass/event; repo đáng học ở xuất prediction, confusion matrix, per-class metrics và ablation tái lập |
| RST-Net seatbelt, Qiu et al. 2024 | 98,25% binary accuracy/F1 trên dữ liệu tự xây dựng | Chưa xác minh được author GitHub | Hai tầng windshield ROI + classifier có ích; không tương đương macro-F1 ba lớp của ROADWATCH có `uncertain_or_occluded` |

Nguồn số liệu: [DMD paper](https://arxiv.org/pdf/2008.12085), [DriveCLIP paper](https://arxiv.org/pdf/2306.10159), [DualScope full text](https://openreview.net/pdf/2fb01354133f2c1096fe9b530f0e954f29492ac0.pdf), [M2DAR paper](https://arxiv.org/pdf/2305.08877), [ActionFormer paper](https://arxiv.org/abs/2202.07925), [SAHI paper](https://arxiv.org/abs/2202.06934), [RST-Net paper](https://www.mdpi.com/2076-3417/14/8/3346). Chi tiết máy đọc được nằm trong `research_github_matrix.json`.

### Ý nghĩa của các headline 92–98%

Các nghiên cứu DMD/DriveCLIP/DualScope trả lời câu hỏi “clip/frame này thuộc action nào?”. ROADWATCH phải trả lời đồng thời “có điện thoại vật lý không, thuộc tài xế nào, có đang sử dụng không, sự kiện bắt đầu/kết thúc lúc nào?”. Một action classifier có Top-1 98% vẫn có thể tạo nhiều FP event nếu prediction nhấp nháy hoặc boundary sai.

M2DAR là đối chứng quan trọng: khi chuyển sang video không cắt sẵn và chấm cả biên, công trình có code chỉ đạt overlap 0,5921 dù dùng multi-view Transformer. Điều này không chứng minh ROADWATCH tốt; nó chứng minh mục tiêu event-level khó hơn nhiều so với clip accuracy.

## 2. Nhược điểm đã xác nhận trong ROADWATCH

### P0 — Chưa có benchmark end-to-end đồng nhất

E2–E5 đọc trace D5 lịch sử, gồm specialist và auxiliary. Raw run hoàn tất lại dùng D2 specialist-only. Raw gZ sau seat fix đạt 4/4/7, nhưng gC chưa hoàn tất trên cùng code. Vì detector/input khác nhau, hiện chưa biết suy giảm đến từ detector, association hay temporal collector bao nhiêu. Đây là blocker lớn nhất trước khi thay kiến trúc.

### P0 — Action và event có thể dùng hai candidate khác nhau

Trong `run_autonomous_v21.py`, action adapter nhận `merged_detections[0]`, tức candidate đứng đầu theo confidence sau NMS. Sau đó event pipeline xếp lại tất cả candidate bằng driver association, handheld state, geometry và confidence để chọn `best_cand`. Khi hai candidate khác nhau, score SAFE/PHONE của vật thứ nhất được dùng để suppress/boost vật thứ hai. Cần tính action score theo từng candidate hoặc chỉ gọi action adapter sau khi đã chọn candidate bằng cùng identity.

### P0 — Không có ground truth ở các tầng cần thiết

Không có exhaustive bbox GT trước filter cho điện thoại, role/occupant GT độc lập và seatbelt bbox/visibility GT. Vì vậy:

- Không đo được detector recall/AP thật, chỉ biết số candidate sau export.
- Không đo được association precision và driver-retention recall độc lập.
- Không biết hai GT dài không có candidate là detector miss, ROI miss, confidence filter hay NMS.
- Không thể dùng RST-Net 98,25% binary để thay gate ba lớp hiện tại.

### P0 — Tập đánh giá quá nhỏ và đã bị xem lặp lại

Phone-event benchmark hiện có 20 GT từ gC-14/gZ-36. Cohort đã được dùng nhiều vòng để sửa code và chọn cấu hình, nên là development chứ không còn là holdout. Với 20 GT, một event làm recall thay đổi 5 điểm phần trăm. Không có confidence interval theo người và không chứng minh tổng quát sang tài xế/camera khác.

### P1 — Action model thiếu dữ liệu và mất chi tiết không gian

- Mỗi LOSO checkpoint gC/gZ chỉ học từ hai người; `validation_subject_ids=[]`.
- Fold có 909/795 samples; sampled training accuracy cuối bằng 1,0. Đây là tín hiệu cần kiểm tra overfit, chưa phải bằng chứng overfit độc lập.
- Driver crop bị giảm xuống 64×64 rồi nội suy lên 112×112 trước ResNet18. Điện thoại, ngón tay và vùng tai là tín hiệu nhỏ; nội suy không khôi phục chi tiết đã mất.
- Score được ghi `UNCALIBRATED_SCORES`; threshold SAFE 0,65, PHONE 0,20/0,40 đang được dùng như xác suất tin cậy dù chưa calibration theo người.
- Window 9 mẫu × 0,5 giây với label ở giữa cần khai báo khoảng 2 giây lookahead. Nó không phải strict causal zero-latency.

DualScope gợi ý local behavior crops + global context và temporal pyramid; DriveCLIP gợi ý pretrained ViT/CLIP representation. Hai hướng này giải quyết đúng các hạn chế trên hơn việc tiếp tục tune GRU threshold.

### P1 — Association dựa nhiều vào heuristic và “verified” nội bộ

- Static driver ROI `[0, 0.10, 0.65, 0.95]` và trạng thái `VERIFIED_STATIC_CALIBRATION` do chính project ghi, chưa có measurement/role GT độc lập.
- Runner lấy pose có confidence cao nhất trong driver crop, không có track đa người ổn định.
- Association dùng containment và bán kính proximity pixel cố định; thay camera/resolution/tư thế có thể đổi phân phối.
- Nếu pose mất, hệ thống fail closed sang UNKNOWN. Điều này giảm FP nhưng gây recall thấp; lần raw đầu đã biến toàn bộ 262 candidate thành UNKNOWN.

### P1 — Detector ensemble chưa được calibration

D5 NMS trực tiếp candidate specialist và COCO auxiliary bằng IoU 0,5 rồi so confidence giữa hai model. Confidence của hai model không nhất thiết cùng thang. Cross-model NMS có thể giữ false candidate có confidence cao và loại phone thật. SAHI/high-resolution ROI đáng thử, nhưng phải giữ source identity, calibrate score theo source và đo duplicate rate.

### P1 — Temporal policy còn mang tính heuristic

Config dùng stride 15 (~0,504 giây/mẫu), hai observations, bridge 4 giây và cooldown 3 giây. Ba GT ngắn chỉ có một mẫu trong interval; 6/13 FP là prediction trùng GT đã match. Hạ threshold có thể phục hồi FN nhưng tăng fragmentation/FP. ActionFormer hoặc head boundary-aware là hướng hợp lý sau khi có dense temporal labels; trước đó cần sửa collector và dedup bằng ablation một biến.

### P1 — Quan sát/log chưa đủ để replay mọi tầng

Raw trace hiện không lưu toàn bộ pre-filter logits/proposals và pose/keypoint cho mọi frame. Không thể tái chạy confidence/NMS/association ablation hoàn toàn từ cache. Repo ZVL-DDD là mẫu tốt về xuất per-sample predictions, classwise metrics, confusion matrix và ablation CSV.

### P2 — Runtime và deployment chưa được chứng minh

Raw D2 gZ xử lý 1024 mẫu trong khoảng 392 giây (~2,61 sampled frames/s). D5 thêm auxiliary model; action backbone, pose, detector đều chạy CPU trong runner. Chưa có benchmark model-inclusive p50/p95 latency, memory, throughput và dropped-frame behavior trên phần cứng mục tiêu. Điểm accuracy cao trong paper không giải quyết giới hạn thời gian thực này.

## 3. Kiến trúc nên thử theo thứ tự bằng chứng

1. **Sửa identity mismatch:** chọn candidate trước, rồi tính rule/action/fusion cho đúng candidate đó; thêm regression test hai candidate A/B.
2. **Thiết lập raw parity:** D5, cùng ROI/stride/config/checkpoint cho gC và gZ; lưu pre/post-filter proposal, pose và transition. Không tối ưu threshold trong bước này.
3. **Tạo phone bbox/role validation set:** đo AP/recall theo kích thước, source detector, người và tình trạng che khuất; mới quyết định SAHI/high-resolution/specialist retraining.
4. **Action baseline tái lập:** MobileNetV2 temporal consensus theo DMD; DriveCLIP ViT-L/14 embedding; local-hand/face/phone branch + global crop. Cùng group split và cùng event decoder.
5. **Temporal localization:** trước hết election/merge kiểu M2DAR trên validation; nếu đủ nhãn và dữ liệu, thử ActionFormer hoặc boundary head nhẹ.
6. **Seatbelt:** observable-vs-uncertain trước, fastened-vs-unfastened sau; person-disjoint split. Không học từ ảnh uncertain như unfastened.
7. **Candidate freeze:** chỉ sau khi validation hoàn tất mới chạy independent subjects một lần và báo cáo event P/R/F1, bbox AP/recall, macro-F1, FP/phút và latency.

## 4. Tiêu chí chọn GitHub để tích hợp

Không clone/integrate chỉ vì README ghi accuracy cao. Một repo chỉ được đưa vào candidate khi có:

- Paper và license xác định; commit hoặc tag được khóa.
- Code inference/evaluation tương ứng đúng model tạo ra headline metric.
- Split theo người/session và danh sách class rõ ràng.
- Checkpoint/preprocessing/model input có hash hoặc metadata kiểm tra được.
- Evaluator xuất raw predictions, không chỉ screenshot metric.
- Có thể chạy trên một subset nhỏ mà không sửa GT/evaluator.

Theo tiêu chí này, ưu tiên kỹ thuật hiện tại là **SAHI probe**, **DriveCLIP representation baseline**, **M2DAR election idea** và **ActionFormer boundary prototype**. DualScope là hướng kiến trúc đáng nghiên cứu nhưng chưa nên tích hợp trực tiếp khi GitHub được công bố đang 404; checkpoint đơn lẻ không đủ để xác minh preprocessing và evaluator.

## 5. Kết luận kỹ thuật

Nhược điểm lớn nhất hiện tại không phải “model quá nhỏ”. Hệ thống chưa có phép đo end-to-end đồng nhất và action/event candidate identity đang có thể lệch nhau. Sau đó mới đến thiếu local detail, ít người huấn luyện, temporal heuristic và thiếu bbox/role GT. Thay ResNet18 bằng model lớn ngay lúc này có thể tăng compute nhưng không sửa association UNKNOWN, candidate mismatch, duplicate FP hoặc boundary sai.

Mục tiêu >0,85 vẫn chưa được chứng minh. Trên development 20 GT, điều kiện đếm tối thiểu là TP≥18, FN≤2, FP≤3. Nghiên cứu khoa học cung cấp kiến trúc và baseline để thử; không cung cấp phép chuyển trực tiếp từ Top-1 98% sang event-F1 85%.
