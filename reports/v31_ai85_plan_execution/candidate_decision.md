# V31 candidate decision — 23/10/2026

## Quyết định

Không promote bất kỳ action candidate V31 nào. Giữ `D5`, `imgsz=640`, `stride=15`, `action_mode=NONE` làm baseline phát triển: **14 TP / 3 FP / 6 FN**, Precision **82,35%**, Recall **70,00%**, Event F1 **75,68%**.

Đây vẫn là development result trên gC-14/gZ-36, không phải independent-holdout claim.

## Evidence đã chạy

| Candidate | TP/FP/FN trên gC/gZ | Precision | Recall | F1 | Quyết định |
|---|---:|---:|---:|---:|---|
| D5-640 baseline | 14/3/6 | 82,35% | 70,00% | 75,68% | Giữ control |
| D5-960 | 13/3/7 | 81,25% | 65,00% | 72,22% | Loại |
| D2 specialist-only 640 | 9/7/11 | 56,25% | 45,00% | 50,00% | Loại |
| D3 auxiliary-only 640 | 14/3/6 | 82,35% | 70,00% | 75,68% | Diagnostic; không hơn control |
| Direct112 weighted, class-aware segmentation | 10/18/10 | 35,71% | 50,00% | 41,67% | Loại |
| Direct112 unweighted, class-aware segmentation | 9/19/11 | 32,14% | 45,00% | 37,50% | Loại |
| Coarse weighted, action-only generic | 14/19/6 | 42,42% | 70,00% | 52,83% | Loại |
| Coarse weighted, object gated | 12/0/8 | 100,00% | 60,00% | 75,00% | Loại vì recall |
| Coarse weighted, object + recovery | 14/7/6 | 66,67% | 70,00% | 68,29% | Loại vì không cứu TP, tăng FP |
| Coarse unweighted, object + recovery | 14/9/6 | 60,87% | 70,00% | 65,12% | Loại |

V2 temporal-only rescore của direct112 weighted cũng không đạt: segmentation **14/14/6**, F1 **58,33%**; recovery **14/15/6**, F1 **57,14%**. Vì vậy thất bại không chỉ do class-aware metric nghiêm hơn.

## Kết luận kỹ thuật

1. Tăng detector resolution không bảo đảm tăng proposal recall; specialist ở 960 sụp mạnh trên gZ-36.
2. Auxiliary-only đạt cùng tổng metric với D5 nhưng event không giống hệt. D3 tạo fragmentation/duplicate trong các phone-call dài; specialist có thể bridge boundary nhưng đồng thời tạo FP riêng trên hand/clothing/neck. Không được diễn giải “specialist vô dụng” chỉ từ tổng 14/3/6.
3. Direct112 giữ chi tiết tốt hơn rõ rệt và gần như không giảm encoder throughput ở 112px, nhưng domain generalization vẫn kém: gZ-36 suy giảm. Crop resolution không giải quyết thiếu subject diversity.
4. Coarse head tăng PHONE_USE clip recall nhưng gần như không nhận SAFE: pooled SAFE recall chỉ 1,35–1,65%. Recovery vì thế tạo nhiều positive interval ngoài GT và vượt ngân sách FP.
5. Baseline còn thiếu tối thiểu bốn TP để đạt 18/3/2. Những FN đã audit gồm event dưới một giây, association/context dropout, localization sai và một event dài có phone rõ. Một threshold toàn cục không tách được TP/FP an toàn.

## Gate tiếp theo

- Không train thêm GRU từ ba subject hiện tại và không tune lại decoder trên gC/gZ.
- Hoàn thiện component truth có human review cho phone bbox/state/driver association trên toàn bộ 6 FN, 3 FP và negative windows đại diện.
- Đánh giá verifier/context bằng out-of-fold subject split; chỉ chạy full video khi oracle cho thấy có thể cứu ít nhất bốn FN mà không thêm FP.
- Dùng live object decision stream đã tích hợp làm anchor causal; action evidence không được tự mở event trước khi qua validation.
- Thêm subject development trước khi chọn representation tiếp theo. gE-28/gZ-37 đã consumed và không được gọi là untouched holdout.
- Chỉ freeze candidate khi cùng Precision, Recall, F1 >85% trên validation subject-disjoint; sau đó mới chạy một lần trên independent holdout.

## Provenance

- Registration: `direct112_experiment_registration.json`, `coarse_action_experiment_registration.json`.
- Direct crop benchmark: `direct_crop_benchmark_gC14.json`.
- Live decision validation: `live_decision_audit.json` và `d5_live_decision_full/`.
- Unified metric rescore: `metric_protocol_matrix.json`.
- Review gate: `component_truth_manifest.json` — 9/9 case đang chờ human review, `training_eligible=false`.
- Candidate results: `reports/v31_direct112_causal_action*/` và `reports/v31_direct112_coarse_action*/`.

