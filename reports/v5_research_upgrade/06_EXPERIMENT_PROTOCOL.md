# Giao thức thực nghiệm

## Population và grain

- 3 development subjects: `gB-9, gZ-36, gC-14`; 22.063 video-minutes.
- 29 phone-use ground-truth events từ annotation DMD thật.
- B2 có 2,616 và B3 có 2,604 heldout LOSO clips sau biên sequence; clip grain khác event grain.
- Labels: SAFE_DRIVING, TEXTING_LEFT/RIGHT, PHONECALL_LEFT/RIGHT, OTHER_ACTIVITY; UNKNOWN chỉ abstain.

## LOSO và chống leakage

Mỗi fold giữ một subject để test, train hai subject còn lại. Threshold grid 0.40–0.80 được chấm **chỉ
trên training subjects** của fold, ưu tiên event F1 với FA/min ≤0.30. Không frame/clip của test subject
đi vào training hoặc threshold selection. `gZ-37` và `gE-28` không truy cập.

## Metrics

Clip: accuracy, macro-F1, per-class P/R/F1, confusion matrix. Event: frozen matcher, TP/FP/FN,
P/R/F1, FA/min, latency, duplicates và fragmentation. Pooled rates được tính từ pooled counts, không
trung bình các fold. Runtime: CPU throughput, p50/p95 inference latency và sampled process RSS.

## Ablation registry

- B0: frozen V4 object/geometry — measured.
- B1: original detector/fusion — không chạy mới; V3 historical artifact không dùng để chọn V5.
- B2: TinyCNN temporal action — measured LOSO.
- B3: frozen ResNet-18 + bidirectional GRU — measured LOSO ở vòng 2.
- B4: CLIP — not run (full VideoCLIP artifact chưa xác minh, compute cao).
- B5: union object + action — measured.
- B6: reliability-gated action + object + temporal engine — measured.
- B7: DualScope-inspired full candidate — not run (code/weights chưa tái lập, GPU unavailable).
- B8: B3 + frozen V4 late fusion — measured; vượt target P/R/F1 nhưng trượt FA/min.
- Research selection vẫn là B2 theo rule khóa; chưa phải deployment model.

## Reproduction

Chạy vòng 1 bằng `py -3.11 -m training.dmd.run_v5_experiments`; chạy vòng 2 bằng
`py -3.11 -m training.dmd.run_v5_b3_experiment`, cùng config `models/model_config_v5_research.yaml`.
Input SHA `e8590f2438ec9655314e8188eac72fd8666b029c3354cebab9f079556865070a`; git HEAD `f9641dbc0ce60e317960f6edcc9043c92e3ef4ab`. Checkpoint reuse chỉ được chấp
nhận nếu heldout/train subjects và architecture metadata khớp.
