# Thiết kế Roadwatch V5 dual-path

## Luồng nghiên cứu

```text
RGB BODY clip ──> driver ROI ──> Tiny 2D CNN/frame ──> mean|max|last-first ──> action logits
      │                                                        │
      └──────────────── timestamps ────────────────────────────┤
                                                               v
YOLO/geometry events ───────────────────────────────> governed late fusion ──> temporal event
                                                               │
                                                NEEDS_REVIEW / PENDING
```

### Path A — object & geometry

V4 YOLO phone confidence, pose và multi-view fusion được giữ nguyên làm frozen comparator. V5 không
thay detector vì chưa có same-data benchmark. Bbox/driver association không có trong feature cache nên
không được giả lập.

### Path B — independent video action

`TinyTemporalCNN` xử lý 5 frame liên tiếp, ROI BODY chuẩn hóa 64×64. Ba Conv-BN-SiLU block tạo 32-D
feature mỗi frame; temporal descriptor ghép mean, max và `(last-first)` rồi phân loại 6 lớp. Loss là
class-weighted cross entropy với label smoothing 0.03. `UNKNOWN` là abstention, không được phát minh
thành ground truth. Thiết kế lấy cảm hứng từ behavior-centric crop, salient motion/difference và video
semantics nhưng là implementation độc lập, nhỏ, phù hợp CPU.

Vòng 2 bổ sung encoder ResNet-18 `IMAGENET1K_V1` đóng băng, embedding 512-D và bidirectional GRU
(hidden size 64) trên chuỗi 9 frame. Backbone weights có SHA-256 và provenance riêng; chỉ temporal head
được huấn luyện trên hai training subjects của từng fold.

### Path C — fusion & decision

Xác suất bốn lớp phone được cộng, smooth 3 mẫu, dùng hysteresis, persistence 2 mẫu và merge gap 2s.
Action-only luôn `NEEDS_REVIEW`; object+action là `PENDING`; code không phát kết luận vi phạm tự động.
B5 union late fusion tăng recall nhưng tăng FP. B6 chỉ giữ action event confidence ≥0.70 và duration ≥1s.

## Integration status

`predict_subject`, `probabilities_to_events` và `merge_evidence_events` là API inference/evidence có unit
test. Candidate **không được bật trong backend production** vì chưa đạt target, subject gZ-36 suy sụp và
chưa có independent holdout. Checkpoint hiện tại chỉ có ý nghĩa theo fold LOSO, không có một production
checkpoint đã được phê duyệt.
