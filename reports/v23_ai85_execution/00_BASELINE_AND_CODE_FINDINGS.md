# ROADWATCH V23 — AUDIT NGẮN, BẰNG CHỨNG VÀ KẾT LUẬN HÀNH ĐỘNG (P0)

**Local HEAD:** `98502a4ab740d3ede3df18a6e217fb1861c009e9`  
**Evaluation Scope:** FULL_COHORT (`gC-14`, `gZ-36`), 15.270497 phút video, 20 Ground Truth events.

---

## 1. Call Graph Thực Tế Trong Pipeline

Chuỗi thực thi đo kiểm từ runner nghiên cứu và backend:
```
Video Source (1280x720 @ 29.76 fps)
  → Frame Sampling (Stride 15 frames ~ 0.504s)
    → Cabin/Driver ROI Localization ([0.0, 0.10, 0.65, 0.95] → [0, 72, 832, 684])
      → Specialist YOLO11 Phone + Auxiliary YOLO11n (COCO 67) → IoU 0.50 Merge
      → Pose Estimator (YOLO11n-pose trên Driver Crop)
        → Occupant Seat Calibration & Identity Tracking (Driver vs Passenger vs Unknown)
          → Driver-Phone Geometric Association (Containment >= 0.35, Proximity >= 0.35)
            → Action Recognition Adapter (NONE / RULE_ONLY / V6_HIERARCHICAL / HYBRID)
              → Multi-evidence Fusion & Observation Generation
                → Temporal Event Engine (Candidate >= 0.45, Activation >= 0.55, Occlusion Bridge 3.5s)
                  → Active Boundary Lifecycle Tracker (bám sát quan sát dương thực tế, không tạo đuôi EOF)
                    → Cooldown Deduplication (3.0s)
                      → Frozen Matcher (IoU 0.3, Onset 2.5s) → Reconciled Metrics
```

---

## 2. Bảng Phân Loại Điểm Nghẽn Mã Nguồn (Root Cause Findings)

| ID | Vấn đề & Vị trí | Phân loại | Bằng chứng mã nguồn | Tác động Metric | Sửa chữa V23 |
|---|---|---|---|---|---|
| **DEF-01** | `training/dmd/run_verified_v22.py:138` — Association gọi rỗng | **CONFIRMED_CODE_DEFECT** | `DriverPhoneAssociation.associate(Box(*local), (), ...)` truyền tuple occupants rỗng, ép `occupant_role="UNKNOWN"`, `confidence=0.0`. | B1/B2 từ chối 100% sự kiện; TP/FP/FN = 0/0/20, Recall = 0.0%. | Bổ sung calibration manifest (`camera_seat_calibration.json`), tạo `OccupantTrack` thật từ pose và seat ROI. |
| **DEF-02** | `training/dmd/run_verified_v22.py:370` — B3/B4 không khởi tạo adapter trong replay | **CONFIRMED_CODE_DEFECT** | Preflight kiểm tra adapter nhưng vòng lặp replay chỉ tạo `RULE_ONLY` cho B2; B3/B4 nhận `adapter = None`. | Deep model forward không bao giờ được kích hoạt trong pipeline execution; B3/B4 bị bỏ dở. | Khởi tạo adapter đúng mode (`V6_HIERARCHICAL`, `HYBRID_RULE_V6`) trong replay, nạp embedding và forward model thật. |
| **DEF-03** | `training/dmd/action_adapter.py:159` — Chặn CENTERED model mà không có contract độ trễ | **CONFIRMED_CODE_DEFECT** / **CONTRACT_LIMITATION** | `_load_verified_model()` ném `CheckpointSafetyError` khi `label_position == "CENTER"`. | Chặn hoàn toàn các checkpoint V6/V7 dù model và weights có thật trên đĩa. | Triển khai contract `causal_mode="DECLARED_LATENCY_CENTERED"`, khai báo độ trễ 2.0s (4 bước), bảo toàn tính trung thực thời gian. |
| **DEF-04** | `models/candidate/v6_action_ablation_001/` — Thiếu `.manifest.json` sidecars | **CONFIRMED_CODE_DEFECT** | Checkpoint và training log có đủ hash trên đĩa nhưng thiếu file manifest sidecar định dạng đúng quy chuẩn V22. | Gây lỗi `CHECKPOINT_PROVENANCE_NOT_PROVABLE` ở preflight. | Tạo manifest sidecar với SHA checkpoint, SHA backbone (`f37072fd...`), và liên kết training log. |
| **DEF-05** | `run_verified_v22.py:388-398` — Event lifecycle chỉ báo cửa sổ kích hoạt ban đầu | **CONFIRMED_CODE_DEFECT** | Chỉ lưu `[start, end]` tại thời điểm phát sự kiện (~1.2s), không cập nhật thời gian kết thúc khi quan sát dương tiếp diễn. | Gây phân mảnh sự kiện, IoU cực thấp so với GT dài (10–50s), duplicate matches bị tính FP (FP=16). | Tích hợp cơ chế `ActiveBoundaryReporter` bám sát quan sát dương thực tế đến khi release, không tạo đuôi EOF giả. |
| **DEF-06** | Seatbelt classifier `uncertain_or_occluded` recall = 4.0% | **MODEL_LIMITATION** / **DATA_GOVERNANCE** | Confusion matrix 641 mẫu validation: 24/25 mẫu uncertain bị đoán nhầm vào fastened/unfastened. | Macro-F1 seatbelt chỉ đạt 55.6% - 57.3%, accuracy 80.9%. Bị chặn fitting bởi `RIGHTS_ACCEPTED = false`. | Lập sổ ledger minh bạch, giữ nguyên bằng chứng kiểm chứng, không gộp lớp hay che giấu độ bất định. |
| **DEF-07** | Multiview fusion trên toàn cohort | **MISSING_EVIDENCE** | `gZ-36` có BODY/FACE/HANDS; `gC-14` chỉ có BODY. Thiếu nhãn đồng bộ nội dung và timestamp offset. | Chưa đủ cơ sở khoa học để fuse đa góc nhìn trên toàn cohort. | Giữ BODY camera làm benchmark chuẩn toàn cohort; ghi nhận multiview feasibility riêng. |

---

## 3. Kết Luận Hành Động Cho V23

1. Không hạ tiêu chuẩn, không đổi giao thức đánh giá (Frozen policy: IoU 0.3, onset 2.5s).
2. Sửa toàn bộ các lỗi mã nguồn (DEF-01 → DEF-05).
3. Chạy ma trận thực nghiệm chuẩn R0 đến R7 trên cùng dữ liệu và điều kiện.
4. Ghi nhận trung thực các giới hạn dữ liệu và quản trị theo quy định khoa học.
