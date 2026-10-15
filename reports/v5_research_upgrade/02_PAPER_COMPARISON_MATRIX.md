# Ma trận so sánh công trình

| Công trình | Dataset / task | Kiến trúc & train | Protocol / metric công bố | Compute | Code / weights / license | Leakage & giới hạn | Áp dụng / effort |
|---|---|---|---|---|---|---|---|
| DualScope, 2026 | 7 DDAR datasets; video action classification | behavior-centric SAM distillation + crops; motion encoder + difference + pyramid Mamba | DMD Top-1 ~98.63% (`REPORTED_IN_PAPER`) | GPU cao; distillation phức tạp | Code URL 404; HF card được lập chỉ mục nhưng artifact chưa xác minh; license code chưa xác minh | Split phải theo subject; Top-1 không phải event F1 | Dùng crop/motion/pyramid nhẹ: trung bình; full reproduction: rất cao |
| DriveCLIP, 2024 | DMD/StateFarm/SynDD/SAM-DD; frame & video classification | CLIP visual encoder, zero-shot/fine-tune, temporal video head | VideoCLIP DMD Top-1 ~98.44% (`REPORTED_IN_PAPER`) | ViT-L/14/GPU cao | Public repo MIT nhưng README nói frame-based; full VideoCLIP checkpoint không được repo này bảo đảm | Phải dùng driver-profile split; frame sampling dễ tạo near-duplicate | Action semantics/teacher embeddings: trung bình-cao |
| Multi-Task Fusion, 2025 | DMD + robustness datasets; action/gaze/hands/attention | 3D ResNet-18 + BiLSTM, BODY/FACE/HANDS, risk aggregation | action F1 ~93.20%; fused F1 ~98.28% (`REPORTED_IN_PAPER`, khác cấu hình) | 3 video streams + 3D CNN; GPU cao | Paper CC BY; code/weights chưa tìm thấy | Subject split cần audit; multi-task F1 không phải event F1 | Multi-view fusion: cao |
| YOLOv12 DMS, 2025 | 17k images; object/behavior detection | YOLOv12, multi-scale/attention/augmentation | P 95.0, R 92.5, F1 93.7, mAP50 96.0 (`REPORTED_IN_PAPER`) | Real-time claim trên cấu hình paper | Article CC BY 4.0; official code/weights chưa xác minh | 70/20/10 image split; subject/sequence leakage chưa loại trừ | Chỉ thử nếu benchmark trực tiếp thắng YOLO11s: cao |
| GM-YOLOv7 + RST-Net, 2024 | Self-built seatbelt images; ROI classification | Ghost/ELAN/Mish + pruning + Triplet Attention | RST-Net accuracy 98.25% (`REPORTED_IN_PAPER`) | Lightweight relative to baselines | Article CC BY; code/weights chưa xác minh | Private dataset; cross-camera generalization chưa biết | Torso ROI + attention: trung bình |
| Overhead YOLOv7, 2025 | Day/night overhead traffic; detection | car/windshield/passenger/seatbelt detector | Day mAP50 97.46, F1 95.37; night 91.64/88.60 (`REPORTED_IN_PAPER`) | Real-time YOLOv7 | Article CC BY 4.0; official code/weights chưa xác minh | Adjacent video frames may leak unless group-split | Phù hợp camera ngoài đường; cần dataset cùng góc: cao |

## Quyết định V5

TinyCNN temporal được chọn làm baseline CPU; vòng 2 tiếp tục với frozen ResNet-18 ImageNet + BiGRU
để kiểm tra giá trị của pretrained spatial representation mà vẫn giữ LOSO subject-disjoint. Không chọn
YOLOv12 vì chưa có đối chứng cùng dữ liệu; B4/B7 không chạy vì artifact/compute chưa đủ chắc chắn.
