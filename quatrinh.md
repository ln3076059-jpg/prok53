# Nhật ký Quá trình Cập nhật & Huấn luyện (V2 Production Pipeline)

## 1. Cập nhật mã nguồn lên GitHub
- **Ngày thực hiện:** 04/09/2026
- **Chi tiết:** 
  - Cập nhật và điều chỉnh `.gitignore` để bỏ qua các file trọng số mô hình lớn (`.pt`) và thư mục `epoch_snapshots/` nhằm tránh lỗi giới hạn 100MB của GitHub.
  - Đồng bộ và push thành công mã nguồn hiện tại cùng file `MULTIMODEL_V2_PORTABLE.sha256` lên branch `main`.

## 2. Đóng gói Dữ liệu Huấn luyện (Portable Bundle)
- **Tiến trình:** Chạy script `py -m training.build_v2_portable_bundle` để sinh tệp `MULTIMODEL_V2_PORTABLE.zip` và file `V2_PORTABLE_EXPECTED_HASHES.json`.
- **Ghi chú:** Do kích thước lớn (~500MB), file zip này không được đẩy lên GitHub mà được cấu hình lưu trữ độc lập.

## 3. Hoàn tất Huấn luyện Mô hình (Pre-training)
- **Trạng thái:** `COMPLETE` cho cả 3 thành phần.
- **Môi trường:** NVIDIA GeForce RTX 3090, VRAM ~24GB, Windows 10, PyTorch 2.8.0+cu128.

### 3.1. Phone Detector (Nhận diện sử dụng điện thoại)
- **mAP50:** 94.48%
- **mAP50-95:** 70.03%
- **Precision:** 97.13%
- **Recall:** 82.73%
- **F1-Score:** 89.34%
- **Đánh giá:** Rất xuất sắc, khả năng nhận diện chính xác cao và tỷ lệ báo động giả cực thấp.

### 3.2. Seatbelt Detector (Phát hiện dây an toàn - Bounding Box)
- **mAP50:** 94.73%
- **mAP50-95:** 53.70%
- **Precision:** 92.57%
- **Recall:** 87.85%
- **F1-Score:** 90.13%
- **Đánh giá:** Khả năng định vị vùng chứa dây an toàn (occupant upper body) rất chuẩn xác.

### 3.3. Seatbelt Classifier (Phân loại trạng thái dây an toàn)
- **Accuracy Top-1:** 80.97%
- **Đánh giá:** Hiệu suất Top-1 đạt ~81% (lưu ý không dùng chỉ số Top-5 vì mô hình chỉ có 3 classes). Sẽ bổ sung macro-F1 và per-class precision-recall trong báo cáo chi tiết sau.

## 4. Trọng số Mô hình (Weights)
- Các tệp trọng số tốt nhất (best weights) hiện đang được lưu tại đường dẫn con `weights/best.pt` của thư mục `runs/MULTIMODEL_V2_PRETRAIN/...`.
- **Bước tiếp theo dự kiến:** 
  1. Lock model / config (đóng băng mô hình).
  2. Threshold / calibration bằng tập validation.
  3. Frozen test (chỉ dùng test split sau khi lock).
  4. Behavior / event evaluation.
  5. Camera / temporal calibration.
  6. Xem xét activation.
