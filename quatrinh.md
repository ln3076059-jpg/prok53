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
- **Môi trường:** RTX 5060 Ti 16GB (All Pretrain Pending Train).

### 3.1. Phone Detector (Nhận diện sử dụng điện thoại)
- **mAP50:** 94.48%
- **mAP50-95:** 70.03%
- **Precision:** 97.13%
- **Recall:** 82.73%
- **Đánh giá:** Rất xuất sắc, khả năng nhận diện chính xác cao và tỷ lệ báo động giả cực thấp.

### 3.2. Seatbelt Detector (Phát hiện dây an toàn - Bounding Box)
- **mAP50:** 94.73%
- **mAP50-95:** 53.70%
- **Precision:** 92.57%
- **Recall:** 87.85%
- **Đánh giá:** Khả năng định vị vùng chứa dây an toàn (occupant upper body) rất chuẩn xác.

### 3.3. Seatbelt Classifier (Phân loại trạng thái dây an toàn)
- **Accuracy Top-1:** 80.97%
- **Accuracy Top-5:** 100.00%
- **Đánh giá:** Hiệu suất tốt ở mức ~81% để phân loại xem người lái có thắt dây hay không.

## 4. Trọng số Mô hình (Weights)
- Các tệp trọng số tốt nhất (best weights) hiện đang được lưu tại đường dẫn con `weights/best.pt` của thư mục `runs/MULTIMODEL_V2_PRETRAIN/...`.
- **Bước tiếp theo dự kiến:** Evaluate (đánh giá trên tập test) hoặc Lock model (đóng băng mô hình) để chuẩn bị tích hợp.
