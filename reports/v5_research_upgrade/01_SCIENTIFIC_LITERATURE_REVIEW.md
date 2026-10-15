# Tổng quan tài liệu khoa học cho Roadwatch V5

**Ngày kiểm tra nguồn:** 2026-10-15. **Quy ước:** `REPORTED_IN_PAPER` là số do tác giả công bố;
`REPRODUCED_LOCALLY` chỉ dùng khi đã chạy lại đúng phương pháp; `MEASURED_ON_ROADWATCH` là số đo từ
artifact của dự án. Ba nhóm không được so sánh như cùng một metric.

## 1. DualScope (AAAI 2026)

Qiu, Li, Zhang, Hu và Ma, “DualScope: Capturing Critical Spatial and Temporal Cues for Distracted
Driving Activity Recognition,” AAAI 40(10), 8602–8611, DOI
[10.1609/aaai.v40i10.37812](https://doi.org/10.1609/aaai.v40i10.37812).

DualScope tách hai loại tín hiệu: (i) biểu diễn không gian tập trung vào vùng hành vi, dùng tri thức vị
trí từ SAM và crop vùng quan trọng; (ii) mô hình thời gian fine-to-coarse gồm motion encoder, dynamic
difference và temporal pyramid Mamba. `REPORTED_IN_PAPER`: DMD Top-1 khoảng 98.63%; đây là video
classification, không phải event F1. Điểm có thể tái hiện trên CPU là crop tài xế, sai phân khung hình,
temporal pooling và sampling động nhẹ. Bản đầy đủ cần GPU đáng kể. URL code được bài báo nêu hiện trả
404 khi kiểm tra; model card Hugging Face tồn tại trong chỉ mục nhưng checkpoint chưa được tải/xác minh.

## 2. DriveCLIP (IEEE T-ITS 2024)

Hasan et al., “Vision-Language Models Can Identify Distracted Driver Behavior From Naturalistic
Videos,” DOI [10.1109/TITS.2024.3381175](https://doi.org/10.1109/TITS.2024.3381175), preprint
[arXiv:2306.10159](https://arxiv.org/abs/2306.10159).

CLIP cung cấp visual embedding và semantics của nhãn; tác giả đánh giá zero-shot, fine-tuned frame
model và video model. `REPORTED_IN_PAPER`: VideoCLIP DMD Top-1 khoảng 98.44%. Repository
[DriveCLIP](https://github.com/zahid-isu/DriveCLIP) có license MIT và tự mô tả là **frame-based**;
output là nhãn/xác suất từng frame và README trỏ sang repository khác cho VideoCLIP. Vì vậy không thể
coi code công khai này là bản tái tạo đầy đủ con số video. Ý tưởng được áp dụng trong V5 là nhánh action
độc lập và xác suất theo clip, không sao chép checkpoint.

## 3. Multi-Task Fusion (Sensors 2025)

Vijeikis, Dada, Abayomi-Alli và Raudonis, “Enhancing Driver Monitoring Systems Based on Novel
Multi-Task Fusion Algorithm,” Sensors 25(21), 6799, DOI
[10.3390/s25216799](https://doi.org/10.3390/s25216799).

Mô hình dùng video BODY/FACE/HANDS, 3D ResNet-18 + BiLSTM cho driver action, gaze và hand-wheel,
rồi tổng hợp danger/attentiveness. `REPORTED_IN_PAPER`: Driver Actions F1 khoảng 93.20%, fused
attentiveness F1 khoảng 98.28% trong cấu hình riêng, và overall multi-task accuracy 93.74% ở bảng khác.
Các số này có task/denominator khác nhau. Không tìm thấy code hoặc weights chính thức. Phân chia subject
cần được kiểm chứng kỹ trước khi coi là bằng chứng tổng quát hóa.

## 4. YOLOv12 DMS (IJSS&E 2025)

Chaman, El Maliki, El Yanboiy, Dahou và Hadjoudja, “YOLOv12-Based Driver Monitoring System:
Real-Time Applications in ADAS,” 15(9), 1921–1932, DOI
[10.18280/ijsse.150915](https://doi.org/10.18280/ijsse.150915).

`REPORTED_IN_PAPER` trên 17,000 ảnh, split ảnh stratified 70/20/10: precision 95.0%, recall 92.5%,
F1 93.7%, mAP50 96.0%, mAP50-95 75%. Đây là object/image evaluation; subject-disjoint và near-frame
isolation chưa được chứng minh. Không có benchmark Roadwatch nên V5 không thay YOLO11s.

## 5. Lightweight seatbelt + attention (Applied Sciences 2024)

Qiu, Rao và Zhao, “Seatbelt Detection Algorithm Improved with Lightweight Approach and Attention
Mechanism,” Applied Sciences 14(8), 3346, DOI
[10.3390/app14083346](https://doi.org/10.3390/app14083346).

Pipeline gồm GM-YOLOv7 định vị và RST-Net phân loại, dùng Ghost/ELAN, Mish, channel pruning và
Triplet Attention. Trang bài báo ghi `REPORTED_IN_PAPER` **accuracy 98.25%** cho RST-Net và giảm
parameters 48%; yêu cầu đầu vào gọi 98.25% là F1 không khớp nhãn metric trên abstract nên báo cáo này
không đổi accuracy thành F1. Dataset tự xây dựng, chưa có code/weights chính thức được xác minh.

## 6. Overhead YOLOv7 seatbelt (Applied Computer Science 2025)

Widodo, Adi, Priyono và Setiawan, “Real-time Detection of Seat Belt Usage in Overhead Traffic
Surveillance Using YOLOv7,” 21(4), 1–12, DOI
[10.35784/acs_7594](https://doi.org/10.35784/acs_7594).

Hệ thống phát hiện car → windshield/passenger → seat belt từ camera trên cao, có dữ liệu ngày/đêm.
`REPORTED_IN_PAPER`: day mAP50 97.46%, best F1 95.37%; night mAP50 91.64%, F1 88.60%.
Phương pháp hợp ngữ cảnh camera ngoài đường hơn classifier cabin, nhưng cần kiểm tra độ phân giải vùng
kính, phản chiếu, góc nhìn và rò rỉ giữa frame cùng video trước khi áp dụng.

## Kết luận chuyển giao

V5 chỉ chuyển giao phần có chi phí/pháp lý phù hợp: driver crop, clip liên tiếp, spatial encoder nhỏ,
temporal mean/max/delta pooling, action probability độc lập, hysteresis và late fusion có trạng thái
`NEEDS_REVIEW`. Không có paper nào được coi là đã tái tạo cục bộ; không có số paper nào được dùng làm
số Roadwatch.
