# Audit khả năng tái lập, license và provenance

## Trạng thái nguồn ngoài tại 2026-10-15

| Nguồn | Paper | Code | Weights | Kết luận tái lập |
|---|---|---|---|---|
| DualScope | AAAI page + DOI xác minh | URL GitHub do paper nêu trả 404 | HF card được search index; API/artifact chưa xác minh | `NOT_REPRODUCED`; không tải hoặc phân phối |
| DriveCLIP | arXiv + IEEE DOI xác minh | GitHub công khai, MIT, frame-based | repo có thư mục checkpoint nhưng không chứng minh full VideoCLIP | `PARTIALLY_REPRODUCIBLE`; không chạy do compute/scope |
| Multi-Task Fusion | MDPI/PubMed xác minh | không tìm thấy official code | không tìm thấy | `NOT_REPRODUCIBLE_FROM_PUBLIC_ARTIFACTS` |
| YOLOv12 DMS | IIETA, article CC BY 4.0 | không tìm thấy official code | không tìm thấy | `PAPER_ONLY` |
| RST-Net seatbelt | MDPI open access | không tìm thấy official code | không tìm thấy | `PAPER_ONLY`; private dataset |
| Overhead YOLOv7 | ACS PDF, CC BY 4.0 | không tìm thấy official code | không tìm thấy | `PAPER_ONLY`; cần dữ liệu cùng camera |

License bài báo chỉ cho phép sử dụng nội dung bài báo theo điều khoản tương ứng; không tự động cấp
quyền cho dataset, code hoặc checkpoint. Vì vậy V5 không vendor code/weights từ sáu nguồn.

## DMD cục bộ

- Dataset role: `DEVELOPMENT_ONLY`; subjects: gB-9, gZ-36, gC-14.
- Input-set SHA-256: `e8590f2438ec9655314e8188eac72fd8666b029c3354cebab9f079556865070a`; config SHA-256: `6ed9fef82a7aeeb6fb3ff542d20b1ae1608b430a0d257f7a7fb219b8bc9c0e2a`.
- B3 embedding-set SHA-256: `4dffa9cdf76a3a876e3475d381ab110bd187fa31671d432a3edad702b9c8d14b`; backbone
  `torchvision ResNet-18 IMAGENET1K_V1` có weights SHA-256 `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`.
- Evaluation: `LEAVE_ONE_SUBJECT_OUT; threshold calibration uses training subjects only`.
- Holdout bị cấm: gZ-37, gE-28; accessed: `[]`.
- Mỗi fold train trên hai subject và test subject còn lại; threshold chỉ chọn trên hai training subjects.
- RGB cache có hash của video và annotation trong `input_provenance`; checkpoint có SHA-256 theo fold.

**Caveat license:** governance README cục bộ của DMD đang ghi quyền sử dụng/chấp thuận chưa hoàn tất dù
file dữ liệu có mặt. Kết quả này chỉ là nghiên cứu nội bộ, không phân phối dataset/cache/checkpoint và
không dùng làm sản phẩm cho đến khi quyền sử dụng được xác nhận.

## Mức tái lập Roadwatch

Command: `py -3.11 -m training.dmd.run_v5_experiments`. Cấu hình cố định tại
`models/model_config_v5_research.yaml`; seed 5305; checkpoints chứa architecture, class names,
train/heldout subjects, threshold, config, git HEAD và hash. CPU/PyTorch hiện tại được ghi trong JSON.
Môi trường kiểm tra là Windows, Python 3.11.9, PyTorch
2.13.0+cpu, CUDA=False; ổ D còn khoảng
10.13 GiB tại lúc render báo cáo.
Kết quả là `REPRODUCED_LOCALLY` cho chính code Roadwatch và `MEASURED_ON_ROADWATCH` cho metrics;
**không** phải reproduction của DualScope/DriveCLIP/Multi-Task Fusion.
