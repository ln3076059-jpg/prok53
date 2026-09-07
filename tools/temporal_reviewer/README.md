# Giao diện duyệt video temporal

Chạy từ thư mục gốc dự án, với môi trường Python đã cài requirements hiện có:

```powershell
py -m tools.temporal_reviewer.app
```

Mở **http://127.0.0.1:8766**. Công cụ chỉ lắng nghe trên máy local, không chạy model,
calibration, freeze hoặc evaluation. Không cần chạy backend chính hay MySQL.

## Cách duyệt

1. Chọn C01, sau đó C07, C10, C11. Xem toàn bộ video gốc bằng trình phát.
2. Chọn từng mục để đọc đề xuất, vai trò, visibility, interval, ghi chú và SHA.
   Bấm **Đến thời điểm bắt đầu** để đối chiếu. Bước khung hình dựa trên FPS danh định,
   không khẳng định trình duyệt seek chính xác từng PTS.
3. Chọn **Phê duyệt**, **Sửa và phê duyệt**, **Từ chối** hoặc **Chưa chắc chắn**.
   Bản sửa nhập bằng JSON đầy đủ, được kiểm tra theo hợp đồng Review1 và toàn timeline.
4. Có hai nút hàng loạt: các mục đang chờ của clip hiện tại, hoặc mọi clip.
   Hộp xác nhận liệt kê chính xác từng mục. Không ghi đè bất kỳ dòng nào đã có trường HUMAN.
5. Người thật nhập danh tính, thời điểm có múi giờ và ghi chú. **Điền giờ hiện tại**
   chỉ chạy khi người dùng bấm. Tự xác nhận đã xem video và nội dung rồi bấm **Lưu quyết định của tôi**.

## Lưu ở đâu

- Queue: `datasets/incoming/v2_sequence_001/review1/HUMAN_APPROVAL_QUEUE.csv`.
- Mỗi lần người thật gửi quyết định: `review1/human_decisions/<batch_id>/` dưới cùng
  thư mục local; gồm `queue_before.csv`, `queue_after.csv`, `receipt_*.json`.
- Chỉ sửa 7 cột HUMAN. Nguồn Review1 JSON, nhãn và SHA proposal được giữ nguyên.
- Biên nhận gắn item, video/record/payload SHA, người duyệt, thời điểm, quyết định,
  ghi chú và bản sửa nếu có. SHA biên nhận tính từ bytes đã ghi.
- Nút **Tải bảng kết quả CSV** tải queue hiện tại. Tất cả dữ liệu duyệt vẫn nằm trong
  `datasets/incoming/` đã gitignore; không đưa video hoặc biên nhận lên GitHub.

Đọc trang không tạo quyết định. Khi lưu, kiểm tra lại video và đề xuất trước khi ghi
bằng thay thế queue nguyên tử. Nếu queue thay đổi hoặc có dữ liệu HUMAN từ trước,
yêu cầu bị từ chối. Chỉ chạy một instance công cụ, không sửa CSV bên ngoài trong lúc lưu.
Đây là công cụ local một người vận hành, không có đăng nhập hay chứng thực danh tính:
biên nhận chứng minh thao tác gửi có khai danh tính, không tự chứng minh người đó là ai.

## Các gate giữ riêng

Phê duyệt đề xuất rights `NEEDS_HUMAN_DECISION` không xác nhận quyền sử dụng.
Phê duyệt lineage `NOT_PROVABLE` vẫn giữ chưa chứng minh được lineage. Cần cung cấp
bằng chứng thực theo quy trình human return. Không dùng các nút này để tạo ID giả.

Biên nhận giao diện là **quyết định cấp item**, không phải final receipt cho toàn bộ
payload. Không tự tạo `HUMAN_APPROVED` record hoặc canonical ground truth. Các mục
từ chối/chưa chắc chắn còn phải được giải quyết. Human return và evidence cần được
kiểm tra tiếp trước khi xét identity roster/canonical conversion.

Không thay đổi readiness, model card, policy, reports hay governance:

```ini
TEMPORAL_POLICY_LOCK = PENDING_SEQUENCE_GROUND_TRUTH
EVENT_EVALUATION = PENDING_NEW_UNTOUCHED_HOLDOUT
HUMAN_VERIFIED = false
PRODUCTION_READY = false
FROZEN_TEST_RUN_COUNT = 1
```

Kiểm tra code giao diện với fixture tạm, không ghi vào queue thật:

```powershell
py -m pytest tests/test_temporal_reviewer.py tests/test_review1_contract.py -q
```
