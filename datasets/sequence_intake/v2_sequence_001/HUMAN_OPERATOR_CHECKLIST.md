# Checklist người duyệt temporal intake

STATUS: **HUMAN_INPUT_REQUIRED**. Rights: **0/12**; lineage hoàn chỉnh: **0/12**;
canonical human-approved sequences: **0/12** tại thời điểm chuẩn bị checklist.
Media đã có SHA/full-decode PASS 12/12, 316.732.819 byte, 4.821 frame.
Đây là hướng dẫn thao tác, không phải bản human approval.

## 1. Mở đúng file để làm việc

Repo local: `D:\.idea\giangdoantotnghiep\projecy7`.
Các đường dẫn dưới đây tính từ repo root; các file trong `datasets/incoming/` chỉ có local,
không đi kèm khi clone GitHub. Bản review gốc được giữ nguyên; làm việc trên ba bản trả kết quả đã chuẩn bị:

| File người thật cần điền | Mục đích |
|---|---|
| `datasets/incoming/v2_sequence_001/human_review_return/rights_review.csv` | Quyết định quyền dùng theo từng clip, bằng chứng và người duyệt |
| `datasets/incoming/v2_sequence_001/human_review_return/physical_lineage_review.csv` | Source/camera/session/nhóm xe/người và bằng chứng |
| `datasets/incoming/v2_sequence_001/human_review_return/sequence_review.csv` | Xác nhận xem toàn video, dẫn chiếu notes/intervals và review evidence |

Các bản rights/lineage được sao nguyên từ template, gồm cả trường reviewer đang trống.
Bản sequence mới chỉ có metadata kỹ thuật; trường human để trống/PENDING.
Không sửa `temporal_content_review.csv` thành human truth: đó là proposal AI để tham khảo.
Không ghi đè evidence cũ; dùng file phiên bản mới nếu cần sửa review đã ký.

Video gốc: `datasets/incoming/v2_sequence_001/temporal_development/<CLIP_ID>.mp4`.
Thông tin tham chiếu: `datasets/incoming/v2_sequence_001/temporal_review_package/README.md`.

## 2. Ba phần bắt buộc cho MỖI clip

### Rights — người chịu trách nhiệm xác nhận

- [ ] Đối chiếu source page, license/terms, creator và asset/rendition với file đã tải.
- [ ] Ghi ngày/phương thức tải từ hồ sơ thực nếu có; chưa biết giữ UNKNOWN, không suy từ filesystem timestamp.
- [ ] Xác định project use được phép hay không; không suy quyền ML/dataset chỉ từ nút tải miễn phí.
- [ ] Người thật điền `PROJECT_USE_REVIEW=APPROVED` hoặc `REJECTED`; chưa quyết định giữ PENDING.
- [ ] Điền `REVIEWER_ID` thật, `REVIEWED_AT` có timezone, `EVIDENCE_PATH`, `EVIDENCE_SHA256` đúng file.

`RIGHTS_STATUS` là kết quả đối chiếu sau khi nhận evidence, không tự nâng VERIFIED chỉ vì chọn APPROVED.
Không chấp nhận terms hay yêu cầu quyền thay người dùng. Template được điền không tự thay bằng chứng quyền.

### Physical lineage — xác định từ bằng chứng

- [ ] Điền `SOURCE_ID`, `CAMERA_ID`, `CAPTURE_SESSION_ID`, `PHYSICAL_VEHICLE_GROUP_ID`, `PERSON_GROUP_IDS`.
- [ ] Bao phủ mọi người/xe; nếu có nhiều canonical vehicle thì bổ sung `VEHICLE_PHYSICAL_GROUPS` đầy đủ sau khi xác nhận cấu trúc.
- [ ] Gắn evidence path/SHA, người duyệt và thời điểm thực.
- [ ] Không suy ID từ Cxx, tên file, uploader, URL khác nhau hoặc phỏng đoán hình ảnh đơn thuần.
- [ ] Nếu không chứng minh được: ghi `PHYSICAL_LINEAGE_STATUS=NOT_PROVABLE`, field liên quan và lý do trong NOTES/evidence.

NOT_PROVABLE là kết quả review trung thực, **không đủ điều kiện** chuyển sang roster/calibration.
Không đổi thành nhóm ID mới để vượt kiểm tra độc lập. Lịch sử dùng và prediction exposure trong intake cũng cần xác nhận trung thực.

### Sequence — xem ORIGINAL FULL VIDEO

- [ ] Xem toàn video gốc, không chỉ contact sheet; dùng frame stepping ở các transition.
- [ ] Với mỗi occupant: ghi identity/role, inside/outside, vehicle và motorcycle applicability.
- [ ] PHONE: `PHONE_USE`, `PHONE_PRESENT_NOT_USED`, `MOUNTED_OR_STATIC_PHONE`, `NO_PHONE`, `UNKNOWN`.
- [ ] SEATBELT: `FASTENED`, `UNFASTENED`, `UNCERTAIN_OR_OCCLUDED`, `NOT_APPLICABLE`.
- [ ] Ghi start/end frame và thời gian chính xác, pre/post context, visibility, occlusion, conditions và full timeline.
- [ ] Xác nhận shot continuity/tốc độ capture; FPS container đều không chứng minh video chưa dựng hoặc slow motion.
- [ ] Không biến timestamp mẫu AI thành biên GT; không suy thiếu dây là UNFASTENED hoặc thấy phone là PHONE_USE.
- [ ] Lưu ghi chú/intervals thật vào `sequence_notes/`, dẫn đường dẫn trong `sequence_review.csv` và ký bằng evidence thật.

Event end frame bao gồm frame cuối; context interval dùng `[start_frame, end_frame)` theo contract hiện tại.
Dùng FPS thực từ metadata, không làm tròn 24000/1001 hoặc 30000/1001 theo tên file gốc.
Các trạng thái quan sát trên là worksheet; canonical annotation phải chuyển đúng schema của repo ở bước sau.

## 3. Checklist riêng từng clip, theo thứ tự

Mọi dòng đều cần hoàn tất ba phần trên; câu hỏi dưới đây chỉ là gợi ý từ proposal cũ, không thêm AI approval.

| Thứ tự | Clip | Những điểm cần người thật quyết định |
|---|---|---|
| PRIMARY 1 | C01 | Onset PHONE_USE; có kết thúc trong clip không; outside→inside; driver role; belt uncertain nếu không thấy rõ |
| PRIMARY 2 | C07 | Belt ban đầu; actual latch/buckle; frame fastening; trạng thái FASTENED cuối có được chứng minh không |
| PRIMARY 3 | C10 | Actual latch/buckle; driver hay passenger; vùng dây lẫn áo tối phải giữ uncertainty khi cần |
| PRIMARY 4 | C11 | Tháo khóa dưới một giây; FASTENED→UNFASTENED; cùng occupant qua đoạn exit; lúc chuyển NOT_APPLICABLE |
| SECONDARY 1 | C02 | Tương tác tay/màn hình; onset/offset có bị cắt; role, belt khuất và điều kiện thời tiết thực |
| SECONDARY 2 | C03 | Chuyển từ gọi điện sang nhìn màn hình có còn PHONE_USE; vùng cuối có non-use thật không |
| SECONDARY 3 | C04 | Tốc độ capture/slow motion; tương tác màn hình→đưa lên tai; belt/lap bị che |
| SECONDARY 4 | C05 | Phone có mounted/static thật không; xem toàn clip để tìm touches; không gán NO_PHONE; xác nhận occupant khi góc thấy hạn chế |
| SECONDARY 5 | C08 | Role của ghế; outside→inside; thao tác lap/buckle và belt state, không dựa tiêu đề trang |
| SECONDARY 6 | C09 | Mọi occupant, role theo cabin; từng belt transition; camera move hay cut; phần cuối có bị cắt hành vi không |
| SECONDARY 7 | C12 | Có phone interaction ngắn không; phân biệt vật trước camera; role/cabin/visibility và điều kiện sáng thực |
| SECONDARY 8 | C16 | Thiết bị navigation mounted/static; mọi tương tác trong full clip; không suy belt ngoài khung hình hoặc NO_PHONE |

Secondary chỉ đưa vào annotation sâu nếu cần thêm diversity. Không tự reject/approve clip vì thứ tự này.

## 4. File evidence người thật phải cung cấp

Các thư mục local đã chuẩn bị, còn trống:

- `datasets/incoming/v2_sequence_001/human_review_return/evidence/rights/`: nguồn/license áp dụng, hồ sơ tải nếu có, quyết định project use có người chịu trách nhiệm.
- `datasets/incoming/v2_sequence_001/human_review_return/evidence/lineage/`: hồ sơ source/camera/session/người/xe hoặc lý do NOT_PROVABLE, có người xác nhận.
- `datasets/incoming/v2_sequence_001/human_review_return/evidence/sequence/`: bằng chứng xem video và review, liên kết video SHA cùng file ghi chú/intervals đúng phiên bản.
- `datasets/incoming/v2_sequence_001/human_review_return/sequence_notes/`: bảng occupant/role/context và interval do người thật ghi; chưa phải roster/annotation canonical đã duyệt.

Chọn tên file đúng thực tế, ghi đường dẫn tương đối repo root trong worksheet và SHA256 tương ứng.
Evidence phải tồn tại, không rỗng; URL đơn thuần không phải evidence file.
Không tự đặt ngày capture từ PTS tương đối, không dùng filesystem timestamp làm ngày tải.

Hồ sơ review người thật hoàn tất phải chứa:

| Field | Điều kiện nhận |
|---|---|
| `reviewer_type` | HUMAN do người thật xác nhận |
| `reviewer_id` | Danh tính người duyệt thật, không placeholder |
| `reviewed_at` | Timestamp thật, có timezone |
| `review_evidence.path` | Evidence thực, tồn tại và không rỗng |
| `review_evidence.sha256` | Khớp bytes trên disk |
| `adjudication_status` | FINAL sau review thực |

Trong `sequence_review.csv`, các cột viết hoa `REVIEWER_TYPE`, `REVIEWER_ID`, `REVIEWED_AT`,
`REVIEW_EVIDENCE_PATH`, `REVIEW_EVIDENCE_SHA256`, `ADJUDICATION_STATUS` ghi thông tin tương ứng.
Người thật mới điền các giá trị này; một chữ HUMAN/FINAL hoặc hash đúng không chứng minh review thực sự đã xảy ra.
Proposal gốc vẫn là AI_REVIEWED_PROPOSAL, adjudication PENDING; không sửa thành HUMAN.

## 5. Chỉ sau khi người thật trả kết quả

1. Đọc bản trả kết quả, giữ nguyên evidence gốc; kiểm tra mọi đường dẫn và tính không rỗng.
2. Tính lại SHA256 của từng evidence, từ chối mismatch; không sửa hash để hợp thức hóa nội dung.
3. Đối chiếu rights quyết định thực với evidence; kiểm tra đầy đủ lineage và mọi NOT_PROVABLE.
4. Kiểm tra reviewer không placeholder, timestamp có timezone, chữ ký/xác nhận review thực và FINAL.
5. Kiểm tra từng reviewed video SHA vẫn khớp media hiện tại, và review bao phủ đúng phiên bản file.
6. Chỉ clip rights chấp nhận được + human lineage hoàn chỉnh mới được chuẩn bị independent identity roster.
7. Roster mới phải bắt đầu UNREVIEWED/PENDING, không lấy model predictions để dựng identity.
8. Sau khi roster được người thật review và frozen hợp lệ theo protocol mới tạo skeleton canonical sequence.
9. Người thật duyệt canonical sequence lần nữa, hoàn thiện mọi field và evidence trước validation.

Chưa có human-completed record hoặc canonical sequence, nên **không có lệnh machine validation để chạy lúc này**.
Lệnh validation sau này dùng `training.validate_event_sequence_annotations` với đúng sequence JSON đã hoàn thiện,
theo `docs/V2_EXTERNAL_TEST_PROTOCOL.md` và schema `datasets/schemas/v2_event_sequence_annotation.schema.json`.
Các bước trên là checklist cho lần nhận human input, không phải validator mới đã được thực thi.

## 6. Đánh giá độ đủ sau validation, chưa calibration

Chỉ đếm trên sequence human-approved đã validation hợp lệ:
sequence count; PHONE_USE; mounted/static negatives; fastening/unfastening transitions;
FASTENED/UNFASTENED/uncertain intervals; camera/view diversity; lighting/conditions;
tổng phút video liên tục. Nếu nhiều cabin/occupant cùng video, không cộng trùng thời lượng video.
Không coi bốn primary clip là đủ GT chỉ vì đã được duyệt. Thiếu mục nào phải nêu rõ loại sequence cần thêm.

## 7. Holdout và bất biến

Untouched holdout hiện 0 video. C01–C05, C07–C12, C16 vĩnh viễn thuộc development và loại khỏi holdout,
cùng copies, transcodes, cuts, reuploads và lineage nguồn/camera/session/người/xe giao nhau theo policy.
Không tìm bản đổi tên của các clip này để làm holdout.
Không tải thêm temporal nếu dữ liệu được review chưa chứng minh thiếu coverage;
không đề xuất package >1 GB, ưu tiên individual video ≤100 MB, giữ chỗ cho holdout mới.

DO NOT RUN: model inference, calibration, identity freeze khi thiếu human prerequisites,
external/truth freeze, final event evaluation, canonical frozen model test.

```ini
TEMPORAL_CALIBRATION = BLOCKED
TEMPORAL_POLICY_LOCK = PENDING_SEQUENCE_GROUND_TRUTH
EVENT_EVALUATION = PENDING_NEW_UNTOUCHED_HOLDOUT
HUMAN_VERIFIED = false
PRODUCTION_READY = false
FROZEN_TEST_RUN_COUNT = 1
```
