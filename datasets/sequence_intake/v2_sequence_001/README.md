# Tiếp nhận sequence thật — v2_sequence_001

Trạng thái: **CHỜ VIDEO THẬT VÀ HUMAN REVIEW**. Đây là gói tiếp nhận dữ liệu,
chưa phải ground truth, manifest được freeze hay bằng chứng đánh giá model.
Kết quả kiểm kê nằm trong `readiness.json`; `intake.csv` hiện chỉ có tiêu đề cột.

## 1. Đưa video vào hai nhóm riêng

| Nhóm đề xuất trong `proposed_role` | Thư mục local đã chuẩn bị | Mục đích |
| --- | --- | --- |
| `TEMPORAL_DEVELOPMENT` | `datasets/incoming/v2_sequence_001/temporal_development/` | Annotation sequence để chọn/calibration temporal policy trên dữ liệu development. |
| `NEW_UNTOUCHED_HOLDOUT` | `datasets/incoming/v2_sequence_001/new_untouched_holdout/` | Dự trữ cho final event evaluation sau khi đã khóa model/policy. |

Đây là nhãn tiếp nhận, không phải giá trị `dataset_role` của frozen manifest.
Không chuyển video đã dùng để chọn model, threshold hoặc temporal policy sang holdout.
Giữ file gốc, không dựng video bằng cách nối ảnh rời. Các đoạn cắt từ cùng video/capture
session phải thuộc cùng nhóm. Duyệt hình ảnh để gán nhãn holdout là cần thiết;
không hiển thị dự đoán model cho người gán nhãn hoặc dùng kết quả holdout để chọn mẫu.

Không đưa video, danh tính người tham gia hay biểu mẫu đồng ý lên GitHub.
Thư mục `datasets/incoming/` đã được Git bỏ qua. Bảng CSV trong gói này là mẫu trống;
copy nó vào thư mục incoming trước khi điền thông tin thật.

## 2. Điền một dòng cho mỗi video trong bản sao local của intake.csv

- `video_path`: đường dẫn file thật. SHA-256, FPS, số frame và thời lượng sẽ được đo
  từ file khi tiếp nhận; không điền thông số ước lượng.
- `source_id`, `source_url`, `rights_evidence_path`: nguồn và bằng chứng quyền sử dụng.
  Không tự đánh dấu đã chấp nhận license của nguồn bên ngoài.
- `camera_id`, `capture_session_id`, `physical_vehicle_group_id`, `person_group_ids`:
  ID nhóm ổn định giữa các video để kiểm tra trùng lặp. Dùng mã giả danh;
  nhiều ID trong một ô ngăn cách bằng dấu `;`.
- `video_id`: ID video duy nhất. ID nhóm xe/người vật lý ở bảng intake khác với
  canonical `vehicle_id`/`occupant_id` trong annotation, vốn có namespace theo video.
- `prior_usage`: liệt kê việc dùng trước đây (train/validation/calibration/test hoặc
  chưa dùng có bằng chứng). `model_predictions_seen`: ghi `true`, `false` hoặc `UNKNOWN`
  theo thông tin thực; `UNKNOWN` chưa đủ để xác nhận untouched.
- `independence_evidence_path`: hồ sơ đối chiếu nguồn/camera/video/xe/người với toàn bộ
  development và canonical test cũ. Khác SHA file không đủ chứng minh độc lập.
- `conditions`: điều kiện quan sát thực, ngăn cách bằng `;`. `assigned_reviewer` chỉ
  là người dự kiến duyệt; `human_review_status` giữ `PENDING` trước khi có review thật.

## 3. Mức tối thiểu cho holdout theo policy đang có

`datasets/v2_external_test_policy.yaml` yêu cầu tối thiểu **2 nguồn, 3 camera,
12 video, 8 xe, 8 người**, cùng đầy đủ 14 điều kiện trong `holdout_coverage.csv`.
Đây là mức tối thiểu của policy hiện tại, không phải cam kết đủ sức mạnh thống kê.
Mỗi điều kiện cần liên kết tới video và bằng chứng người duyệt xác nhận.
Không tạo source/camera/person ID mới chỉ để đạt số lượng.

Nếu không chứng minh được danh tính người giữa các tập, ghi `NOT_PROVABLE` trong
ghi chú và giữ pending; không khai báo person-disjoint hoặc sửa policy để vượt gate.

Những tình huống cần ghi nhận cho temporal development và đối chiếu độ phủ holdout:
driver dùng điện thoại; tay gần mặt nhưng không có điện thoại; passenger dùng điện thoại;
điện thoại gắn cố định; dây đai fastened/unfastened/occluded; vào/ra cabin;
người ngoài xe; xe máy; mất rồi lấy lại context; nhiều xe và chuyển động/che khuất.
Các tình huống dàn dựng như dùng điện thoại hoặc tháo đai thực hiện khi xe đỗ an toàn.

## 4. Quy trình sau khi có file và người duyệt

1. Kiểm tra metadata/hash file và lineage; phân nhóm development/holdout trước inference.
2. Người duyệt lập identity roster độc lập từ video, gồm mọi occupant/cabin.
   Dùng namespace trong `docs/identity_roster_contract.md`; không lấy prediction làm GT.
3. Sau khi roster được duyệt thật, dùng công cụ hiện có để extract/freeze identity manifest
   vào đường dẫn phiên bản mới, rồi sinh annotation skeleton cho từng cabin.
4. Người duyệt sửa skeleton thành event intervals và context đầy đủ timeline;
   event dùng frame cuối bao gồm, context dùng `[start_frame, end_frame)`.
   Không suy UNFASTENED từ việc không nhìn thấy dây đai. Kiểm tra role và context thực tế.
   Skeleton AI chưa phải human-approved; chỉ người duyệt thật mới ký provenance.
5. Validate từng annotation bằng `training.validate_event_sequence_annotations`.
   Chỉ export event/context CSV từ các file đã duyệt hợp lệ, giữ riêng hai nhóm.
6. Dùng nhóm temporal development cho calibration/lock policy. Holdout chưa được
   inference trong giai đoạn này. Kiểm tra ACTIVE governed model và mọi điều kiện readiness.
7. Chỉ khi đủ bằng chứng độc lập, độ phủ và human review mới freeze holdout/truth theo
   `docs/V2_EXTERNAL_TEST_PROTOCOL.md`, vào đường dẫn phiên bản mới; final evaluation
   thực hiện đúng protocol sau khi các khóa cần thiết đã hoàn tất.

`tools/annotation_reviewer` hiện là luồng ảnh có hiển thị model proposals;
không dùng queue đó để tạo independent sequence holdout truth.

## 5. Bất biến của đợt tiếp nhận

- `FROZEN_TEST_RUN_COUNT = 1`: không rerun canonical frozen model test.
- Temporal lock tiếp tục `PENDING_SEQUENCE_GROUND_TRUTH` khi chưa có GT hợp lệ.
- Event evaluation tiếp tục `PENDING_NEW_UNTOUCHED_HOLDOUT` khi chưa đủ dữ liệu.
- Không nâng `HUMAN_VERIFIED`, `PRODUCTION_READY` hay claim cross-camera từ gói này.
- Gói tiếp nhận không tải dữ liệu, chạy inference, calibration, freeze hoặc evaluation.
