# Tiếp nhận sequence thật — v2_sequence_001

Trạng thái: **ĐÃ NHẬN VIDEO TEMPORAL LOCAL; CHỜ RIGHTS/LINEAGE, GT VÀ HUMAN REVIEW**. Đây là gói tiếp nhận dữ liệu,
chưa phải ground truth, manifest được freeze hay bằng chứng đánh giá model.
Kết quả kiểm kê nằm trong `readiness.json`; `intake.csv` hiện chỉ có tiêu đề cột.

Đợt tiếp nhận local hiện có **12 MP4/H.264**, tổng **316.732.819 byte** (316,7 MB);
mỗi file dưới 100 MB. Tên file theo mã candidate: C01–C05, C07–C12 và C16.
Đã đổi tên và đối chiếu SHA256 trước/sau cho 12/12 file; không chuyển mã hay đổi timing.
Full-decode video bằng PyAV/FFmpeg đã **PASS 12/12**, tổng **4.821 frame**;
số frame khớp header, không ghi nhận lỗi decode, frame corrupt hoặc timestamp thiếu/không tăng.
Đây là kiểm tra stream video đến EOF; chưa kiểm tra audio hoặc human ground truth.
Báo cáo local: `datasets/incoming/v2_sequence_001/temporal_full_decode_report.json`.
`readiness.json` lưu tóm tắt, thời điểm kiểm tra và SHA256 của báo cáo;
cập nhật tài liệu không chạy lại full-decode hoặc pytest.
Bảng tên gốc/hash/metadata nằm tại
`datasets/incoming/v2_sequence_001/temporal_development/video_file_manifest.json`
(local, gitignored). Video và manifest này **không được tải lên GitHub**;
clone repo ở máy khác không bao gồm các file local này.

Holdout vẫn **0 video**, human-approved sequences vẫn **0**.
Intake local có **12 dòng nháp, 0 dòng hoàn chỉnh** tại
`datasets/incoming/v2_sequence_001/temporal_intake.csv`; CSV template trong Git vẫn trống.
Các dòng nháp có hash, metadata và URL tham chiếu theo research/tên file gốc;
URL này chưa chứng minh nguồn gốc bytes hoặc physical lineage.
Danh sách cần review nằm tại `datasets/incoming/v2_sequence_001/temporal_rights_lineage_review.json`.
Ngày tải và hồ sơ rights/lineage chưa được xác nhận; không suy ngày tải từ filesystem timestamp.
Cả 12 clip đã được người vận hành phân vào development, **không dùng lại cho untouched holdout**,
kể cả đổi tên. Việc phân nhóm này không khẳng định đã chạy training hoặc calibration.
Các bản intake, báo cáo và hồ sơ review local đều gitignored, không được commit.
Chưa xác nhận quyền sử dụng, source/camera/session hay nhóm người/xe vật lý;
mã candidate không thay thế physical identity. Chưa chạy intake gate, inference,
calibration, freeze hoặc evaluation cho đợt tiếp nhận này. Readiness ghi nhận
kiểm kê, full-decode và intake nháp, không nâng trạng thái governance. Người vận hành cần điền bản
intake local cùng rights/lineage evidence và hoàn tất identity/sequence human review
trước các bước tiếp theo.

### Bàn giao cho người duyệt thật

Làm theo [HUMAN_OPERATOR_CHECKLIST.md](HUMAN_OPERATOR_CHECKLIST.md): ưu tiên
**C01 → C07 → C10 → C11**, sau đó C02, C03, C04, C05, C08, C09, C12, C16 nếu cần diversity.
Trạng thái bàn giao: **HUMAN_INPUT_REQUIRED**; rights/lineage/canonical human sequence hoàn tất đều **0/12**.

Gói review local tại `datasets/incoming/v2_sequence_001/temporal_review_package/`
có audit 312 ô intake, 12 mẫu rights, 12 phiếu lineage và đề xuất nội dung từ 108 frame mẫu.
Đề xuất chỉ là **AI_REVIEWED_PROPOSAL**, không phải full-video human review hoặc canonical GT.
Packet timing PASS chỉ xác nhận timeline trình chiếu, không chứng minh capture liên tục hay tốc độ thực.

Ba worksheet làm việc đã chuẩn bị tại `datasets/incoming/v2_sequence_001/human_review_return/`:
`rights_review.csv`, `physical_lineage_review.csv`, `sequence_review.csv`.
Người thật cần xem original full video, điền quyết định và nộp evidence theo checklist.
Bản proposal, evidence gốc và mọi trường reviewer hiện có được giữ nguyên.
Không tự chọn APPROVED/FINAL; lineage không chứng minh được phải ghi NOT_PROVABLE cùng lý do.
Các worksheet, evidence, video và contact sheets chỉ ở local, không đi kèm GitHub.
Chỉ sau khi người thật trả kết quả mới kiểm tra evidence/SHA, reviewer và video binding;
chưa tạo roster, canonical skeleton hay chạy validation/calibration trong bước bàn giao này.

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
- `vehicle_physical_groups`: bắt buộc cho clip có nhiều canonical vehicle, ánh xạ mọi
  canonical vehicle ID tới group xe vật lý. JSONL dùng object; CSV dùng chuỗi JSON được
  quote theo CSV. Group xe chính ở `physical_vehicle_group_id` phải có trong mapping.
  Cả holdout và development phải liệt kê mọi xe; gate so/count toàn bộ giá trị mapping.
- `prior_usage`: dùng `NEVER_USED` khi chưa dùng và có bằng chứng; nếu đã dùng, ghi
  train/validation/calibration/test tương ứng. `model_predictions_seen`: ghi `true`, `false` hoặc `UNKNOWN`
  theo thông tin thực; `UNKNOWN` chưa đủ để xác nhận untouched.
- `independence_evidence_path`: hồ sơ đối chiếu nguồn/camera/video/xe/người với toàn bộ
  development và canonical test cũ. Khác SHA file không đủ chứng minh độc lập.
- `rights_evidence_sha256`, `independence_evidence_sha256`: SHA-256 của hai file hồ sơ
  không rỗng; validator kiểm tra lại nội dung từ disk. Đường dẫn tương đối được tính
  từ working directory khi chạy lệnh (khuyến nghị chạy tại repo root).
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
   Official sequence cần `review_provenance.review_evidence` chứa path/SHA-256 của file
   bằng chứng review không rỗng, và `evidence_hash` bằng SHA đó. File evidence được đọc/hash lại.
5. Validate từng annotation bằng `training.validate_event_sequence_annotations`.
   Chỉ export event/context CSV từ các file đã duyệt hợp lệ, giữ riêng hai nhóm.
6. Dùng nhóm temporal development cho calibration/lock policy. Holdout chưa được
   inference trong giai đoạn này. Kiểm tra ACTIVE governed model và mọi điều kiện readiness.
7. Chỉ khi đủ bằng chứng độc lập, độ phủ và human review mới freeze holdout/truth theo
   `docs/V2_EXTERNAL_TEST_PROTOCOL.md`, vào đường dẫn phiên bản mới; final evaluation
   thực hiện đúng protocol sau khi các khóa cần thiết đã hoàn tất.

`tools/annotation_reviewer` hiện là luồng ảnh có hiển thị model proposals;
không dùng queue đó để tạo independent sequence holdout truth.

## 5. Machine gate trước freeze

Chạy với hai file CSV hoặc JSONL riêng, không trộn role trong holdout input:

```powershell
py -m training.validate_sequence_intake `
  datasets/incoming/v2_sequence_001/holdout_intake.csv `
  datasets/incoming/v2_sequence_001/development_lineage.csv `
  --development-lineage-lock datasets/incoming/v2_sequence_001/development_lineage_frozen.json `
  --output datasets/incoming/v2_sequence_001/intake_check.json
```

File development lineage phải bao phủ toàn bộ train/validation/calibration và dữ liệu
đã dùng trước đây; mỗi dòng phải có SHA/source/camera/video/session/xe vật lý và tất cả
người vật lý. Giữ group ID ổn định xuyên tập; không dùng namespace `video:...` cho
session/xe vật lý/người vật lý. `person_group_ids` nhận JSON list hoặc chuỗi CSV
ngăn cách bằng `;`; kiểm tra giao nhau trên từng ID, không so nguyên chuỗi danh sách.

Trước precheck chính thức, người duyệt phải xác nhận completeness của toàn bộ lineage,
gắn SHA file với bản duyệt HUMAN/APPROVED/FINAL và hồ sơ đối chiếu. Tạo lock bằng
`training.freeze_development_lineage`; xem contract và lệnh đầy đủ trong
`docs/V2_EXTERNAL_TEST_PROTOCOL.md`. CLI intake và official freezer đều bắt lock này,
kiểm tra lại file lineage, bản duyệt và evidence trên disk. Đổi CSV sang JSONL cũng đổi
hash, nên phải review/lock đúng file sẽ truyền vào freezer.

Holdout phải có `proposed_role=NEW_UNTOUCHED_HOLDOUT`, `prior_usage=NEVER_USED`,
`model_predictions_seen=false`, video đúng SHA và hai hồ sơ đúng SHA. Thiếu metadata
development, file/hồ sơ không tồn tại, ID không chứng minh được, overlap hoặc input rỗng
đều trả `NOT_READY_FOR_UNTOUCHED_FREEZE` và exit code 1. File output không bị ghi đè.

Khi chuẩn bị external manifest JSONL, giữ nguyên các field intake này, bổ sung field
annotation/human review theo policy và đặt `dataset_role=EXTERNAL_TEST`.
`annotation_path` trỏ thẳng tới sequence JSON chuẩn đã human-review, không tạo JSON
event-list riêng. Với nhiều cabin, thêm `additional_sequence_annotations` chứa các
object `path`/`sha256` của sequence chuẩn cho những cabin còn lại. Freezer kiểm tra cùng
schema/semantic validator, HUMAN/FINAL, video/manifest binding và đủ mọi occupant.
CSV event/context được export từ chính các sequence này.
Hai truth freezer sẽ tái tạo và đối chiếu toàn bộ rows CSV với đúng sequence SHA đã khóa;
sửa field, thêm/xóa rows sau export đều bị reject. Truth lock lưu
`source_sequence_set_sha256` và `source_sequence_bindings`; evaluator/verifier kiểm tra lại
cả source, evidence và nội dung CSV. Giữ nguyên các file nguồn sau freeze.
`freeze_external_test` tự chạy lại gate trên manifest thực và hồ sơ trên disk;
không tin báo cáo precheck cũ. Official policy bắt `FULL_SYSTEM_EVENT_EVALUATION`.
Số lượng tối thiểu 8 xe/8 người được đếm trên group vật lý, không trên ID theo video.

`SEQUENCE_INTAKE_CHECKS_PASSED` chỉ chứng minh kiểm tra khai báo/đối chiếu/hash đã qua;
nó không xác thực nội dung pháp lý, tính trung thực của hồ sơ hoặc sự đầy đủ của danh mục
development. Những phần đó vẫn cần người duyệt độc lập. Kết quả không thay human approval,
identity manifest, coverage gate hay frozen external-test artifact.

## 6. Bất biến của đợt tiếp nhận

- `FROZEN_TEST_RUN_COUNT = 1`: không rerun canonical frozen model test.
- Temporal lock tiếp tục `PENDING_SEQUENCE_GROUND_TRUTH` khi chưa có GT hợp lệ.
- Event evaluation tiếp tục `PENDING_NEW_UNTOUCHED_HOLDOUT` khi chưa đủ dữ liệu.
- Không nâng `HUMAN_VERIFIED`, `PRODUCTION_READY` hay claim cross-camera từ gói này.
- Validator intake không tải dữ liệu, chạy inference, calibration, freeze hoặc evaluation.
