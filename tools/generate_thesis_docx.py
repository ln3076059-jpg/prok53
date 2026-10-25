"""
Generate the complete Graduation Thesis Word Document (.docx) following the Hanoi Open University
(TRƯỜNG ĐẠI HỌC MỞ HÀ NỘI - TRUNG TÂM ĐÀO TẠO TRỰC TUYẾN) template structure from D:\.idea\mau.docx
and incorporating all scientific/technical contents from FINAL_PROJECT_REPORT.md, quatrinh.md, and V32 AI85 results.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor


def set_cell_bg(cell, hex_color: str) -> None:
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tcPr.append(shd)


def add_formatted_table(doc: Document, headers: list[str], rows: list[list[str]]) -> None:
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    hdr_cells = table.rows[0].cells
    for idx, text in enumerate(headers):
        hdr_cells[idx].text = text
        hdr_cells[idx].vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        set_cell_bg(hdr_cells[idx], "1F4E79")
        for p in hdr_cells[idx].paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in p.runs:
                run.font.bold = True
                run.font.size = Pt(11)
                run.font.name = "Times New Roman"
                run.font.color.rgb = RGBColor(255, 255, 255)

    for r_idx, row_data in enumerate(rows):
        row_cells = table.rows[r_idx + 1].cells
        bg = "F2F6FA" if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, val in enumerate(row_data):
            row_cells[c_idx].text = str(val)
            row_cells[c_idx].vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            set_cell_bg(row_cells[c_idx], bg)
            for p in row_cells[c_idx].paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT if c_idx == 0 else WD_ALIGN_PARAGRAPH.CENTER
                for run in p.runs:
                    run.font.size = Pt(11)
                    run.font.name = "Times New Roman"
    doc.add_paragraph()


def add_heading_styled(doc: Document, text: str, level: int = 1) -> None:
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        run.font.name = "Times New Roman"
        run.font.bold = True
        if level == 1:
            run.font.size = Pt(15)
            run.font.color.rgb = RGBColor(0x0F, 0x3B, 0x66)
        elif level == 2:
            run.font.size = Pt(13)
            run.font.color.rgb = RGBColor(0x16, 0x4E, 0x87)
        else:
            run.font.size = Pt(12)
            run.font.color.rgb = RGBColor(0x1F, 0x1F, 0x1F)


def add_body(doc: Document, text: str, bold_prefix: str = "") -> None:
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.35
    p.paragraph_format.space_after = Pt(6)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    if bold_prefix:
        r_bold = p.add_run(bold_prefix)
        r_bold.font.name = "Times New Roman"
        r_bold.font.size = Pt(13)
        r_bold.font.bold = True
    r = p.add_run(text)
    r.font.name = "Times New Roman"
    r.font.size = Pt(13)


def add_bullet(doc: Document, bold_prefix: str, text: str) -> None:
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.line_spacing = 1.3
    p.paragraph_format.space_after = Pt(4)
    if bold_prefix:
        rb = p.add_run(bold_prefix)
        rb.font.name = "Times New Roman"
        rb.font.size = Pt(13)
        rb.font.bold = True
    r = p.add_run(text)
    r.font.name = "Times New Roman"
    r.font.size = Pt(13)


def build_thesis_document(output_path: Path) -> None:
    doc = Document()

    # Thiết lập khổ giấy A4 và căn lề chuẩn đồ án tốt nghiệp
    for section in doc.sections:
        section.page_width = Cm(21.0)
        section.page_height = Cm(29.7)
        section.top_margin = Cm(2.5)
        section.bottom_margin = Cm(2.5)
        section.left_margin = Cm(3.0)
        section.right_margin = Cm(2.0)

    style_normal = doc.styles["Normal"]
    style_normal.font.name = "Times New Roman"
    style_normal.font.size = Pt(13)

    # ==================== TRANG BÌA ====================
    p_org = doc.add_paragraph()
    p_org.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r1 = p_org.add_run("TRƯỜNG ĐẠI HỌC MỞ HÀ NỘI\nTRUNG TÂM ĐÀO TẠO TRỰC TUYẾN\n--------------------------\n\n")
    r1.font.name = "Times New Roman"
    r1.font.size = Pt(14)
    r1.font.bold = True

    p_type = doc.add_paragraph()
    p_type.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r2 = p_type.add_run("BÁO CÁO ĐỒ ÁN TỐT NGHIỆP\n\n")
    r2.font.name = "Times New Roman"
    r2.font.size = Pt(20)
    r2.font.bold = True
    r2.font.color.rgb = RGBColor(0x0F, 0x3B, 0x66)

    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r3 = p_title.add_run(
        "Đề tài:\nXÂY DỰNG HỆ THỐNG AI PHÁT HIỆN HÀNH VI VI PHẠM AN TOÀN GIAO THÔNG "
        "TRONG KHOANG LÁI (SỬ DỤNG ĐIỆN THOẠI VÀ KHÔNG THẮT DÂY AN TOÀN) "
        "KẾT HỢP SUY LUẬN NHÂN QUẢ V32 AI85 (ROADWATCH DMS)\n\n"
    )
    r3.font.name = "Times New Roman"
    r3.font.size = Pt(15)
    r3.font.bold = True

    p_meta = doc.add_paragraph()
    p_meta.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p_meta.paragraph_format.left_indent = Cm(3.0)
    p_meta.paragraph_format.line_spacing = 1.5
    r_meta = p_meta.add_run(
        "Ngành đào tạo: Công nghệ Thông tin\n"
        "Hệ đào tạo: Đại học Trực tuyến\n"
        "Phiên bản kiến trúc: Roadwatch DMS V32 AI85 (97.44% F1 Score)\n"
        "Môi trường thực nghiệm: Local DMD Full-Resolution + Kaggle Tesla T4 GPU\n\n\n"
    )
    r_meta.font.name = "Times New Roman"
    r_meta.font.size = Pt(13)

    p_year = doc.add_paragraph()
    p_year.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_year = p_year.add_run("HÀ NỘI – NĂM 2026")
    r_year.font.name = "Times New Roman"
    r_year.font.size = Pt(13)
    r_year.font.bold = True

    doc.add_page_break()

    # ==================== LỜI CẢM ƠN & TÓM TẮT ĐỒ ÁN ====================
    add_heading_styled(doc, "LỜI CẢM ƠN", level=1)
    add_body(
        doc,
        "Trong quá trình thực hiện và hoàn thiện Đồ án tốt nghiệp với đề tài “Xây dựng hệ thống AI phát hiện hành vi vi phạm "
        "an toàn giao thông trong khoang lái (Sử dụng điện thoại và Không thắt dây an toàn) kết hợp suy luận nhân quả V32 AI85 "
        "(Roadwatch DMS)”, em đã nhận được sự hướng dẫn tận tình của các thầy cô giáo thuộc Trường Đại học Mở Hà Nội – Trung tâm "
        "Đào tạo Trực tuyến. Em xin bày tỏ lòng biết ơn chân thành tới giảng viên hướng dẫn cùng các thầy cô trong Khoa Công nghệ "
        "Thông tin đã trang bị cho em kiến thức nền tảng vững chắc về Trí tuệ nhân tạo (Artificial Intelligence), Thị giác máy tính "
        "(Computer Vision), Học sâu (Deep Learning) và Kỹ thuật phần mềm để hoàn thành đồ án này."
    )

    add_heading_styled(doc, "TÓM TẮT KẾT QUẢ ĐẠT ĐƯỢC CỦA ĐỒ ÁN", level=1)
    add_body(
        doc,
        "Đồ án thiết kế, huấn luyện và triển khai hoàn chỉnh hệ thống Roadwatch DMS (Driver Monitoring System) phiên bản V32 AI85 "
        "nhằm phát hiện tự động hai hành vi mất an toàn nghiêm trọng nhất trong khoang lái ô tô: (1) Sử dụng điện thoại khi lái xe "
        "(PHONE - bao gồm gọi điện và nhắn tin hai tay) và (2) Không thắt dây an toàn (NO_SEATBELT). Toàn bộ các chỉ số thực nghiệm "
        "chính đều vượt mục tiêu > 85% (AI85 Gate):"
    )
    add_bullet(doc, "Track A (Frozen Temporal Event Matcher - Giao thức chính): ", "Đạt 97.44% F1 Score, 100.00% Precision (0 False Positive), 95.00% Recall (19/20 True Positive), tỷ lệ báo động giả 0.0000 FA/phút trên 15.27 phút video DMD thực tế.")
    add_bullet(doc, "Track B (4-Class Exact Action Attribution): ", "Đạt 97.44% F1 Score, khớp chính xác 19/19 sự kiện theo cả 4 lớp hành vi chi tiết (phonecall_right, phonecall_left, texting_right, texting_left).")
    add_bullet(doc, "Track C (Generic Phone Episode Gap ≤ 3.0s): ", "Đạt 100.00% F1 Score, 100.00% Precision, 100.00% Recall (16/16 chuỗi sự kiện).")
    add_bullet(doc, "Phân loại Dây an toàn (Seatbelt Classifier V32 S3): ", "Đạt 86.69% Binary Accuracy (86.44% Macro-F1) trên tập 616 ảnh kiểm định độc lập và nâng Recall của lớp bị che khuất (uncertain_or_occluded) từ 4.00% lên 60.00%.")
    add_bullet(doc, "Kiểm toán Nhân quả Thời gian thực (Live Causal Audit): ", "Độ trễ kích hoạt trung bình đạt 0.504 giây, không có sự kiện nào bị chốt trễ ở cuối phiên video (0 session-end finalization).")

    doc.add_page_break()

    # ==================== CHƯƠNG 1: TỔNG QUAN ====================
    add_heading_styled(doc, "CHƯƠNG 1: TỔNG QUAN ĐỀ TÀI", level=1)

    add_heading_styled(doc, "1.1. Lý do chọn đề tài và Tính cấp thiết", level=2)
    add_body(
        doc,
        "Tai nạn giao thông đường bộ hiện đang là một trong những thách thức lớn nhất đối với an toàn cộng đồng tại Việt Nam "
        "cũng như trên toàn thế giới. Theo thống kê của Tổ chức Y tế Thế giới (WHO) và Ủy ban An toàn Giao thông Quốc gia, hai nguyên "
        "nhân hàng đầu làm gia tăng tỷ lệ tai nạn và mức độ thương vong nghiêm trọng đối với người điều khiển phương tiện ô tô là: "
        "(1) Mất tập trung do sử dụng điện thoại di động (nhắn tin, lướt màn hình hoặc nghe gọi khi xe đang di chuyển) và "
        "(2) Không thắt dây an toàn (Seatbelt) dẫn đến chấn thương nặng khi xảy ra va chạm."
    )
    add_body(
        doc,
        "Tại Việt Nam, Nghị định 100/2019/NĐ-CP (sửa đổi bởi Nghị định 123/2021/NĐ-CP) và Nghị định 168/2024/NĐ-CP quy định bắt buộc "
        "lắp đặt thiết bị giám sát hành trình và camera ghi nhận hình ảnh người lái xe đối với các phương tiện kinh doanh vận tải. "
        "Tuy nhiên, phần lớn các hệ thống hiện nay mới dừng ở việc ghi hình lưu trữ thủ công hoặc sử dụng một mô hình nhận diện vật thể "
        "đơn lẻ (Single-frame Object Detector). Cách tiếp cận đơn lẻ này gặp hạn chế lớn trong môi trường khoang lái thực tế do ánh sáng "
        "yếu, góc quay nghiêng, sự che khuất của vô-lăng, trang phục tối màu trùng màu dây đai an toàn, hoặc nhầm lẫn hành khách sử dụng "
        "điện thoại với tài xế. Từ thực tiễn đó, việc nghiên cứu và xây dựng hệ thống “Roadwatch DMS V32 AI85” kết hợp thị giác máy tính "
        "đa mô hình (Multi-stage Specialist Vision) với suy luận giải phẫu chuẩn hóa (Body-Normalized Pose Reasoning) và thuật toán tách "
        "sự kiện nhân quả theo thời gian thực (Causal Boundary Collector) là hết sức cấp thiết và có ý nghĩa khoa học lẫn thực tiễn cao."
    )

    add_heading_styled(doc, "1.2. Mục tiêu của đề tài", level=2)
    add_bullet(doc, "Mục tiêu tổng quát: ", "Xây dựng hoàn chỉnh hệ thống phần mềm và pipeline Trí tuệ nhân tạo có khả năng tự động phân tích video khoang lái, phát hiện chính xác hành vi sử dụng điện thoại (PHONE) và không thắt dây an toàn (NO_SEATBELT), trích xuất gói bằng chứng đa phương tiện toàn vẹn mã băm SHA-256 và hiển thị trực tiếp video kết quả kèm khung nhận diện (Bounding Box) trên giao diện Web.")
    add_bullet(doc, "Mục tiêu khoa học (Cổng chất lượng AI85 - Tất cả chỉ số chính > 85%): ", "Khắc phục triệt để 5 điểm nghẽn của các hệ thống thế hệ trước (V1/V31 chỉ đạt 75.68% F1) để đưa toàn bộ các chỉ số đánh giá cấp độ sự kiện (Event-level Precision, Recall, F1 Score) và độ chính xác phân loại dây an toàn vượt ngưỡng 85% mà không vi phạm tính nhân quả thời gian thực.")
    add_bullet(doc, "Mục tiêu ứng dụng: ", "Xây dựng ứng dụng Web hoàn chỉnh gồm Backend FastAPI + SQLAlchemy và Frontend React + TypeScript (giao diện tiếng Việt rõ ràng, giữ nguyên thuật ngữ khoa học chuẩn quốc tế), cho phép tải video lên, phân tích tự động, phát lại video kèm lớp phủ Bounding Box trực tiếp, xem video bằng chứng evidence.mp4 và phê duyệt sự kiện.")

    add_heading_styled(doc, "1.3. Đối tượng và Phạm vi nghiên cứu", level=2)
    add_bullet(doc, "Đối tượng nghiên cứu: ", "Các mô hình học sâu phát hiện vật thể và khung xương người thế hệ mới (YOLO11s, YOLO11n-pose, YOLO11s-cls), thuật toán chuẩn hóa khoảng cách giải phẫu theo đường chéo cơ thể, thuật toán gom/tách sự kiện nhân quả theo chuỗi thời gian (Causal Temporal Event Collector), và kiến trúc hệ thống quản lý bằng chứng vi phạm.")
    add_bullet(doc, "Phạm vi dữ liệu và thực nghiệm: ", "Thực nghiệm trên bộ dữ liệu chuẩn quốc tế Driver Monitoring Dataset (DMD) với nhãn sự kiện OpenLABEL chính thức (các đối tượng gC-14, gZ-36 có độ phân giải Full HD 1920x1080) và bộ dữ liệu kiểm định dây an toàn độc lập gồm 641 mẫu cắt vùng ngực/vai tài xế.")

    add_heading_styled(doc, "1.4. Phương pháp nghiên cứu", level=2)
    add_bullet(doc, "Phương pháp nghiên cứu lý thuyết: ", "Tổng hợp các công trình nghiên cứu về Driver Monitoring System (DMS), bài toán nhãn bộ phận (Partial-label conflict) khi kết hợp nhiều tập dữ liệu công khai, và phương pháp hiệu chuẩn xác suất hậu nghiệm có xét đến độ không chắc chắn (Uncertainty-aware Calibration).")
    add_bullet(doc, "Phương pháp thực nghiệm định lượng có kiểm soát (Governed Scientific Benchmarking): ", "Thiết lập bộ quy tắc đối sánh sự kiện đóng băng bằng mã băm SHA-256 (reports/DMD_EVENT_MATCHING_POLICY_FREEZE.json), kết hợp chạy suy luận trên máy trạm cục bộ đối với video DMD gốc và kiểm chứng hiệu năng phần cứng trên GPU đám mây Kaggle Tesla T4.")

    doc.add_page_break()

    # ==================== CHƯƠNG 2: KHẢO SÁT VÀ PHÂN TÍCH HỆ THỐNG ====================
    add_heading_styled(doc, "CHƯƠNG 2: KHẢO SÁT VÀ PHÂN TÍCH YÊU CẦU HỆ THỐNG", level=1)

    add_heading_styled(doc, "2.1. Khảo sát hiện trạng và Bài toán Xung đột Nhãn bộ phận (Partial-Label Conflict)", level=2)
    add_body(
        doc,
        "Trong giai đoạn đầu của dự án (thế hệ V1 — MC_BOOTSTRAP_001), nhóm nghiên cứu đã khảo sát phương pháp truyền thống: "
        "gom nhiều tập dữ liệu công khai từ nguồn mở để huấn luyện một mô hình YOLO11s 3 lớp duy nhất gồm: phone, seatbelt_fastened, "
        "và seatbelt_unfastened. Kết quả kiểm định cho thấy cách tiếp cận đơn khối (Monolithic Detector) gặp thất bại nghiêm trọng "
        "với mAP@50 toàn cục chỉ đạt 0.3408 và F1-score cấp độ khung hình đạt 0.3802."
    )
    add_body(
        doc,
        "Qua phân tích chuyên sâu, nguyên nhân cốt lõi được xác định là hiện tượng Xung đột Nhãn bộ phận (Partial-Label Conflict): "
        "các bộ dữ liệu dây an toàn không gán nhãn cho điện thoại (khiến điện thoại bị phạt thành nền âm tính giả), trong khi các bộ "
        "dữ liệu điện thoại lại không gán nhãn dây an toàn (khiến dây an toàn bị phạt thành nền). Hơn nữa, một khung bao (Bounding Box) "
        "của điện thoại trên một khung hình đơn lẻ không đủ cơ sở để kết luận tài xế đang vi phạm nếu thiếu ngữ cảnh vị trí ghế ngồi "
        "(Driver ROI), mối liên hệ giải phẫu với bàn tay/vùng tai-mặt và sự duy trì liên tục theo thời gian."
    )

    add_heading_styled(doc, "2.2. Phân tích Yêu cầu Chức năng và Phi chức năng", level=2)
    add_bullet(doc, "Nhóm chức năng Xác thực & Phân quyền (Authentication & RBAC): ", "Đăng nhập bảo mật bằng JWT Bearer Token, phân tách quyền Quản trị viên / Kiểm duyệt viên (Reviewer) và quyền Xuất báo cáo (Exporter).")
    add_bullet(doc, "Nhóm chức năng Tải lên & Phân tích Video Tự động: ", "Cho phép tải lên các định dạng video phổ biến (MP4, MOV, AVI, MKV, WebM), kiểm tra mã băm SHA-256, khởi tạo tiến trình phân tích bất đồng bộ (AnalysisJob) và hiển thị trực tiếp video sau khi xử lý kèm lớp phủ Bounding Box ngay trên trang Tải lên.")
    add_bullet(doc, "Nhóm chức năng Suy luận AI Đa tầng V32: ", "Thực hiện 4 giai đoạn: (1) Định vị khoang lái & vùng thân trên tài xế, (2) Nhận diện điện thoại đa vùng cắt D5 + lọc nhiễu tỷ lệ khung hình, (3) Suy luận tư thế giải phẫu chuẩn hóa theo đường chéo cơ thể YOLO11-pose (phân biệt LAP_TEXTING và EAR_CALL), (4) Tách sự kiện nhân quả Causal Boundary Collector & phân loại dây an toàn Seatbelt S3.")
    add_bullet(doc, "Nhóm chức năng Đóng gói Bằng chứng & Kiểm duyệt (Evidence & Human Review): ", "Mỗi sự kiện vi phạm tự động sinh ra ảnh gốc (original_keyframe.jpg), ảnh vẽ khung nhận diện (annotated_keyframe.jpg), đoạn video bằng chứng (evidence.mp4) và nhật ký truy vết (trace.json) được khóa bằng mã băm toàn vẹn SHA-256.")
    add_bullet(doc, "Nhóm chức năng Báo cáo Thực nghiệm V32 AI85 & Thống kê: ", "Cung cấp bảng điều khiển trực quan hóa kết quả Track A/B/C, biểu đồ Timeline đối sánh OpenLABEL Ground Truth với dự đoán nhân quả V32, ma trận nhầm lẫn dây an toàn và thông số GPU Kaggle Tesla T4.")

    add_heading_styled(doc, "2.3. Biểu đồ Phân rã Chức năng (BFD - Business Function Diagram)", level=2)
    add_body(
        doc,
        "Hệ thống Roadwatch DMS V32 AI85 được phân rã thành 5 phân hệ chức năng chính như trình bày trong Bảng 2.1 dưới đây:"
    )
    add_formatted_table(
        doc,
        ["Mã phân hệ", "Tên phân hệ chức năng (BFD)", "Các chức năng thành phần chi tiết"],
        [
            ["F1", "Quản lý Truy cập & Cấu hình", "1.1 Đăng nhập JWT; 1.2 Phân quyền người dùng; 1.3 Cấu hình tham số V32 (Driver side, cửa sổ thời gian, ngưỡng tỷ lệ khung, ngưỡng Seatbelt S3)."],
            ["F2", "Tiếp nhận & Phân tích Video", "2.1 Tải video lên & kiểm tra SHA-256; 2.2 Khởi tạo AnalysisJob; 2.3 Phát trực tiếp video đã xử lý kèm lớp phủ Bounding Box Canvas trên trang Upload."],
            ["F3", "Pipeline Suy luận AI Đa tầng V32", "3.1 Định vị Cabin & Occupant ROI; 3.2 Nhận diện D5 Specialist + Auxiliary; 3.3 Chuẩn hóa khoảng cách giải phẫu YOLO11-pose; 3.4 Causal Boundary Collector & Seatbelt S3."],
            ["F4", "Quản lý Bằng chứng & Kiểm duyệt", "4.1 Sinh gói bằng chứng (Keyframe, Clip MP4, Trace JSON, SHA-256); 4.2 Phê duyệt sự kiện (Confirm / Reject / Needs Review); 4.3 Kiểm duyệt nghiên cứu (13 FN Cases)."],
            ["F5", "Thực nghiệm V32 & Xuất Báo cáo", "5.1 Hiển thị ma trận thực nghiệm Track A/B/C; 5.2 Biểu đồ Timeline OpenLABEL vs V32; 5.3 Thống kê vận hành & Xuất tệp báo cáo CSV."],
        ],
    )

    add_heading_styled(doc, "2.4. Đặc tả Ca sử dụng (Use Case Specification)", level=2)
    add_formatted_table(
        doc,
        ["Mã UC", "Tên Use Case", "Tác nhân", "Mô tả luồng xử lý chính"],
        [
            ["UC01", "Đăng nhập hệ thống", "Kiểm duyệt viên / Admin", "Người dùng nhập Email & mật khẩu tại /login -> Hệ thống xác thực bcrypt/argon2 và cấp JWT Bearer Token."],
            ["UC02", "Tải lên, Phân tích & Xem trực tiếp Video đã xử lý", "Kiểm duyệt viên / Admin", "Chọn tệp video tại /upload -> Tải lên server -> Chạy pipeline V32 -> Ngay khi hoàn tất (100%), trang /upload tự động phát video kèm khung Bounding Box AI trực tiếp và hiển thị danh sách sự kiện + clip bằng chứng."],
            ["UC03", "Kiểm tra Bằng chứng & Phê duyệt Sự kiện", "Kiểm duyệt viên", "Mở chi tiết sự kiện tại /events/{id} -> Xem ảnh Annotated Keyframe, phát video evidence.mp4, kiểm tra mã SHA-256 và bấm Xác nhận (CONFIRMED) hoặc Từ chối (REJECTED)."],
            ["UC04", "Tra cứu Thực nghiệm V32 AI85 & Timeline Nhân quả", "Giảng viên / Nghiên cứu viên", "Truy cập /benchmark -> Xem bảng so sánh V31 vs V32 (Track A/B/C), chọn đối tượng gC-14 hoặc gZ-36 để xem biểu đồ Timeline và nhật ký quyết định nhân quả START/END."],
            ["UC05", "Thống kê & Xuất báo cáo CSV", "Người vận hành", "Truy cập /events hoặc /statistics -> Lọc theo loại vi phạm (PHONE / NO_SEATBELT), trạng thái duyệt và tải về báo cáo events.csv."],
        ],
    )

    add_heading_styled(doc, "2.5. Biểu đồ Luồng Dữ liệu (DFD - Data Flow Diagram)", level=2)
    add_body(
        doc,
        "Luồng dữ liệu mức ngữ cảnh (DFD Mức 0) và mức đỉnh (DFD Mức 1) của hệ thống vận hành theo chu trình khép kín: "
        "(1) Luồng video đầu vào được lưu trữ an toàn tại thư mục uploads/ và ghi nhận bản ghi Video vào cơ sở dữ liệu; "
        "(2) Tiến trình InferenceWorker đọc từng khung hình của video, đưa qua bộ phát hiện đa mô hình SafetyDetector và bộ "
        "phân tích thời gian VideoAnalyzer; (3) Các khung bao đối tượng kèm thông tin tư thế (phone_posture, source_action) được "
        "lưu vào bảng Detections; (4) Khi bộ thu thập nhân quả xác nhận sự kiện vi phạm, hệ thống trích xuất gói bằng chứng "
        "vào thư mục evidence/<event_id>/ và ghi nhận vào bảng Events, Evidence; (5) Giao diện Frontend truy vấn API /videos/{id}/analysis-summary "
        "và /videos/{id}/stream để đồng bộ hóa việc phát video với vẽ khung Bounding Box thời gian thực."
    )

    doc.add_page_break()

    # ==================== CHƯƠNG 3: THIẾT KẾ HỆ THỐNG & GIẢI THUẬT V32 AI85 ====================
    add_heading_styled(doc, "CHƯƠNG 3: THIẾT KẾ HỆ THỐNG VÀ CẢI TIẾN KIẾN TRÚC V32 AI85", level=1)

    add_heading_styled(doc, "3.1. Kiến trúc Tổng thể Đa tầng của Hệ thống Roadwatch V32", level=2)
    add_body(
        doc,
        "Thay vì sử dụng một mô hình nhận diện đơn lẻ, Roadwatch V32 được thiết kế theo kiến trúc 4 giai đoạn phân cấp có kiểm soát "
        "(Multi-Stage Governed Causal Architecture):"
    )
    add_bullet(doc, "Giai đoạn 1 — Định vị Khoang lái & Theo dõi Người ngồi (Cabin & Occupant ROI Tracking): ", "Sử dụng mô hình phát hiện người kết hợp bộ theo dõi đa đối tượng (Tracker) để tách biệt vùng thân trên của Tài xế (driver) và Hành khách (passenger), duy trì vai trò đã xác thực qua các khung hình bị che khuất tạm thời.")
    add_bullet(doc, "Giai đoạn 2 — Nhận diện Điện thoại Chuyên biệt Đa vùng cắt (D5 Specialist + Auxiliary Consensus): ", "Kết hợp quét toàn khung hình (Full-frame pass) và quét lần hai trên vùng cắt phóng đại của tài xế (Driver ROI pass) bằng mô hình chuyên biệt D5 (models/active/v2/phone_detector.pt), đồng thời đối chiếu với mô hình phụ trợ COCO và lọc bỏ các dải nhiễu dọc có tỷ lệ khung hình > 3.0.")
    add_bullet(doc, "Giai đoạn 3 — Suy luận Giải phẫu Chuẩn hóa theo Khung xương (Body-Normalized Pose Reasoning): ", "Sử dụng mô hình YOLO11n-pose trích xuất 17 điểm khớp cơ thể (mũi, mắt, tai, vai, khuỷu tay, cổ tay). Khoảng cách từ tâm điện thoại đến cổ tay và đến vùng tai/mặt được chuẩn hóa theo đường chéo khung bao cơ thể tài xế, giúp bất biến với độ phân giải video và khoảng cách camera.")
    add_bullet(doc, "Giai đoạn 4 — Tách Sự kiện Nhân quả (Causal Boundary Collector) & Hiệu chuẩn Dây an toàn Seatbelt S3: ", "Tích lũy bằng chứng trên cửa sổ trượt 2.0 giây, tự động tách các pha chuyển tiếp liên tiếp giữa Nhắn tin (LAP_TEXTING) và Nghe gọi (EAR_CALL), đồng thời áp dụng quy tắc hiệu chuẩn độ không chắc chắn S3 cho bộ phân loại dây an toàn 3 lớp.")

    add_heading_styled(doc, "3.2. Phân tích 5 Nguyên nhân Gốc rễ khiến Phiên bản V31 dưới 85% và Giải pháp Đột phá V32", level=2)
    add_body(
        doc,
        "Trước khi nâng cấp lên V32, phiên bản V31 Baseline chỉ đạt 75.68% F1 Score trên giao thức chính Track A (14 TP, 3 FP, 6 FN). "
        "Thông qua phân tích truy vết từng khung hình trên tập dữ liệu DMD, đồ án đã phát hiện và giải quyết triệt để 5 nguyên nhân kỹ thuật "
        "cốt lõi như tổng hợp tại Bảng 3.1:"
    )
    add_formatted_table(
        doc,
        ["Mã lỗi", "Mô-đun ảnh hưởng", "Nguyên nhân gốc rễ (Root Cause)", "Giải pháp Cải tiến V32", "Tác động Định lượng"],
        [
            [
                "Bug 1",
                "backend/ai/auxiliary.py",
                "Khoảng cách cổ tay & tai/mặt bị chia cố định cho 180.0 px (chuẩn crop nhỏ). Trên video DMD Full HD 1920x1080, khoảng cách 140–260 px làm hand_proximity = 0.0 khi tài xế nhắn tin dưới đùi.",
                "Chuẩn hóa khoảng cách theo đường chéo khung xương tài xế (diag = hypot(w, h)), bổ sung vùng đùi/ngực (in_lap_or_chest_zone) và xác nhận tư thế cầm tay.",
                "Khôi phục toàn bộ các sự kiện nhắn tin trên đùi (texting_right / texting_left) tại gZ-36.",
            ],
            [
                "Bug 2",
                "training/dmd/evaluate_dmd_Event_v2.py",
                "Tài xế chuyển trực tiếp từ Nhắn tin sang Đưa điện thoại lên tai gọi (khoảng nghỉ < 1s). Bộ gom cũ nối liền thành một sự kiện dài 35–42 giây.",
                "Xây dựng thuật toán phân loại tư thế nhân quả (LAP_TEXTING vs EAR_CALL) và tự động tách sự kiện (POSTURE_TRANSITION_SPLIT) khi tư thế mới duy trì >= 0.45s.",
                "Tách chính xác các cặp sự kiện nối tiếp, khôi phục 5 True Positives bị gộp.",
            ],
            [
                "Bug 3",
                "backend/ai/auxiliary.py & video_analyzer.py",
                "Các dải viền dọc ở mép ghế/cột A có tỷ lệ cạnh h/w từ 3.2 đến 6.0 bị nhận nhầm là điện thoại ở 1 khung hình rồi tự bắc cầu (self-bridging).",
                "Áp dụng hàm is_valid_phone_aspect_ratio (<= 3.0) và chặn không cho ứng viên đơn lẻ có điểm số thấp (< 0.48) tự nhân bản.",
                "Loại bỏ 100% các báo động giả (3 FP giảm xuống còn 0 FP; Precision đạt 100%).",
            ],
            [
                "Bug 4",
                "evaluate_dmd_Event_v2.py",
                "Ngưỡng thời lượng dương tính tối thiểu 0.90s làm bỏ sót các thao tác sử dụng điện thoại ngắn (1.4s – 2.2s).",
                "Hiệu chuẩn ngưỡng kích hoạt nhân quả xuống 0.50s kết hợp bộ lọc tư thế giải phẫu và tỷ lệ khung hình.",
                "Giảm độ trễ kích hoạt trung bình xuống 0.504s và bắt trọn các sự kiện ngắn.",
            ],
            [
                "Bug 5",
                "backend/ai/auxiliary.py (Seatbelt S3)",
                "Quy tắc argmax thuần túy làm lớp bị che khuất (uncertain_or_occluded) chỉ đạt Recall 4.00% (1/25 mẫu).",
                "Thiết kế bộ hiệu chuẩn xác suất S3 (tau_u = 0.055, margin = 0.22) định tuyến các mẫu có biên độ quyết định hẹp về lớp không chắc chắn.",
                "Nâng Recall lớp uncertain từ 4.00% lên 60.00% và Balanced Accuracy 3 lớp lên 74.22%.",
            ],
        ],
    )

    add_heading_styled(doc, "3.3. Thiết kế Cơ sở Dữ liệu Quan hệ (Database Schema Design)", level=2)
    add_body(
        doc,
        "Cơ sở dữ liệu của hệ thống được thiết kế chuẩn hóa trên SQLAlchemy ORM (tương thích SQLite và PostgreSQL), bao gồm 8 bảng "
        "thực thể cốt lõi như mô tả trong Bảng 3.2:"
    )
    add_formatted_table(
        doc,
        ["Tên bảng", "Khóa chính", "Các trường dữ liệu chính", "Mô tả chức năng lưu trữ"],
        [
            ["users", "id (String 32)", "email, password_hash, role, created_at", "Lưu tài khoản quản trị viên và kiểm duyệt viên (RBAC)."],
            ["cameras", "id (String 32)", "name, driver_side, driver_roi, stream_url_encrypted", "Lưu cấu hình hình học camera và vùng ROI ghế lái."],
            ["videos", "id (String 32)", "original_name, storage_path, sha256, size_bytes, mime_type, input_scope", "Lưu thông tin tệp video tải lên và mã băm toàn vẹn SHA-256."],
            ["analysis_jobs", "id (String 32)", "video_id (FK), status, progress, error, created_at, completed_at", "Quản lý trạng thái tiến trình phân tích bất đồng bộ (QUEUED, RUNNING, COMPLETED, FAILED)."],
            ["detections", "id (String 32)", "job_id (FK), frame_number, timestamp_seconds, class_name, confidence, bbox (JSON), occupant_role", "Lưu toàn bộ các Bounding Box phát hiện theo từng khung hình kèm phone_posture và source_action."],
            ["events", "id (String 32)", "job_id (FK), video_id (FK), event_type, confidence, frame_number, timestamp_seconds, review_status", "Lưu các sự kiện vi phạm cấp cao (PHONE, NO_SEATBELT) đã qua bộ lọc nhân quả."],
            ["evidence", "id (String 32)", "event_id (FK), original_path, annotated_path, driver_crop_path, sha256", "Lưu đường dẫn tệp bằng chứng hình ảnh/video và mã băm gói bằng chứng SHA-256."],
            ["reviews", "id (String 32)", "event_id (FK), user_id (FK), previous_status, new_status, notes, created_at", "Nhật ký kiểm toán bất biến (Append-only) lưu lịch sử phê duyệt của con người."],
        ],
    )

    doc.add_page_break()

    # ==================== CHƯƠNG 4: CÀI ĐẶT, THỰC NGHIỆM VÀ KIỂM THỬ HỆ THỐNG ====================
    add_heading_styled(doc, "CHƯƠNG 4: CÀI ĐẶT, THỰC NGHIỆM KHOA HỌC VÀ KIỂM THỬ HỆ THỐNG", level=1)

    add_heading_styled(doc, "4.1. Môi trường Cài đặt và Triển khai Thực tế", level=2)
    add_bullet(doc, "Ngôn ngữ và Framework Backend: ", "Python 3.11, FastAPI, Uvicorn, SQLAlchemy 2.0, Pydantic v2, OpenCV (cv2), PyTorch và Ultralytics YOLO11.")
    add_bullet(doc, "Ngôn ngữ và Framework Frontend: ", "TypeScript, React 18, Vite 7, IBM Carbon Design System (@carbon/react), HTML5 Video & Canvas 2D Overlay.")
    add_bullet(doc, "Hạ tầng Thực nghiệm Kép (Hybrid Execution): ", "Suy luận trực tiếp trên video DMD độ phân giải gốc tại máy trạm cục bộ (đảm bảo tuyệt đối chính sách bảo mật dữ liệu cá nhân của DMD) kết hợp kiểm chứng hiệu năng phần cứng trên GPU đám mây Kaggle NVIDIA Tesla T4 (16 GB VRAM).")

    add_heading_styled(doc, "4.2. Kết quả Thực nghiệm Phát hiện Hành vi Sử dụng Điện thoại trên Tập dữ liệu DMD (Track A, B, C)", level=2)
    add_body(
        doc,
        "Hệ thống V32 AI85 được đánh giá trên 2 phiên video DMD phát triển có đầy đủ nhãn OpenLABEL chính thức là gC-14 "
        "(thời lượng 400.00 giây, 795 khung hình lấy mẫu, 9 sự kiện GT) và gZ-36 (thời lượng 515.81 giây, 1,024 khung hình lấy mẫu, "
        "11 sự kiện GT). Kết quả so sánh định lượng giữa V31 Baseline và V32 AI85 trên cả 3 giao thức đánh giá được trình bày tại Bảng 4.1:"
    )
    add_formatted_table(
        doc,
        ["Giao thức / Tập đánh giá", "Tổng GT", "TP", "FP", "FN", "Precision (%)", "Recall (%)", "F1 Score (%)", "FA / Phút", "Đánh giá (>85%)"],
        [
            ["V31 Baseline (Trước nâng cấp - Track A)", "20", "14", "3", "6", "82.35%", "70.00%", "75.68%", "0.1965", "KHÔNG ĐẠT (<85%)"],
            ["V32 Track A — Frozen Temporal Matcher (Tổng hợp)", "20", "19", "0", "1", "100.00%", "95.00%", "97.44%", "0.0000", "ĐẠT VƯỢT BẬC (+21.76%)"],
            ["  ↳ Đối tượng DMD gC-14 (6.67 phút)", "9", "9", "0", "0", "100.00%", "100.00%", "100.00%", "0.0000", "ĐẠT TUYỆT ĐỐI (100%)"],
            ["  ↳ Đối tượng DMD gZ-36 (8.60 phút)", "11", "10", "0", "1", "100.00%", "90.91%", "95.24%", "0.0000", "ĐẠT (>90%)"],
            ["V32 Track B — Khớp chính xác 4 Lớp Hành vi", "20", "19", "0", "1", "100.00%", "95.00%", "97.44%", "0.0000", "ĐẠT (97.44%)"],
            ["V32 Track C — Chuỗi sự kiện Điện thoại (Gap ≤ 3s)", "16", "16", "0", "0", "100.00%", "100.00%", "100.00%", "0.0000", "ĐẠT TUYỆT ĐỐI (100%)"],
        ],
    )
    add_body(
        doc,
        "Nhận xét khoa học: Trên giao thức chính Track A và giao thức 4 lớp hành vi Track B, hệ thống V32 đạt 97.44% F1 Score "
        "với 100.00% Precision (không có bất kỳ báo động giả nào trong suốt 15.27 phút lái xe). Trường hợp False Negative duy nhất "
        "còn lại ở gZ-36 (từ 162.89s đến 163.80s) là một đoạn chuyển tiếp cực ngắn chỉ dài 0.91 giây khi tài xế đổi tay cầm điện thoại "
        "trước khi áp lên tai nghe cuộc gọi lúc 163.81s. Khi đánh giá theo chuỗi sự kiện sử dụng điện thoại liên tục (Track C), hệ thống "
        "đạt 100.00% trên cả Precision, Recall và F1 Score."
    )

    add_heading_styled(doc, "4.3. Kết quả Thực nghiệm Phân loại Dây an toàn (Seatbelt Classifier V32 S3)", level=2)
    add_body(
        doc,
        "Mô hình phân loại dây an toàn (models/active/v2/seatbelt_classifier.pt) được kiểm định độc lập trên 641 ảnh cắt vùng ngực/vai "
        "tài xế (bao gồm 616 mẫu nhị phân rõ ràng và 25 mẫu bị che khuất hoặc thiếu sáng thuộc lớp uncertain_or_occluded). Kết quả thực nghiệm "
        "được trình bày tại Bảng 4.2:"
    )
    add_formatted_table(
        doc,
        ["Tập kiểm định / Chính sách phân loại", "Số mẫu (n)", "Accuracy (%)", "Balanced Accuracy (%)", "Macro-F1 (%)", "Recall lớp Uncertain"],
        [
            ["Phân loại Nhị phân (Fastened vs Unfastened)", "616", "86.69%", "86.44%", "86.44%", "N/A (2 lớp)"],
            ["Phân loại 3 Lớp S2 Baseline (Raw Argmax)", "641", "83.46%", "60.19%", "59.88%", "4.00% (1/25)"],
            ["Phân loại 3 Lớp V32 S3 Calibrated (τ_u=0.055, m=0.22)", "641", "81.90%", "74.22%", "64.10%", "60.00% (15/25)"],
        ],
    )

    add_heading_styled(doc, "4.4. Kết quả Kiểm chứng Phần cứng trên GPU Kaggle Tesla T4", level=2)
    add_body(
        doc,
        "Để chứng minh khả năng vận hành tốc độ cao trên phần cứng GPU chuyên dụng, gói mã nguồn V32 đã được đẩy lên Kaggle và thực thi "
        "trên nhân GPU NVIDIA Tesla T4 (Kernel: giangtoan/roadwatch-v27-gpu-preflight, Version 2). Kết quả đo đạc độ trễ suy luận (Latency) "
        "và mức tiêu thụ bộ nhớ VRAM được trình bày tại Bảng 4.3:"
    )
    add_formatted_table(
        doc,
        ["Giai đoạn Suy luận (Inference Stage)", "Thiết bị thực thi", "Độ trễ Trung vị (p50)", "Độ trễ Phân vị 95 (p95)", "Độ trễ Trung bình (Mean)", "VRAM Đỉnh"],
        [
            ["Phone Specialist Stage (D5)", "cuda:0 (Tesla T4)", "14.43 ms", "21.97 ms", "15.10 ms", "109.58 MB"],
            ["Auxiliary Object Detector Stage", "cuda:0 (Tesla T4)", "13.38 ms", "14.12 ms", "13.31 ms", "109.58 MB"],
            ["Anatomical Pose Stage (YOLO11n-pose)", "cuda:0 (Tesla T4)", "16.44 ms", "18.88 ms", "16.53 ms", "109.58 MB"],
        ],
    )

    add_heading_styled(doc, "4.5. Cài đặt Giao diện Web Tiếng Việt & Tính năng Hiển thị Trực tiếp Video Đã Xử lý", level=2)
    add_body(
        doc,
        "Hệ thống giao diện người dùng (Frontend) đã được Việt hóa rõ ràng, dễ hiểu đối với tất cả các tiêu đề, thanh điều hướng, nhãn nút bấm "
        "và hướng dẫn vận hành, đồng thời giữ nguyên các thuật ngữ khoa học chuẩn quốc tế (Precision, Recall, F1 Score, IoU, Track A/B/C, "
        "OpenLABEL Ground Truth, Causal Boundary Collector, LAP_TEXTING, EAR_CALL, YOLO11-pose, ROI, Macro-F1, Balanced Accuracy, SHA-256). "
        "Đặc biệt, trang Tải lên & Phân tích Video (/upload) được tích hợp trình phát HTML5 Video đồng bộ với lớp phủ HTML5 Canvas 2D, cho phép:"
    )
    add_bullet(doc, "Hiển thị trực tiếp video ngay sau khi xử lý xong: ", "Tự động nạp luồng video từ endpoint GET /videos/{video_id}/stream và dữ liệu tổng hợp từ GET /videos/{video_id}/analysis-summary ngay khi tiến trình đạt 100%.")
    add_bullet(doc, "Vẽ khung Bounding Box AI theo thời gian thực: ", "Khi video phát hoặc khi người dùng tua tới bất kỳ thời điểm nào, hệ thống tự động vẽ khung nhận diện màu đỏ cho PHONE (kèm nhãn tư thế LAP_TEXTING / EAR_CALL và độ tự tin) và khung màu xanh cho vùng DRIVER ROI.")
    add_bullet(doc, "Xem ngay Ảnh Keyframe Đánh dấu & Video Bằng chứng (evidence.mp4): ", "Hiển thị song song các thẻ sự kiện vi phạm kèm ảnh annotated_keyframe.jpg và đoạn video bằng chứng evidence.mp4 đã được backend trích xuất, cùng bảng danh sách Detections từng khung hình có nút bấm tua nhanh.")

    add_heading_styled(doc, "4.6. Kiểm thử Tự động Toàn diện (Automated Test Suite)", level=2)
    add_body(
        doc,
        "Toàn bộ hệ thống Backend, Pipeline AI V32 và các cổng bảo mật được kiểm thử tự động bằng bộ 107 bài kiểm thử Pytest "
        "(bao gồm tests/test_v32_ai85_execution.py, tests/test_core.py, tests/test_runtime_v2.py và tests/test_v2.py). Kết quả chạy "
        "kiểm thử đạt 107/107 bài PASS tuyệt đối (100%)."
    )

    doc.add_page_break()

    # ==================== CHƯƠNG 5: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN ====================
    add_heading_styled(doc, "CHƯƠNG 5: KẾT LUẬN VÀ ĐÁNH GIÁ", level=1)

    add_heading_styled(doc, "5.1. Những kết quả đã đạt được", level=2)
    add_bullet(doc, "Về mặt phương pháp luận và thuật toán AI: ", "Đồ án đã chứng minh thành công nguyên nhân gây suy giảm độ chính xác của mô hình đơn khối V1 (xung đột nhãn bộ phận) và khắc phục 5 điểm nghẽn của phiên bản V31 thông qua kiến trúc V32 AI85: chuẩn hóa khoảng cách giải phẫu theo đường chéo cơ thể, lọc nhiễu tỷ lệ khung hình <= 3.0, tách chuyển tiếp tư thế nhân quả (LAP_TEXTING <-> EAR_CALL) và hiệu chuẩn xác suất dây an toàn 3 lớp Seatbelt S3.")
    add_bullet(doc, "Về mặt chỉ số thực nghiệm: ", "Đưa toàn bộ các chỉ số chính vượt ngưỡng 85%: Track A đạt 97.44% F1 (100% Precision, 95% Recall), Track B đạt 97.44% F1, Track C đạt 100.00% F1, và Seatbelt Binary Accuracy đạt 86.69%.")
    add_bullet(doc, "Về mặt kỹ thuật phần mềm: ", "Xây dựng hoàn chỉnh hệ thống Web ফুল-stack (FastAPI + React/TypeScript) có giao diện tiếng Việt trực quan, hỗ trợ tải video lên và xem ngay video đã xử lý kèm khung Bounding Box AI trực tiếp, trích xuất bằng chứng SHA-256 và xuất báo cáo CSV.")

    add_heading_styled(doc, "5.2. Hạn chế của đề tài", level=2)
    add_bullet(doc, "Điều kiện chiếu sáng cực đoan ban đêm: ", "Đối với các khoang lái tối hoàn toàn không có đèn hồng ngoại (IR illumination), việc nhận diện dây an toàn màu đen trên áo khoác đen vẫn cần định tuyến sang trạng thái NEEDS_REVIEW để con người xác minh.")
    add_bullet(doc, "Chuyển tiếp hành vi dưới 1 giây: ", "Các pha đổi tay cầm điện thoại diễn ra trong khoảng thời gian dưới 1 giây (như khoảng 0.91s tại gZ-36) có thể được gộp liền vào pha gọi điện kế tiếp.")

    add_heading_styled(doc, "5.3. Hướng phát triển trong tương lai", level=2)
    add_bullet(doc, "Tích hợp Camera Hồng ngoại (NIR / IR Camera): ", "Bổ sung dữ liệu huấn luyện phổ cận hồng ngoại để nâng cao độ chính xác nhận diện dây an toàn trong điều kiện ban đêm.")
    add_bullet(doc, "Mở rộng các hành vi mất an toàn khác: ", "Tích hợp thêm mô-đun phát hiện trạng thái buồn ngủ / ngáp (Drowsiness / Yawning) thông qua tỷ lệ mở mắt (EAR) và hướng nhìn của đầu (Head Pose Estimation).")
    add_bullet(doc, "Tối ưu hóa TensorRT / ONNX trên thiết bị biên (Edge AI): ", "Lượng tử hóa các mô hình sang định dạng FP16 TensorRT để triển khai trực tiếp trên các thiết bị biên lắp trên xe như NVIDIA Jetson Orin Nano.")

    doc.add_page_break()

    # ==================== TÀI LIỆU THAM KHẢO ====================
    add_heading_styled(doc, "TÀI LIỆU THAM KHẢO", level=1)
    refs = [
        "[1] Ortega, J. D., et al. (2020), 'DMD: A Large-Scale Multi-Modal Driver Monitoring Dataset for Attention and Alertness Analysis', Computer Vision – ECCV 2020 Workshops.",
        "[2] Jocher, G., Qiu, J., & Chaurasia, A. (2024), 'Ultralytics YOLO11: State-of-the-Art Real-Time Object Detection, Instance Segmentation, and Pose Estimation', Ultralytics Technical Documentation.",
        "[3] ASAM OpenLABEL Standard (2021), 'ASAM OpenLABEL: Concept and Multi-Sensor Scenario Labeling Format Specification v1.0.0', Association for Standardization of Automation and Measuring Systems.",
        "[4] Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017), 'On Calibration of Modern Neural Networks', Proceedings of the 34th International Conference on Machine Learning (ICML).",
        "[5] Chính phủ nước CHXHCN Việt Nam (2024), 'Nghị định 168/2024/NĐ-CP quy định xử phạt vi phạm hành chính về trật tự, an toàn giao thông trong lĩnh vực giao thông đường bộ'."
    ]
    for ref in refs:
        add_body(doc, ref)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output_path))


if __name__ == "__main__":
    out_main = Path(r"D:\.idea\giangdoantotnghiep\BAO_CAO_DO_AN_TOT_NGHIEP_ROADWATCH_V32.docx")
    build_thesis_document(out_main)
    out_copy = Path(r"D:\.idea\BAO_CAO_DO_AN_TOT_NGHIEP_ROADWATCH_V32.docx")
    shutil.copy2(out_main, out_copy)
    print(f"Generated thesis Word documents:\n - {out_main}\n - {out_copy}")
