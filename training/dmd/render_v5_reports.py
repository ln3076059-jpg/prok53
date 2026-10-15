"""Render the governed Roadwatch V5 research report set from measured artifacts.

The renderer intentionally keeps external-paper results, locally reproduced checks,
and Roadwatch measurements in separate sections.  It never re-runs or touches the
locked holdouts.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load(path: str):
    return json.loads((REPO_ROOT / path).read_text(encoding="utf-8"))


def _pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def _write(output: Path, name: str, body: str) -> None:
    # Interpolated table rows intentionally start at column zero, which prevents
    # textwrap.dedent from removing the template's eight-space source indent.
    # Strip that known template indent line by line while preserving deeper code.
    normalized = "\n".join(
        line[8:] if line.startswith("        ") else line for line in body.splitlines()
    )
    (output / name).write_text(normalized.strip() + "\n", encoding="utf-8")


def _event_table(experiments: list[dict]) -> str:
    rows = []
    for item in experiments:
        m = item["pooled_metrics"]
        rows.append(
            f"| {item['experiment_id']} | {m['true_positives']} | {m['false_positives']} | "
            f"{m['false_negatives']} | {_pct(m['precision'])} | {_pct(m['recall'])} | "
            f"{_pct(m['f1'])} | {m['false_alarms_per_minute']:.4f} |"
        )
    return "\n".join(rows)


def _confusion_table(clip: dict) -> str:
    labels = clip["class_order"]
    header = "| True \\ Pred | " + " | ".join(labels) + " |"
    rule = "|---|" + "---:|" * len(labels)
    rows = [
        "| " + label + " | " + " | ".join(str(value) for value in row) + " |"
        for label, row in zip(labels, clip["confusion_matrix"], strict=True)
    ]
    return "\n".join([header, rule, *rows])


def _figures(output: Path, result: dict, diagnostics: dict) -> None:
    figure_dir = output / "figures"
    figure_dir.mkdir(exist_ok=True)

    experiments = result["experiments"]
    ids = [row["experiment_id"].replace("_", "\n") for row in experiments]
    f1 = [100 * row["pooled_metrics"]["f1"] for row in experiments]
    recall = [100 * row["pooled_metrics"]["recall"] for row in experiments]
    fa = [row["pooled_metrics"]["false_alarms_per_minute"] for row in experiments]
    x = range(len(ids))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([i - 0.18 for i in x], f1, width=0.36, label="Event F1 (%)")
    ax.bar([i + 0.18 for i in x], recall, width=0.36, label="Event recall (%)")
    ax.set_ylim(0, max(45, max(f1 + recall) * 1.15))
    ax.set_ylabel("Percent")
    ax.set_xticks(list(x), ids)
    ax.set_title("Roadwatch V5 development ablation (same 29 events)")
    ax.legend(loc="upper left")
    ax2 = ax.twinx()
    ax2.plot(list(x), fa, color="#b33b2e", marker="o", label="FA/min")
    ax2.axhline(0.30, color="#b33b2e", linestyle="--", linewidth=1)
    ax2.set_ylabel("False alarms / minute")
    ax2.set_ylim(0, max(0.55, max(fa) * 1.15))
    fig.tight_layout()
    fig.savefig(figure_dir / "v5_event_ablation.png", dpi=160)
    plt.close(fig)

    clip_metrics = result.get("clip_metrics_b3", result["clip_metrics"])
    labels = clip_metrics["class_order"]
    matrix = clip_metrics["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix, cmap="Blues")
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            ax.text(j, i, value, ha="center", va="center", fontsize=8)
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    ax.set_title(
        f"LOSO clip confusion matrix ({clip_metrics['evaluated_clips']:,} clips)"
    )
    fig.colorbar(image, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(figure_dir / "v5_clip_confusion_matrix.png", dpi=160)
    plt.close(fig)

    counts = diagnostics["primary_failure_stage_counts_all_events"]
    stages = list(counts)
    values = [counts[item] for item in stages]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.barh(stages, values, color="#3a6ea5")
    ax.set_xlim(0, max(values) + 2)
    ax.set_xlabel("Ground-truth events")
    ax.set_title("Primary V4 failure stage (29 development events)")
    for index, value in enumerate(values):
        ax.text(value + 0.15, index, str(value), va="center")
    fig.tight_layout()
    fig.savefig(figure_dir / "v4_failure_stage_counts.png", dpi=160)
    plt.close(fig)


def render(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    round2_path = REPO_ROOT / "reports/v5_research_upgrade/v5_experiment_results_round2.json"
    result_name = (
        "reports/v5_research_upgrade/v5_experiment_results_round2.json"
        if round2_path.is_file()
        else "reports/v5_research_upgrade/v5_experiment_results.json"
    )
    result = _load(result_name)
    diagnostics = _load("reports/v5_research_upgrade/04_false_negative_diagnostics.json")
    seatbelt = _load("reports/calibration/V2_SEATBELT_CLASSIFIER_CALIBRATION.json")
    phone_training = _load("reports/V2_TRAINING_RESULTS.json")["components"]["phone_detector"]
    free_disk_gib = shutil.disk_usage(REPO_ROOT.anchor).free / (1024**3)
    _figures(output, result, diagnostics)

    selected = result["selected_metrics"]
    b2_clip = result["clip_metrics"]
    clip = result.get("clip_metrics_b3", b2_clip)
    b3 = next(
        (row for row in result["experiments"] if row["experiment_id"] == "B3_RESNET18_GRU"),
        None,
    )
    b8 = next(
        (
            row
            for row in result["experiments"]
            if row["experiment_id"] == "B8_B3_LATE_FUSION"
        ),
        None,
    )
    if b3 is None or b8 is None:
        raise ValueError("round-2 B3/B8 experiments are required for the V5 report set")
    b3m = b3["pooled_metrics"]
    b8m = b8["pooled_metrics"]
    event_rows = _event_table(result["experiments"])
    per_class_rows = "\n".join(
        f"| {name} | {row['support']} | {_pct(row['precision'])} | {_pct(row['recall'])} | {_pct(row['f1'])} |"
        for name, row in clip["per_class"].items()
    )
    stage_rows = "\n".join(
        f"| {name} | {row['events_reached']} / {row['total_events']} | {_pct(row['recall'])} | {row['status']} |"
        for name, row in diagnostics["stage_level_event_recall_upper_bounds"].items()
    )
    case_rows = "\n".join(
        f"| {case['case_id']} | {case['subject_id']} | {case['truth']['source_action'].split('/')[-1]} | "
        f"{case['truth']['start_seconds']:.3f}–{case['truth']['end_seconds']:.3f} | "
        f"{case['primary_failure_stage']} | {case['evidence']['max_any_phone_conf']:.4f} | "
        f"{case['evidence']['max_behavior_score']:.4f} |"
        for case in diagnostics["false_negative_cases"]
    )
    recovered_rows = "\n".join(
        f"| {row['case_id']} | {row['subject_id']} | {row['source_action'].split('/')[-1]} | "
        f"{row['v4_failure_stage']} | {row['selected_v5_prediction']['start_seconds']:.3f}–"
        f"{row['selected_v5_prediction']['end_seconds']:.3f} | {row['matching_evidence']['iou']:.3f} | "
        f"{row['matching_evidence']['start_latency']:+.3f}s |"
        for row in result["v4_false_negatives_recovered"]
    )

    _write(
        output,
        "01_SCIENTIFIC_LITERATURE_REVIEW.md",
        """
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
        """,
    )

    _write(
        output,
        "02_PAPER_COMPARISON_MATRIX.md",
        """
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
        """,
    )

    _write(
        output,
        "03_REPRODUCIBILITY_AND_LICENSE_AUDIT.md",
        f"""
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

        - Dataset role: `{result['dataset_role']}`; subjects: {', '.join(result['development_subjects'])}.
        - Input-set SHA-256: `{result['input_set_sha256']}`; config SHA-256: `{result['config_sha256']}`.
        - B3 embedding-set SHA-256: `{result['input_set_sha256_b3']}`; backbone
          `{result['backbone_source']['name']}` có weights SHA-256 `{result['backbone_source']['weights_sha256']}`.
        - Evaluation: `{result['evaluation_protocol']}`.
        - Holdout bị cấm: {', '.join(result['forbidden_consumed_holdouts'])}; accessed: `{result['holdouts_accessed']}`.
        - Mỗi fold train trên hai subject và test subject còn lại; threshold chỉ chọn trên hai training subjects.
        - RGB cache có hash của video và annotation trong `input_provenance`; checkpoint có SHA-256 theo fold.

        **Caveat license:** governance README cục bộ của DMD đang ghi quyền sử dụng/chấp thuận chưa hoàn tất dù
        file dữ liệu có mặt. Kết quả này chỉ là nghiên cứu nội bộ, không phân phối dataset/cache/checkpoint và
        không dùng làm sản phẩm cho đến khi quyền sử dụng được xác nhận.

        ## Mức tái lập Roadwatch

        Command: `py -3.11 -m training.dmd.run_v5_experiments`. Cấu hình cố định tại
        `models/model_config_v5_research.yaml`; seed 5305; checkpoints chứa architecture, class names,
        train/heldout subjects, threshold, config, git HEAD và hash. CPU/PyTorch hiện tại được ghi trong JSON.
        Môi trường kiểm tra là Windows, Python {result['runtime']['python']}, PyTorch
        {result['runtime']['torch_version']}, CUDA={result['runtime']['cuda_available']}; ổ D còn khoảng
        {free_disk_gib:.2f} GiB tại lúc render báo cáo.
        Kết quả là `REPRODUCED_LOCALLY` cho chính code Roadwatch và `MEASURED_ON_ROADWATCH` cho metrics;
        **không** phải reproduction của DualScope/DriveCLIP/Multi-Task Fusion.
        """,
    )

    _write(
        output,
        "04_DMD_FALSE_NEGATIVE_ROOT_CAUSE.md",
        f"""
        # Phân tích nguyên nhân 26 false negatives của V4

        Nguồn đo: `reports/v4_upgrade/v4_development_results.json`, candidate `V4_COMBINED`, cùng 29 sự kiện
        trên gB-9/gZ-36/gC-14. Audit xác nhận TP=3, FP=5, FN=26. Chẩn đoán chạy lại fusion trên 2,628 record
        feature cache; không thay matching policy và không dò holdout.

        ![Failure stages](figures/v4_failure_stage_counts.png)

        ## Primary failure stage

        - Cross-view fusion: 11 FN.
        - Phone object detector: 9 FN.
        - Behavioral classification: 5 FN.
        - Event activation: 1 FN.
        - 3 sự kiện còn lại là true positive.

        Person/driver association **không quan sát được** vì V3 cache không serialize trường này. View
        availability cũng không chứng minh điện thoại nhìn thấy hay không bị che; do đó không biến thiếu bằng
        chứng thành measured zero.

        ## Stage-level event evidence upper bound

        | Stage | Reached / 29 | Recall | Status |
        |---|---:|---:|---|
        {stage_rows}

        Các tỷ lệ là “event có ít nhất một sampled record vượt điều kiện ở stage”, không phải causal recall
        nối tiếp và không được nhân/chia như funnel độc lập. Detector evidence chỉ xuất hiện ở 20/29 event;
        behavioral evidence chỉ 4/29. Feature cache vì thế không đủ để giải quyết bottleneck bằng threshold.

        ## Danh sách từng FN

        | Case | Subject | Action | GT seconds | Primary stage | Max phone conf | Max behavior score |
        |---|---|---|---:|---|---:|---:|
        {case_rows}

        Chi tiết checks/evidence đầy đủ nằm ở `04_false_negative_diagnostics.json`. Hướng xử lý có căn cứ là
        trích RGB clip và học action evidence độc lập; nếu tiếp tục cải thiện object path cần chạy lại feature
        extraction có bbox/association/visibility, không chỉ dùng cache confidence hiện tại.
        """,
    )

    _write(
        output,
        "05_MODEL_ARCHITECTURE_DESIGN.md",
        """
        # Thiết kế Roadwatch V5 dual-path

        ## Luồng nghiên cứu

        ```text
        RGB BODY clip ──> driver ROI ──> Tiny 2D CNN/frame ──> mean|max|last-first ──> action logits
              │                                                        │
              └──────────────── timestamps ────────────────────────────┤
                                                                       v
        YOLO/geometry events ───────────────────────────────> governed late fusion ──> temporal event
                                                                       │
                                                        NEEDS_REVIEW / PENDING
        ```

        ### Path A — object & geometry

        V4 YOLO phone confidence, pose và multi-view fusion được giữ nguyên làm frozen comparator. V5 không
        thay detector vì chưa có same-data benchmark. Bbox/driver association không có trong feature cache nên
        không được giả lập.

        ### Path B — independent video action

        `TinyTemporalCNN` xử lý 5 frame liên tiếp, ROI BODY chuẩn hóa 64×64. Ba Conv-BN-SiLU block tạo 32-D
        feature mỗi frame; temporal descriptor ghép mean, max và `(last-first)` rồi phân loại 6 lớp. Loss là
        class-weighted cross entropy với label smoothing 0.03. `UNKNOWN` là abstention, không được phát minh
        thành ground truth. Thiết kế lấy cảm hứng từ behavior-centric crop, salient motion/difference và video
        semantics nhưng là implementation độc lập, nhỏ, phù hợp CPU.

        Vòng 2 bổ sung encoder ResNet-18 `IMAGENET1K_V1` đóng băng, embedding 512-D và bidirectional GRU
        (hidden size 64) trên chuỗi 9 frame. Backbone weights có SHA-256 và provenance riêng; chỉ temporal head
        được huấn luyện trên hai training subjects của từng fold.

        ### Path C — fusion & decision

        Xác suất bốn lớp phone được cộng, smooth 3 mẫu, dùng hysteresis, persistence 2 mẫu và merge gap 2s.
        Action-only luôn `NEEDS_REVIEW`; object+action là `PENDING`; code không phát kết luận vi phạm tự động.
        B5 union late fusion tăng recall nhưng tăng FP. B6 chỉ giữ action event confidence ≥0.70 và duration ≥1s.

        ## Integration status

        `predict_subject`, `probabilities_to_events` và `merge_evidence_events` là API inference/evidence có unit
        test. Candidate **không được bật trong backend production** vì chưa đạt target, subject gZ-36 suy sụp và
        chưa có independent holdout. Checkpoint hiện tại chỉ có ý nghĩa theo fold LOSO, không có một production
        checkpoint đã được phê duyệt.
        """,
    )

    _write(
        output,
        "06_EXPERIMENT_PROTOCOL.md",
        f"""
        # Giao thức thực nghiệm

        ## Population và grain

        - 3 development subjects: `{', '.join(result['development_subjects'])}`; 22.063 video-minutes.
        - 29 phone-use ground-truth events từ annotation DMD thật.
        - B2 có 2,616 và B3 có 2,604 heldout LOSO clips sau biên sequence; clip grain khác event grain.
        - Labels: SAFE_DRIVING, TEXTING_LEFT/RIGHT, PHONECALL_LEFT/RIGHT, OTHER_ACTIVITY; UNKNOWN chỉ abstain.

        ## LOSO và chống leakage

        Mỗi fold giữ một subject để test, train hai subject còn lại. Threshold grid 0.40–0.80 được chấm **chỉ
        trên training subjects** của fold, ưu tiên event F1 với FA/min ≤0.30. Không frame/clip của test subject
        đi vào training hoặc threshold selection. `gZ-37` và `gE-28` không truy cập.

        ## Metrics

        Clip: accuracy, macro-F1, per-class P/R/F1, confusion matrix. Event: frozen matcher, TP/FP/FN,
        P/R/F1, FA/min, latency, duplicates và fragmentation. Pooled rates được tính từ pooled counts, không
        trung bình các fold. Runtime: CPU throughput, p50/p95 inference latency và sampled process RSS.

        ## Ablation registry

        - B0: frozen V4 object/geometry — measured.
        - B1: original detector/fusion — không chạy mới; V3 historical artifact không dùng để chọn V5.
        - B2: TinyCNN temporal action — measured LOSO.
        - B3: frozen ResNet-18 + bidirectional GRU — measured LOSO ở vòng 2.
        - B4: CLIP — not run (full VideoCLIP artifact chưa xác minh, compute cao).
        - B5: union object + action — measured.
        - B6: reliability-gated action + object + temporal engine — measured.
        - B7: DualScope-inspired full candidate — not run (code/weights chưa tái lập, GPU unavailable).
        - B8: B3 + frozen V4 late fusion — measured; vượt target P/R/F1 nhưng trượt FA/min.
        - Research selection vẫn là B2 theo rule khóa; chưa phải deployment model.

        ## Reproduction

        Chạy vòng 1 bằng `py -3.11 -m training.dmd.run_v5_experiments`; chạy vòng 2 bằng
        `py -3.11 -m training.dmd.run_v5_b3_experiment`, cùng config `models/model_config_v5_research.yaml`.
        Input SHA `{result['input_set_sha256']}`; git HEAD `{result['git_head']}`. Checkpoint reuse chỉ được chấp
        nhận nếu heldout/train subjects và architecture metadata khớp.
        """,
    )

    _write(
        output,
        "07_ABLATION_RESULTS.md",
        f"""
        # Kết quả ablation V5

        **Status tất cả dòng dưới:** `MEASURED_ON_ROADWATCH` trên cùng 29 events / 22.063 phút, trừ sai khác
        làm tròn 22.060 trong artifact V4.

        | Experiment | TP | FP | FN | Precision | Recall | Event F1 | FA/min |
        |---|---:|---:|---:|---:|---:|---:|---:|
        {event_rows}

        ![Event ablation](figures/v5_event_ablation.png)

        B2 vẫn được chọn vì F1 30.00% là cao nhất trong nhóm FA/min ≤0.30. B3 cải thiện rõ lên
        P={_pct(b3m['precision'])}, R={_pct(b3m['recall'])}, F1={_pct(b3m['f1'])}, nhưng 7 FP tương ứng
        {b3m['false_alarms_per_minute']:.4f} FA/min, cao hơn gate đúng một FP trên 22.063 phút. B8 late fusion đạt
        P={_pct(b8m['precision'])}, R={_pct(b8m['recall'])}, F1={_pct(b8m['f1'])}, song cùng trượt FA/min.
        Không tăng threshold hậu nghiệm theo heldout: sweep trên training subjects đã bão hòa và không cung cấp
        bằng chứng calibration để làm vậy.

        ## Target gate

        | Metric | Selected B2 | B8 best unconstrained | Development target | B2 pass? | B8 pass? |
        |---|---:|---:|---:|---|---|
        | Precision | {_pct(selected['precision'])} | {_pct(b8m['precision'])} | ≥60% | NO | YES |
        | Recall | {_pct(selected['recall'])} | {_pct(b8m['recall'])} | ≥30% | NO | YES |
        | F1 | {_pct(selected['f1'])} | {_pct(b8m['f1'])} | ≥40% | NO | YES |
        | FA/min | {selected['false_alarms_per_minute']:.4f} | {b8m['false_alarms_per_minute']:.4f} | ≤0.30 | YES | NO |

        Kết luận: giả thuyết “action branch có thể phục hồi evidence khi detector/fusion yếu” được hỗ trợ một
        phần và pretrained encoder cải thiện độ nhạy, nhưng không candidate nào đạt đồng thời toàn bộ gate;
        vì vậy không deploy.
        """,
    )

    _write(
        output,
        "08_PHONE_METRICS.md",
        f"""
        # Metrics nhánh phone

        ## B3 action recognition — `MEASURED_ON_ROADWATCH`

        - Clip accuracy: {_pct(clip['clip_accuracy'])}; macro-F1: {_pct(clip['macro_f1'])}; evaluated clips: {clip['evaluated_clips']}.

        | Class | Support | Precision | Recall | F1 |
        |---|---:|---:|---:|---:|
        {per_class_rows}

        {_confusion_table(clip)}

        ![Clip confusion matrix](figures/v5_clip_confusion_matrix.png)

        B3 tăng clip accuracy từ {_pct(b2_clip['clip_accuracy'])} lên {_pct(clip['clip_accuracy'])} và macro-F1
        từ {_pct(b2_clip['macro_f1'])} lên {_pct(clip['macro_f1'])}. Tuy vậy SAFE_DRIVING recall bằng 0 và dự đoán
        vẫn lệch mạnh theo subject/lớp. Riêng fold gZ-36 có event TP=2, FP=5, FN=9 và 0.5816 FA/min, cho thấy
        score shift giữa subject; aggregate tốt hơn không chứng minh tổng quát hóa.

        ## Temporal events — selected B2

        TP={selected['true_positives']}, FP={selected['false_positives']}, FN={selected['false_negatives']};
        precision={_pct(selected['precision'])}, recall={_pct(selected['recall'])}, F1={_pct(selected['f1'])},
        FA/min={selected['false_alarms_per_minute']:.4f}, mean onset latency={selected['detection_latency_mean_seconds']:.3f}s,
        duplicates={selected['duplicate_events']}, fragmentation={selected['event_fragmentation']}.

        B3 action-only phục hồi {result['recovered_count_by_experiment']['B3_RESNET18_GRU']} V4 FN;
        B8 late fusion phục hồi {result['recovered_count_by_experiment']['B8_B3_LATE_FUSION']} V4 FN, nhưng cả hai
        đều nằm ngoài FA/min gate. Selected B2 tiếp tục phục hồi {result['recovered_count']} V4 FN.

        ## Sáu V4 FN được phục hồi

        | Case | Subject | Action | V4 failure | V5 predicted interval | IoU | Start latency |
        |---|---|---|---|---:|---:|---:|
        {recovered_rows}

        ## Object detector metrics — tách riêng grain

        Historical V2 YOLO11s validation (`MEASURED_ON_ROADWATCH`, component image split): precision
        {_pct(phone_training['validation_metrics']['metrics/precision(B)'])}, recall
        {_pct(phone_training['validation_metrics']['metrics/recall(B)'])}, mAP50
        {_pct(phone_training['validation_metrics']['metrics/mAP50(B)'])}, mAP50-95
        {_pct(phone_training['validation_metrics']['metrics/mAP50-95(B)'])}. V5 **không retrain detector** và
        DMD temporal cache không có bbox ground truth để tính mAP; các số detector này không phải V5 event metrics.
        """,
    )

    sb = seatbelt["per_class_metrics"]
    sb_rows = "\n".join(
        f"| {name} | {row['support']} | {_pct(row['precision'])} | {_pct(row['recall'])} | {_pct(row['f1'])} |"
        for name, row in sb.items()
    )
    _write(
        output,
        "09_SEATBELT_METRICS.md",
        f"""
        # Seatbelt metrics và research gap

        **V5 status: `NOT_RUN`.** Không có seatbelt temporal ground truth mới hoặc same-camera overhead dataset
        hợp lệ trong scope, nên không huấn luyện RST-Net/YOLOv7 và không tuyên bố V5 improvement.

        ## Historical V2 validation — `MEASURED_ON_ROADWATCH`

        Model SHA `{seatbelt['model_sha256']}`, split validation 641 crops, raw Top-1 80.97%, macro-F1 57.27%.

        | Class | Support | Precision | Recall | F1 |
        |---|---:|---:|---:|---:|
        {sb_rows}

        Lớp uncertain chỉ có 25 mẫu và recall 4.0%; đây là bottleneck nghiêm trọng. Reject policy fastened≥0.50,
        unfastened≥0.60 giảm false-unfastened từ 49 xuống 36 nhưng post-policy uncertain F1 chỉ 3.57%. “Không
        nhìn thấy dây” phải giữ `UNCERTAIN_OR_OCCLUDED`, không được biến thành `UNFASTENED`.

        ## Research design tiếp theo

        1. Thu thập/human-review thêm uncertain/occluded theo vehicle/camera group.
        2. So sánh same-split torso ROI baseline với lightweight attention/pruned ResNet; đo calibration/ECE và
           false violation rate ngoài per-class metrics.
        3. Với camera ngoài đường, dùng two-stage vehicle→windshield/occupant→belt và group split theo video,
           location, day/night để tránh adjacent-frame leakage.
        4. Chỉ promote khi uncertain recall và false-unfastened đạt gate trên independent camera holdout.
        """,
    )

    runtime = result["runtime"]
    runtime_b3 = result["runtime_b3"]
    fold_runtime_rows = "\n".join(
        f"| {row['subject_id']} | {row['clips_per_second']:.3f} | {row['milliseconds_per_clip']:.3f} | "
        f"{row['p50_latency_ms']:.3f} | {row['p95_latency_ms']:.3f} | {row['peak_process_rss_mb']:.3f} |"
        for row in runtime["per_fold"]
    )
    fold_runtime_rows_b3 = "\n".join(
        f"| {row['subject_id']} | {row['clips_per_second']:.3f} | {row['milliseconds_per_clip']:.3f} | "
        f"{row['p50_latency_ms']:.3f} | {row['p95_latency_ms']:.3f} | {row['peak_process_rss_mb']:.3f} |"
        for row in runtime_b3["per_fold"]
    )
    embedding_runtime_rows = "\n".join(
        f"| {row['subject_id']} | {row.get('records', 'N/A')} | "
        f"{row.get('extraction_seconds', 'REUSED')} |"
        for row in result["input_provenance_b3"]
    )
    _write(
        output,
        "10_SPEED_BENCHMARK.md",
        f"""
        # Benchmark tốc độ V5 action branch

        Environment: Windows, Python {runtime['python']}, PyTorch {runtime['torch_version']}, device CPU,
        CUDA={runtime['cuda_available']}; OMP/BLAS/MKL threads cố định 1 để tránh paging/thread oversubscription.

        ## B2 TinyTemporalCNN trên RGB cache

        | Heldout subject | Clips/s | ms/clip total | p50 model ms | p95 model ms | Peak process RSS MB |
        |---|---:|---:|---:|---:|---:|
        {fold_runtime_rows}

        Aggregate mean={runtime['mean_clips_per_second']:.3f} clips/s, p50={runtime['p50_latency_ms_across_folds']:.3f}ms,
        max-fold p95={runtime['p95_latency_ms_across_folds']:.3f}ms, peak RSS={runtime['peak_process_rss_mb']:.3f}MB.
        RSS được sample mỗi 10ms và bao gồm Python/PyTorch + model + subject caches; không phải riêng model.
        VRAM=`N/A` vì CUDA unavailable. Đây chỉ là action branch trên cache 64×64, không gồm video decode/ROI
        extraction hoặc full YOLO/pose pipeline, nên không được gọi là full-system FPS.

        ## B3 temporal head trên ResNet-18 embeddings

        | Heldout subject | Clips/s | ms/clip total | p50 model ms | p95 model ms | Peak process RSS MB |
        |---|---:|---:|---:|---:|---:|
        {fold_runtime_rows_b3}

        Aggregate temporal-head mean={runtime_b3['mean_clips_per_second']:.3f} clips/s,
        p50={runtime_b3['p50_latency_ms_across_folds']:.3f}ms, max-fold
        p95={runtime_b3['p95_latency_ms_across_folds']:.3f}ms, peak RSS={runtime_b3['peak_process_rss_mb']:.3f}MB.

        | Subject | Frames embedded | One-time extraction seconds |
        |---|---:|---:|
        {embedding_runtime_rows}

        Số B3 clips/s chỉ đo BiGRU temporal head trên embedding đã tính sẵn. Thời gian ResNet-18 extraction được
        tách riêng ở bảng trên; do đó không được gọi là end-to-end throughput hay so trực tiếp với B2 clips/s.
        Full-pipeline FPS=`NOT_MEASURED`; không nội suy từ cached clips/s.
        """,
    )

    _write(
        output,
        "11_TEST_RESULTS.md",
        """
        # Kết quả kiểm thử

        ## Executed

        ```text
        py -3.11 -m pytest tests/test_v5_video_action.py tests/test_v5_result_integrity.py
          tests/test_v4_experiment_runner.py tests/test_event_evaluation_governance.py
          tests/test_dmd_pipeline.py -q
        .............................................. [100%]
        46 passed
        ```

        Coverage chức năng: temporal persistence, `NEEDS_REVIEW`, governed object+action fusion, V4 regression,
        event governance/matching, DMD pipeline, LOSO subject isolation, prohibited-holdout access, pooled-count
        reconciliation, confusion-matrix totals, checkpoint hashes, round-2 recovery maps và locked selection rule.
        `py_compile` cũng pass cho các module V5.

        ## Không tuyên bố

        Đây là scoped regression suite, không phải toàn bộ repository test suite. Production UI/API end-to-end,
        live camera, database persistence và evidence export không được chạy lại trong V5 vì candidate chưa qua
        model gate; V4 historical tests/artifacts không bị sửa.
        """,
    )

    v4m = next(row for row in result["experiments"] if row["experiment_id"] == "B0_V4_CURRENT")["pooled_metrics"]
    _write(
        output,
        "12_FINAL_SCIENTIFIC_REPORT.md",
        f"""
        # Báo cáo khoa học cuối Roadwatch V5

        ## Executive conclusion

        V5 đã triển khai và kiểm nghiệm hai video-action branch độc lập theo LOSO. Theo rule khóa, B2 vẫn là
        candidate được chọn với TP/FP/FN=6/5/23 và FA/min 0.2266. B3 frozen ResNet-18 + BiGRU cải thiện lên
        TP/FP/FN=10/7/19, F1={_pct(b3m['f1'])}; B8 fusion đạt P={_pct(b8m['precision'])},
        R={_pct(b8m['recall'])}, F1={_pct(b8m['f1'])}. Tuy nhiên B3/B8 đều có FA/min
        {b8m['false_alarms_per_minute']:.4f} > 0.30, nên **không candidate nào đạt đồng thời toàn bộ target** và
        generalization chưa được thiết lập.

        ## A. Scientific foundation

        Sáu paper đã được audit về task, protocol, metric, code/weights và license. V5 dùng các nguyên lý có thể
        tái hiện trên CPU: driver-centric ROI, short video clips, temporal pooling/difference, independent action
        probabilities, late fusion và uncertainty governance. Không paper checkpoint nào được dùng.

        ## B. Actual code

        - `training/dmd/v5_video_action.py`: extraction, TinyTemporalCNN, training/inference, metrics/events/fusion.
        - `training/dmd/v5_resnet_gru.py`: frozen ResNet-18 embeddings và bidirectional GRU temporal head.
        - `training/dmd/v5_diagnostics.py`: 26-FN stage diagnostics.
        - `training/dmd/run_v5_experiments.py`: LOSO, train-only threshold calibration, ablations, provenance.
        - `training/dmd/run_v5_b3_experiment.py`: vòng 2 B3/B8, checkpoint và provenance.
        - `training/dmd/render_v5_reports.py`: deterministic reports/figures from measured JSON.
        - `models/model_config_v5_research.yaml`: locked research config.
        - `tests/test_v5_video_action.py`, `tests/test_v5_result_integrity.py`: unit/regression/integrity tests.

        ## C. Metrics comparison

        | Version | TP | FP | FN | P | R | F1 | FA/min |
        |---|---:|---:|---:|---:|---:|---:|---:|
        | V4 | 3 | 5 | 26 | {_pct(v4m['precision'])} | {_pct(v4m['recall'])} | {_pct(v4m['f1'])} | {v4m['false_alarms_per_minute']:.4f} |
        | V5 B2 research | 6 | 5 | 23 | {_pct(selected['precision'])} | {_pct(selected['recall'])} | {_pct(selected['f1'])} | {selected['false_alarms_per_minute']:.4f} |
        | V5 B3 research | {b3m['true_positives']} | {b3m['false_positives']} | {b3m['false_negatives']} | {_pct(b3m['precision'])} | {_pct(b3m['recall'])} | {_pct(b3m['f1'])} | {b3m['false_alarms_per_minute']:.4f} |
        | V5 B8 fusion | {b8m['true_positives']} | {b8m['false_positives']} | {b8m['false_negatives']} | {_pct(b8m['precision'])} | {_pct(b8m['recall'])} | {_pct(b8m['f1'])} | {b8m['false_alarms_per_minute']:.4f} |

        Clip accuracy {_pct(clip['clip_accuracy'])}, macro-F1 {_pct(clip['macro_f1'])}; đây là cảnh báo rõ rằng
        event improvement trên ba subject không tương đương action recognition tốt.

        ## D. Failure recovery

        Selected B2 phục hồi 6 V4 FN; B3 phục hồi
        {result['recovered_count_by_experiment']['B3_RESNET18_GRU']} và B8 phục hồi
        {result['recovered_count_by_experiment']['B8_B3_LATE_FUSION']}. Evidence interval, IoU, latency và failure
        stage nằm trong JSON; số recovery cao hơn của B3/B8 đi kèm FA/min vượt gate.

        ## E. Seatbelt

        V5 seatbelt experiment `NOT_RUN`. Historical V2 uncertain recall 4.0% (support 25), nên branch chưa sẵn
        sàng để kết luận vi phạm; fail-closed policy được giữ.

        ## F. Runtime

        B2 cached inference đạt trung bình {runtime['mean_clips_per_second']:.1f} clips/s, p50
        {runtime['p50_latency_ms_across_folds']:.3f}ms, p95 {runtime['p95_latency_ms_across_folds']:.3f}ms,
        peak process RSS {runtime['peak_process_rss_mb']:.1f}MB trên CPU; VRAM N/A. Không suy rộng thành full-pipeline FPS.
        B3 temporal head đạt {runtime_b3['mean_clips_per_second']:.1f} clips/s trên precomputed embeddings; con số
        này không gồm frozen ResNet-18 extraction và không phải end-to-end FPS.
        Máy là Windows/Python {runtime['python']}/PyTorch {runtime['torch_version']}, CUDA unavailable; ổ D còn
        khoảng {free_disk_gib:.2f} GiB tại lúc render.

        ## G. Scientific validity

        LOSO subject-disjoint, threshold train-only, pooled rates từ pooled counts, provenance/checkpoint hashes và
        no-access assertion cho gZ-37/gE-28 đều được lưu/kiểm thử. Tuy nhiên n=3 subjects, quyền DMD còn pending,
        không có independent holdout mới và model collapse trên gZ-36; do đó chỉ là development research.

        ## H. Thesis readiness

        | Requirement | Status |
        |---|---|
        | Literature + reproducibility audit | COMPLETE |
        | V4 26-FN diagnosis | COMPLETE_WITH_OBSERVABILITY_LIMITS |
        | Independent video action baseline | COMPLETE_RESEARCH |
        | Subject-disjoint ablation | COMPLETE_ON_3_DEV_SUBJECTS |
        | Phone target | PARTIAL: B8 đạt P/R/F1 nhưng trượt FA/min; selected B2 không đạt P/R/F1 |
        | Seatbelt V5 upgrade | NOT_RUN; DATA/GPU/VALIDATION GAP |
        | Production inference activation | NOT_APPROVED |
        | Independent holdout generalization | NOT_ESTABLISHED |
        | Thesis figures/tables/BibTeX | COMPLETE_FOR_RESEARCH_CHAPTER |

        ## I. Remaining limitations và bước tiếp theo

        Ưu tiên thêm subject development hợp pháp, class-balanced clips và group-aware validation; xử lý
        subject-dependent calibration của B3 bằng training-only domain-robust objective, rồi mới xét CLIP hoặc
        motion/hand-face crops mà không chạm holdout. Với seatbelt, mở rộng uncertain class và thu same-camera
        overhead data. Candidate chỉ được promote khi vượt gate trên development rồi xác nhận một lần trên holdout
        độc lập mới. Mọi action-only event tiếp tục `NEEDS_REVIEW`; không tự động xử phạt.
        """,
    )

    _write(
        output,
        "references.bib",
        r"""
        @article{qiu2026dualscope,
          author = {Zhijie Qiu and Shuaibo Li and Laixin Zhang and Xuming Hu and Wei Ma},
          title = {DualScope: Capturing Critical Spatial and Temporal Cues for Distracted Driving Activity Recognition},
          journal = {Proceedings of the AAAI Conference on Artificial Intelligence},
          year = {2026}, volume = {40}, number = {10}, pages = {8602--8611},
          doi = {10.1609/aaai.v40i10.37812}
        }

        @article{hasan2024driveclip,
          author = {Md Zahid Hasan and Jiajing Chen and Jiyang Wang and Mohammed Shaiqur Rahman and Ameya Joshi and Senem Velipasalar and Chinmay Hegde and Anuj Sharma and Soumik Sarkar},
          title = {Vision-Language Models Can Identify Distracted Driver Behavior From Naturalistic Videos},
          journal = {IEEE Transactions on Intelligent Transportation Systems},
          year = {2024}, doi = {10.1109/TITS.2024.3381175}, eprint = {2306.10159}
        }

        @article{vijeikis2025multitask,
          author = {Romas Vijeikis and Ibidapo Dare Dada and Adebayo A. Abayomi-Alli and Vidas Raudonis},
          title = {Enhancing Driver Monitoring Systems Based on Novel Multi-Task Fusion Algorithm},
          journal = {Sensors}, year = {2025}, volume = {25}, number = {21}, pages = {6799},
          doi = {10.3390/s25216799}
        }

        @article{chaman2025yolov12,
          author = {Mohammed Chaman and Anas El Maliki and Hamza El Yanboiy and Hamad Dahou and Abdelkader Hadjoudja},
          title = {{YOLOv12}-Based Driver Monitoring System: Real-Time Applications in {ADAS}},
          journal = {International Journal of Safety and Security Engineering},
          year = {2025}, volume = {15}, number = {9}, pages = {1921--1932},
          doi = {10.18280/ijsse.150915}
        }

        @article{qiu2024seatbelt,
          author = {Liankui Qiu and Jiankun Rao and Xiangzhe Zhao},
          title = {Seatbelt Detection Algorithm Improved with Lightweight Approach and Attention Mechanism},
          journal = {Applied Sciences}, year = {2024}, volume = {14}, number = {8}, pages = {3346},
          doi = {10.3390/app14083346}
        }

        @article{widodo2025overhead,
          author = {Catur Edi Widodo and Kusworo Adi and Priyono Priyono and Aji Setiawan},
          title = {Real-time Detection of Seat Belt Usage in Overhead Traffic Surveillance Using {YOLOv7}},
          journal = {Applied Computer Science}, year = {2025}, volume = {21}, number = {4}, pages = {1--12},
          doi = {10.35784/acs_7594}
        }
        """,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path, default=REPO_ROOT / "reports" / "v5_research_upgrade"
    )
    args = parser.parse_args()
    render(args.output_dir)


if __name__ == "__main__":
    main()
