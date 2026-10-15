# Báo cáo khoa học cuối Roadwatch V5

## Executive conclusion

V5 đã triển khai và kiểm nghiệm hai video-action branch độc lập theo LOSO. Theo rule khóa, B2 vẫn là
candidate được chọn với TP/FP/FN=6/5/23 và FA/min 0.2266. B3 frozen ResNet-18 + BiGRU cải thiện lên
TP/FP/FN=10/7/19, F1=43.48%; B8 fusion đạt P=61.11%,
R=37.93%, F1=46.81%. Tuy nhiên B3/B8 đều có FA/min
0.3173 > 0.30, nên **không candidate nào đạt đồng thời toàn bộ target** và
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
| V4 | 3 | 5 | 26 | 37.50% | 10.30% | 16.20% | 0.2270 |
| V5 B2 research | 6 | 5 | 23 | 54.55% | 20.69% | 30.00% | 0.2266 |
| V5 B3 research | 10 | 7 | 19 | 58.82% | 34.48% | 43.48% | 0.3173 |
| V5 B8 fusion | 11 | 7 | 18 | 61.11% | 37.93% | 46.81% | 0.3173 |

Clip accuracy 35.87%, macro-F1 31.41%; đây là cảnh báo rõ rằng
event improvement trên ba subject không tương đương action recognition tốt.

## D. Failure recovery

Selected B2 phục hồi 6 V4 FN; B3 phục hồi
8 và B8 phục hồi
8. Evidence interval, IoU, latency và failure
stage nằm trong JSON; số recovery cao hơn của B3/B8 đi kèm FA/min vượt gate.

## E. Seatbelt

V5 seatbelt experiment `NOT_RUN`. Historical V2 uncertain recall 4.0% (support 25), nên branch chưa sẵn
sàng để kết luận vi phạm; fail-closed policy được giữ.

## F. Runtime

B2 cached inference đạt trung bình 1567.2 clips/s, p50
0.367ms, p95 0.535ms,
peak process RSS 274.6MB trên CPU; VRAM N/A. Không suy rộng thành full-pipeline FPS.
B3 temporal head đạt 5419.5 clips/s trên precomputed embeddings; con số
này không gồm frozen ResNet-18 extraction và không phải end-to-end FPS.
Máy là Windows/Python 3.11.9/PyTorch 2.13.0+cpu, CUDA unavailable; ổ D còn
khoảng 10.13 GiB tại lúc render.

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
