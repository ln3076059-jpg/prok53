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
