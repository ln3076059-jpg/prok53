# V9 frozen phone-context event replay

This replay uses the frozen temporal config and matching policy. It changes only `phone_context`: the
naive variant treats every positive detection as handheld use; the aware variant preserves stored
`HANDHELD_USE`, `MOUNTED_OR_STATIC`, and `UNKNOWN` values.

| Subject | Variant | TP | FP | FN | Precision | Recall | F1 | FA/min |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gC-14 | context_naive | 3 | 4 | 6 | 42.86% | 33.33% | 37.50% | 0.5990 |
| gC-14 | context_aware | 3 | 3 | 6 | 50.00% | 33.33% | 40.00% | 0.4500 |
| gZ-36 | context_naive | 3 | 0 | 8 | 100.00% | 27.27% | 42.86% | 0.0000 |
| gZ-36 | context_aware | 3 | 0 | 8 | 100.00% | 27.27% | 42.86% | 0.0000 |
| **Aggregate** | **context_naive** | **6** | **4** | **14** | **60.00%** | **30.00%** | **40.00%** | **0.2620** |
| **Aggregate** | **context_aware** | **6** | **3** | **14** | **66.67%** | **30.00%** | **41.38%** | **0.1965** |

Across 20 ground-truth events, context gating keeps TP/FN unchanged at 6/14 and removes one FP (4 to 3).
Precision changes by +6.67 percentage points, F1 by
+1.38 points, and FA/min by -0.0655.

The improvement occurs on `gC-14`; `gZ-36` has no stored mounted/static observations and is unchanged.
This result supports negative context evidence but remains a two-subject development replay. `gB-9` has
no eligible cache, so this is not comparable to the 29-event controlled V9 baseline. It does not establish
independent generalization or target attainment. No threshold, ground truth, matcher, or model changed,
and protected holdouts were not accessed.
