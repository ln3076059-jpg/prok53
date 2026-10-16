# All false positives including duplicates

| Subject | Kind | Start | End | Sampled true classes |
| --- | --- | --- | --- | --- |
| gZ-36 | UNMATCHED_PREDICTION | 25.202 | 35.786 | {"OTHER_ACTIVITY": 20} |
| gZ-36 | UNMATCHED_PREDICTION | 272.177 | 280.242 | {"OTHER_ACTIVITY": 17} |
| gZ-36 | UNMATCHED_PREDICTION | 392.641 | 395.161 | {"SAFE_DRIVING": 5} |
| gZ-36 | UNMATCHED_PREDICTION | 503.024 | 507.056 | {"OTHER_ACTIVITY": 8} |
| gC-14 | DUPLICATE_MATCH | 153.73 | 155.746 | {"PHONECALL_RIGHT": 3} |

The baseline has four unmatched predictions plus one duplicate match. The inherited V7 diagnostic described only the four unmatched predictions; this implementation audits both categories and asserts their total equals evaluator FP. This corrects error accounting without changing any event metric or matcher. SAFE/OTHER confusion and fragmentation remain measurable failure modes.
