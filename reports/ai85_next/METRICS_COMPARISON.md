# Metrics comparison

| Variant | TP | FP | FN | Precision | Recall | F1 | FA/min | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| V6 baseline fusion | 16 | 5 | 13 | 76.19% | 55.17% | 64.00% | 0.2266 | historical comparator |
| SN_DIRECT_COARSE_INHERITED_POLICY | 0 | 0 | 29 | 0.00% | 0.00% | 0.00% | 0.0000 | CONTROLLED_HEAD_AND_REPRESENTATION_ABLATION |
| SN_FLAT_INHERITED_POLICY | 1 | 0 | 28 | 100.00% | 3.45% | 6.67% | 0.0000 | CONTROLLED_REPRESENTATION_ABLATION |
| SN_PHONE_INHERITED_POLICY | 16 | 4 | 13 | 80.00% | 55.17% | 65.31% | 0.1813 | CONTROLLED_REPRESENTATION_ABLATION |
| SN_PHONE_NESTED_POLICY | 16 | 5 | 13 | 76.19% | 55.17% | 64.00% | 0.2266 | VALIDATION_SELECTED_DEVELOPMENT_ONLY |
| SN_PHONE_EXPLORATORY_POLICY | 16 | 2 | 13 | 88.89% | 55.17% | 68.09% | 0.0906 | OUTER_DEVELOPMENT_TUNED_NOT_ELIGIBLE_FOR_SELECTION |
| SN_PHONE_INHERITED_POLICY_FUSION | 16 | 4 | 13 | 80.00% | 55.17% | 65.31% | 0.1813 | CONTROLLED_FUSION_ABLATION |

The best observed development trade-off is `SN_PHONE_EXPLORATORY_POLICY`: Precision 88.89%,
Recall 55.17%, F1 68.09%, and 0.0906 FA/min. It is explicitly ineligible because the policy was
inspected on the same three outer development subjects. The nested validation policy reproduces
the baseline event result: 76.19% / 55.17% / 64.00% with 0.2266 FA/min.

No controlled candidate meets Recall or Event F1 >=85%. The direct coarse probe reaches SAFE
recall 70.81%, below the 85% target.
