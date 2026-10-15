# False-positive root causes

The V6 hierarchical fusion produced 4 unmatched events. Auditable categories: `{"OTHER_ACTIVITY_MISCLASSIFIED": 3, "SAFE_DRIVING_MISCLASSIFIED": 1}`. Event-duration quantiles are `{"0.0": 2.52, "0.1": 2.9736, "0.25": 3.654, "0.5": 6.0485, "0.75": 8.69475, "0.9": 9.8283, "1.0": 10.584}`.

Each case, sampled label composition, overlap with ground-truth events, and source evidence is recorded in `v7_safe_failure_analysis.json`. Visual cause remains unestablished where original ROI pixels are unavailable. The event matcher and 2-second merge rule were not changed; therefore improvements cannot be attributed to matcher tuning.
