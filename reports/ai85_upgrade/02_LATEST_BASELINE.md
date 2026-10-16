# Latest verified baseline

V7 did not select an eligible replacement. Its comparator reuses the V6 hierarchical outer-LOSO predictions. Re-evaluation from original DMD annotations and stored event predictions exactly reproduces the stored metrics.

```json
{
  "true_positives": 16,
  "false_positives": 5,
  "false_negatives": 13,
  "precision": 0.7619,
  "recall": 0.5517,
  "f1": 0.64,
  "false_alarms_per_minute": 0.2266,
  "detection_latency_mean_seconds": 2.986,
  "duplicate_events": 1,
  "event_fragmentation": 1,
  "evaluated_minutes": 22.063
}
```

There are 29 annotated phone events over 22.063 minutes and three development subjects. The frozen matcher counts repeated matches to one truth as duplicate false positives. `error_analysis.json` reconciles every match and false positive. SAFE recall in this comparison is the three-class coarse-head recall (12.57%); the six-class flat argmax SAFE recall is 14.82%. These quantities must not be interchanged.
