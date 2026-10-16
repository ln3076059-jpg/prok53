# Phone detector resolution probe

Ran the unchanged phone detector on original BODY driver ROIs at each of 13 FN midpoints. Paired inputs use imgsz 384 and 960, confidence floor 0.10 and NMS IoU 0.7. All coordinates, confidences, input hashes, timings and model hash are saved.

```json
{
  "384": {
    "cases": 13,
    "cases_with_phone_proposals": 1
  },
  "960": {
    "cases": 13,
    "cases_with_phone_proposals": 1
  }
}
```

These are unverified proposals on selected diagnostic snapshots, not detection recall or recovered events. Resolution alone did not increase proposal coverage in this probe. The single inspected gB-9 calling ROI visibly contains a handheld phone despite low action confidence; this supports further domain-specific representation research, but is not a new box annotation. No phone fine-tuning used inferred labels.
