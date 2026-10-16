# All false negatives

| Case | Behavior | Seconds | Max PHONE | Action failure |
| --- | --- | --- | --- | --- |
| gB-9-phone-01 | phonecall_right | 39.987–69.825 | 0.4227624535560608 | ACTION_PROBABILITY_OR_SUSTAINED_ACTIVATION |
| gB-9-phone-03 | phonecall_right | 115.155–151.277 | 0.7218237519264221 | ACTION_PROBABILITY_OR_SUSTAINED_ACTIVATION |
| gB-9-phone-05 | phonecall_left | 219.388–265.995 | 0.6306087374687195 | ACTION_PROBABILITY_OR_SUSTAINED_ACTIVATION |
| gB-9-phone-06 | texting_left | 266.028–267.036 | 0.03696706146001816 | ACTION_PROBABILITY_OR_SUSTAINED_ACTIVATION |
| gB-9-phone-07 | texting_left | 284.980–301.512 | 0.46719834208488464 | ACTION_PROBABILITY_OR_SUSTAINED_ACTIVATION |
| gB-9-phone-08 | phonecall_left | 315.491–354.503 | 0.67585289478302 | ACTION_PROBABILITY_OR_SUSTAINED_ACTIVATION |
| gB-9-phone-09 | texting_left | 370.497–382.863 | 0.4168599545955658 | ACTION_PROBABILITY_OR_SUSTAINED_ACTIVATION |
| gZ-36-phone-01 | texting_right | 52.453–53.259 | 0.5471053719520569 | ACTION_PROBABILITY_OR_SUSTAINED_ACTIVATION |
| gZ-36-phone-03 | texting_right | 110.719–111.324 | 0.8559412956237793 | ACTION_PROBABILITY_OR_SUSTAINED_ACTIVATION |
| gZ-36-phone-05 | texting_right | 162.063–162.970 | 0.9157363176345825 | TEMPORAL_BOUNDARY_OR_EVENT_MATCHING |
| gC-14-phone-02 | texting_right | 96.001–108.636 | 0.7713298797607422 | ACTION_PROBABILITY_OR_SUSTAINED_ACTIVATION |
| gC-14-phone-05 | texting_left | 225.907–226.310 | None | ACTION_PROBABILITY_OR_SUSTAINED_ACTIVATION |
| gC-14-phone-06 | phonecall_left | 226.344–269.993 | 0.8543280363082886 | ACTION_PROBABILITY_OR_SUSTAINED_ACTIVATION |

```json
{
  "any_detector_response": {
    "events": 20,
    "fraction": 0.6896551724137931
  },
  "detector_confidence_ge_010": {
    "events": 20,
    "fraction": 0.6896551724137931
  },
  "action_sustained_ge_080": {
    "events": 16,
    "fraction": 0.5517241379310345
  },
  "action_event_overlap": {
    "events": 17,
    "fraction": 0.5862068965517241
  },
  "fused_event_match": {
    "events": 16,
    "fraction": 0.5517241379310345
  }
}
```

These are parallel branch coverage measurements, not a causal sequential funnel. One short gC-14 event has no centered action sample. Other cases fail probability activation, sustained activation or matching. The source pixels for every FN midpoint were exported to `fn_original_rois/`, with exact frame IDs and hashes in `fn_frame_manifest.json`. Per-sample class probabilities, detector confidences and pose vectors are in `error_analysis.json`. The legacy cache lacks phone boxes, association IDs, calibrated fused scores and internal temporal states; those stages cannot be attributed from this cache alone.
