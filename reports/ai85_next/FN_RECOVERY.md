# False-negative recovery

All 13 original BODY ROI images were manually inspected. Every interval is consistent with its
PHONE-use annotation. The table evaluates continuous evidence across each full interval rather
than selecting the maximum frame only.

| Case | Sec | Baseline samples | Base max | Base longest >=.80 | New max | New longest >=.80 | Detector-positive frames | Recovered by fixed new branch | Root cause |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| gB-9-phone-01 | 29.84 | 59 | 0.371 | 0 | 0.894 | 2 | 0 | yes | INSUFFICIENT_CONTINUOUS_ACTION_EVIDENCE |
| gB-9-phone-03 | 36.12 | 72 | 0.707 | 0 | 0.956 | 22 | 0 | yes | INSUFFICIENT_CONTINUOUS_ACTION_EVIDENCE |
| gB-9-phone-05 | 46.61 | 92 | 0.611 | 0 | 0.925 | 3 | 0 | yes | INSUFFICIENT_CONTINUOUS_ACTION_EVIDENCE |
| gB-9-phone-06 | 1.01 | 2 | 0.054 | 0 | 0.243 | 0 | 0 | no | INSUFFICIENT_CONTINUOUS_ACTION_EVIDENCE |
| gB-9-phone-07 | 16.53 | 33 | 0.356 | 0 | 0.627 | 0 | 0 | no | INSUFFICIENT_CONTINUOUS_ACTION_EVIDENCE |
| gB-9-phone-08 | 39.01 | 78 | 0.658 | 0 | 0.911 | 6 | 2 | yes | INSUFFICIENT_CONTINUOUS_ACTION_EVIDENCE |
| gB-9-phone-09 | 12.37 | 24 | 0.391 | 0 | 0.631 | 0 | 5 | no | INSUFFICIENT_CONTINUOUS_ACTION_EVIDENCE |
| gZ-36-phone-01 | 0.81 | 1 | 0.595 | 0 | 0.461 | 0 | 0 | no | INSUFFICIENT_CONTINUOUS_ACTION_EVIDENCE |
| gZ-36-phone-03 | 0.60 | 1 | 0.811 | 1 | 0.219 | 0 | 0 | no | INSUFFICIENT_CONTINUOUS_ACTION_EVIDENCE |
| gZ-36-phone-05 | 0.91 | 2 | 0.914 | 2 | 0.722 | 0 | 0 | no | TEMPORAL_BOUNDARY_OR_EVENT_MATCHING |
| gC-14-phone-02 | 12.63 | 25 | 0.714 | 0 | 0.497 | 0 | 0 | no | INSUFFICIENT_CONTINUOUS_ACTION_EVIDENCE |
| gC-14-phone-05 | 0.40 | 0 | 0.000 | 0 | 0.000 | 0 | 0 | no | MISSING_CENTERED_SAMPLE |
| gC-14-phone-06 | 43.65 | 86 | 0.777 | 0 | 0.956 | 7 | 0 | yes | INSUFFICIENT_CONTINUOUS_ACTION_EVIDENCE |

The subject-normalized binary branch recovers `5` of the 13 baseline FNs under the
inherited event policy. Overall TP remains 16 because it also loses previously matched events;
recovery of selected errors is therefore not a net recall gain.

## Visual review

- `gB-9-phone-01`: Phone clearly visible at right ear; hand and device occupy a large region.
- `gB-9-phone-03`: Phone clearly visible at right ear; little occlusion.
- `gB-9-phone-05`: Phone clearly visible at left ear; strong behavior evidence.
- `gB-9-phone-06`: Short interval; gaze and hands move toward lap, but phone is occluded.
- `gB-9-phone-07`: Phone clearly visible in left hand near torso.
- `gB-9-phone-08`: Phone at left ear, partly behind hand but behavior is clear.
- `gB-9-phone-09`: Phone clearly visible in left hand near torso.
- `gZ-36-phone-01`: Phone clearly visible in right hand near lap.
- `gZ-36-phone-03`: Phone clearly visible in right hand near torso.
- `gZ-36-phone-05`: Phone clearly visible in right hand; bright screen and partial glare.
- `gC-14-phone-02`: Phone is small and partly clipped at the lower-right ROI boundary.
- `gC-14-phone-05`: Phone is visible during rapid hand motion with substantial motion blur.
- `gC-14-phone-06`: Phone clearly visible at right ear; little occlusion.

`gB-9-phone-06` is only about one second and has two centered samples with an occluded phone.
`gC-14-phone-05` is about 0.40 seconds and has no centered sample. `gZ-36-phone-05` reaches two
baseline activations but fails temporal matching. These three mechanisms cannot be repaired by a
single global threshold without additional false alarms.

The detector trace records every proposal box, confidence, driver ROI, timestamp and legacy view
confidence in `detector_trace.json`. Association is limited to containment in a driver ROI; no
phone track, occupant track, or ground-truth box exists.
