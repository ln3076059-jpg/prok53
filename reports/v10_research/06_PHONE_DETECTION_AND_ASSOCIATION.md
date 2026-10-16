# V10 phone detection and driver association

Evidence level: **LEVEL 1 diagnostic** on 13 repeatedly examined development FNs.

At the fixed 0.02 mining floor, the auxiliary detector produced 226 proposals:
127 were classified `HANDHELD_USE` and
99 were `MOUNTED_OR_STATIC`. Strict context gating removed
99 mounted/static candidates while retaining all 12
human-selected handheld boxes. The remaining hard example is `gB-9-phone-06`.

This result measures proposal yield only. It does not measure detector AP, event recall, or independent
generalization. Raw proposal confidence remains separate from context admission. All 13 cases have
`UNVERIFIED_NO_OCCUPANT_TRACK`; the current geometry is a useful prior but cannot prove driver identity.
The association design therefore returns UNKNOWN when no driver track exists and blocks any claim that a
passenger or mounted device belongs to the driver.

No detector was fine-tuned: DMD rights are unaccepted, the reviewed lane is diagnostic-only, and training
on these selected FNs would leak evaluation cases. A future approved experiment needs full-frame/driver-ROI
coordinate round-trip tests, high-resolution border crops, person/seat tracks, hand-phone association, and
held-out negative scenarios for mounted camera, passenger phone, hand near ear, occlusion, and phone on lap.
