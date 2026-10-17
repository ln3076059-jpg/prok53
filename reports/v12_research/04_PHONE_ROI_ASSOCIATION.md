# V12 phone ROI and association

V12 decoded one original-resolution BODY frame for each of the 13 reviewed FNs and verified source video hashes against the V11 manifest. Twelve reviewed phone boxes are preserved in real-pixel dynamic crops; `gB-9-phone-06` has no valid reviewed box. No detector was rerun, and no detection AP or event recovery is claimed.

The reviewed phone boxes would be only 2.19–11.73 pixels wide after reduction to the 64-pixel driver cache. This supplies direct evidence that the V11 representation cache is too coarse for small-phone localization. The full crops and coordinate provenance are stored under `roi_evidence/` and `roi_audit.json`.

The inherited root-cause counts remain: specialist miss 11/13, auxiliary valid candidate not admitted 12/13, action below gate 9/13, driver association UNKNOWN 13/13. Association now abstains when mounted/static context lacks independent verification or multiple verified occupant boxes are similarly plausible. Its score remains a heuristic, not a calibrated probability. Real association accuracy remains unmeasured because occupant-track ground truth is absent.
