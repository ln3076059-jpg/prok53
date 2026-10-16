# V10 class imbalance and SAFE driving

The frozen three-subject cache contains 689 SAFE, 1,439 PHONE, and 497 OTHER coarse clips. The source
distribution by subject is preserved in `dataset_manifest.json`: `{"gB-9": {"OTHER_ACTIVITY": 152, "PHONECALL_LEFT": 170, "PHONECALL_RIGHT": 131, "SAFE_DRIVING": 240, "TEXTING_LEFT": 59, "TEXTING_RIGHT": 56}, "gC-14": {"OTHER_ACTIVITY": 170, "PHONECALL_LEFT": 150, "PHONECALL_RIGHT": 164, "SAFE_DRIVING": 208, "TEXTING_LEFT": 49, "TEXTING_RIGHT": 53}, "gZ-36": {"OTHER_ACTIVITY": 175, "PHONECALL_LEFT": 219, "PHONECALL_RIGHT": 217, "SAFE_DRIVING": 241, "TEXTING_LEFT": 81, "TEXTING_RIGHT": 90}}`.

D2 per-class F1 is SAFE 43.79%, PHONE
60.36%, and OTHER 50.46%. Pooled SAFE
recall is 44.27%, while `gZ-36` SAFE recall is only 1.66%, showing
subject-specific class collapse hidden by pooled results. Balanced class weights alone do not solve it.

No focal-loss, contrastive-loss, hard-negative-training, or calibration sweep was admitted. The 13 reviewed
FNs and their selected boxes remain diagnostic-only. Approved future data should explicitly include hand near
face, mirror adjustment, drinking, steering adjustment, mounted devices, passengers, and phone-like objects,
with group-disjoint splits and duplicate checks before loss-function ablations.
