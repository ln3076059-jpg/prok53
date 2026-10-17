# V11 final AI research review

Decision: **RESEARCH_ONLY; REJECT FOR PRODUCTION**.

V11 adds reusable behavior ROI, causal motion, and driver-phone association modules. The initial representation set failed its worst-subject gate. A follow-up, pre-declared normalization study selected E3: development clip Macro-F1 56.39%, SAFE recall 60.38%, PHONE recall 66.78%, worst-subject Macro-F1 52.14%, and worst-subject SAFE recall 54.36%. Subject silhouette falls to -0.0047. This supports continued study of causal subject-normalized features.

It does not establish the requested system targets. The controlled event comparator remains P=80.00%, R=55.17%, F1=65.31%, FA/min=0.1813; no new V11 Event F1 exists. Seatbelt best development Macro-F1 remains 58.52%. Rights are unaccepted, all three subjects are repeatedly used development data, protected subjects were untouched, and no independent holdout exists.

`production_approved=false` and `final_status=RESEARCH_ONLY`. No active model/config changed, no DMD media was uploaded, no fine-tuning occurred, and no paper metric was presented as Roadwatch evidence. The next admissible step is rights-approved new subjects with human-validated labels, deduplication, frozen group splits, then a lightweight causal boundary/fusion experiment with full event IDs and one locked evaluation.
