# Seatbelt results

No seatbelt training was run. All 500 queued candidates still require authenticated human
review, so they cannot become training truth automatically. The best existing development-only
Macro-F1 remains **58.52%** (S2). Its UNCERTAIN recall is
**4.00%**, false-unfastened rate is
**8.00%**, and abstention rate is
**5.46%**.

The three-state policy remains FASTENED, UNFASTENED and UNCERTAIN_OR_OCCLUDED. Occluded belt
evidence is not converted to UNFASTENED, and pending samples stay in the denominator of any
future coverage/abstention report.
