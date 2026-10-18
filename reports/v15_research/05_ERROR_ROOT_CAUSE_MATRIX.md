# Error root-cause matrix

The case ledger in `fn_gate_ledger.json` preserves all 13 IDs and source hash. Categories are nonexclusive:

| Evidence-backed stage | Cases |
|---|---:|
| Specialist detector miss | 11 |
| Auxiliary candidate not admitted | 12 |
| Driver association unknown | 13 |
| Action evidence below fixed gate | 9 |
| Subsecond temporal risk | 4 |
| Occlusion/localization failure | 1 |

Primary frozen rejection was `PHONE_DETECTOR_NO_PROPOSAL` for 11 cases and `PHONE_BOX_PRESENT_ACTION_UNDER_GATE` for 2. All 13 reviewed cases show a phone and all 13 lack verified driver association. Nine fail the action gate. These facts support a staged detector → association → action study; they do not estimate possible new FP, precision, AP or recovered TP.
