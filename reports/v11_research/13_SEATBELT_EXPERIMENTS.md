# V11 seatbelt experiments

The frozen seatbelt variants remain the only measured evidence. Best development Macro-F1 is 58.52%; UNCERTAIN recall remains 4% in the inherited evaluation. This is far below the 85% research target and does not justify production use.

The 500-candidate annotation manifest remains pending. Model-assisted suggestions are not ground truth, and V11 creates zero automatic approved labels. Fine-tuning, torso/upper-torso/belt-line ROI comparison, Triplet Attention, visibility-state multitask learning, calibration error, selective risk, and coverage-adjusted risk are therefore BLOCKED/NOT MEASURED under the data-rights and human-approval gates.

The state contract remains `FASTENED`, `UNFASTENED`, and `UNCERTAIN_OR_OCCLUDED`. Invisible or occluded belt evidence cannot become `UNFASTENED`. A future experiment must report per-class F1, Macro-F1, balanced accuracy, false-unfastened rate, abstention rate, and risk at coverage; predicting UNCERTAIN for nearly every sample will not count as success.
