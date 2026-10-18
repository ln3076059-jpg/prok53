# SAHI and action results

The V15 sliced adapter SHA-256 is `80188c39dcd0081906dc7fc36e1a98575468013f07b88ac64ae92a85175e18b7`. Its geometry diagnostic uses six
384×384 tiles at 25% overlap. It measured adapter overhead only and excluded model inference.
D0-D4 AP50/AP50-95, negative-frame FP rate and small-phone recall remain blocked. SAHI has not
demonstrated a Phone AP increase.

No action fitting ran. The inherited development-only E3 metrics remain Macro-F1
**56.39%**, SAFE recall
**60.38%**, PHONE recall
**66.78%** and OTHER recall
**43.26%**. There is still no valid fitted
head/scaler. SAFE recall did not improve in V17.
