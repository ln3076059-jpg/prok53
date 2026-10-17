# V12 metric reconciliation

V12 recomputed confusion matrices, class metrics, balanced accuracy and pooled metrics directly from all 30 V11 NPZ prediction artifacts. Status: **PASS**, with 0 discrepancies. Source SHA-256 values and fold outputs are in `metric_reconciliation.json`.

The frozen selected E3 result remains clip Macro-F1 56.39%, balanced accuracy 56.81%, SAFE recall 60.38%, PHONE recall 66.78%, OTHER recall 43.26%. Worst-subject Macro-F1 is 52.14%; worst SAFE recall is 54.36%.

A new integrity finding is that stored `predictions` disagree with `argmax(probabilities)` on 25 rows. This follows the V11 one-vs-rest API: class prediction uses decision-function argmax while stored probabilities are separately normalized one-vs-rest sigmoids. V12 portable inference preserves decision argmax and marks probabilities UNCALIBRATED. The artifacts contain 0 timestamp gaps above 0.75 s across repeated variants; V12 streaming resets on cadence breaks rather than bridging them.
