# V2 Validation Error Analysis

## 1. Overview
This error analysis relies entirely on the **Canonical Validation splits**. The frozen test dataset remains untouched for this analysis.

## 2. Phone Detector Analysis
**False Positives (Visual Confusion):**
- Hand near face or ear.
- Dashboard objects resembling a phone.
- Reflections on windows.

**False Negatives (Missed Detections):**
- Tiny phones (far distance).
- Partially occluded phones (hidden behind steering wheel or hand).
- Dark cabin or heavy glare.

**System-Level Rejections (Business Logic):**
*Note: The detector may correctly find a phone, but the system must reject it if:*
- It is a passenger's phone.
- It is a mounted/static phone (e.g., GPS).
- It belongs to a person outside the vehicle.
- It is located outside the cabin.
- The occupant role is ambiguous.

## 3. Seatbelt Detector Analysis
**False Positives (Irrelevant Occupants):**
- Persons outside the vehicle.
- Motorcycle riders.

**False Negatives (Missed Torsos):**
- Dark clothing or belts matching the shirt color.
- Diagonal seat edges mimicking belts.
- Partial belts or severe occlusion.
- Rear passengers or heavily cropped torsos.
- UNKNOWN belt visibility.

*Note: Missing an occupant's upper body DOES NOT mean the occupant is unfastened. It simply means the classifier won't run, failing closed.*

## 4. Seatbelt Classifier Analysis
**Confusion Modes Observed:**
- Fastened predicted as Unfastened (Critical: False Violation).
- Unfastened predicted as Fastened (Missed Violation).
- Uncertain predicted as confident Fastened or Unfastened.
- Valid classes predicted as Uncertain.

**Mitigation:** 
We introduced an asymmetric confidence reject policy (`0.60` for unfastened, `0.50` for fastened). Low-confidence predictions are forced into `uncertain_or_occluded`, avoiding automated false violations.

## 5. Conclusion
No weights will be mutated based on this error analysis. The `V2_BASELINE_001` lock remains absolute. If further improvements are strictly necessary, a `V2_1_RETRAIN_RECOMMENDATION` must be drafted, establishing a completely new experimental track.
