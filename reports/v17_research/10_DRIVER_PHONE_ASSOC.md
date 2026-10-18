# 10. Driver-Phone Association & Occlusion Handling

## 1. Association Algorithm
Implemented in `training/dmd/driver_phone_association.py`:
* **Containment Test:** Intersection over Phone Box $\text{IoB} \ge 0.55$.
* **Ambiguity Margin:** If two occupants have containment scores within $\pm 0.15$, status defaults to `UNKNOWN`.
* **Proximity Elevation:** Hand proximity and ear proximity multiply association confidence:
  $$C_{\text{assoc}} = \min(1.0, \text{IoB} \cdot C_{\text{track}} \cdot C_{\text{phone}} \cdot (0.75 + 0.25 \cdot \text{Proximity}))$$

## 2. Handling Missing Evidence
* `UNKNOWN` is never converted to `SAFE`.
* When phone is visible but occupant identity is ambiguous, candidates are marked `NEEDS_REVIEW`.
