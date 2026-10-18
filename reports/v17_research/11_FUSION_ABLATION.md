# 11. Multi-Evidence Fusion Architecture

## 1. Fusion Formulation
Implemented in `backend/ai/fusion.py` (`LogisticFusion`):
$$\text{Logit} = w_0 + \sum_{k=1}^K w_k \frac{f_k - \mu_k}{\sigma_k}$$
Features ($K=9$):
1. `detector_confidence`
2. `classifier_unfastened`
3. `classifier_fastened`
4. `phone_hand_proximity`
5. `phone_face_proximity`
6. `pose_confidence`
7. `temporal_ratio`
8. `driver_role`
9. `vehicle_context`

## 2. Contradiction Penalization
When unfastened classifier contradicts fastened classifier, or detector confidence is high but vehicle context is missing, the fusion score degrades gracefully to avoid spurious alerts.
