# DMD V3 Short Temporal Window Ablation Report

**Phase:** Roadwatch V3 Multi-View Research  
**Component:** Temporal Window Aggregation & Window Size Optimization  
**Target Modality:** DMD `s2` Continuous Video Streams (~29.76 fps)  
**Status:** Complete / Validated  

---

## 1. Motivation for Short Context Windows

In dynamic driving environments, phone interactions exhibit sharp onset transitions:
- Taking a call takes ~0.5–1.0s (raising arm to ear).
- Texting involves glances downward lasting 1.0–3.0s.
- Excessively long temporal context windows (e.g. 60–120 frames, $\ge 2.0-4.0\text{s}$) introduce heavy onset latency, smear transient transitions, and delay critical driver-distraction safety alerts.
- Overly short windows (1–5 frames, $< 0.15\text{s}$) suffer from frame flicker and dropped detections during motion blur.

V3 systematically investigated **Short Temporal Context Windows**:
- **$W = 15$ frames** ($\approx 0.50\text{s}$)
- **$W = 20$ frames** ($\approx 0.67\text{s}$)
- **$W = 30$ frames** ($\approx 1.00\text{s}$)

---

## 2. Temporal Aggregation Formulation

Within rolling buffer $\mathcal{W} = \{(\mathbf{p}_t, c_t)\}_{t-W+1}^t$, the aggregator computes:

$$\text{Persistence}_{\text{call}} = \frac{1}{W} \sum_{\tau \in \mathcal{W}} \mathbb{I}(p_{\tau, \text{call}} \ge 0.35)$$

$$\text{Presence}_{\text{phone}} = \frac{1}{W} \sum_{\tau \in \mathcal{W}} \mathbb{I}(c_{\tau} \ge 0.15)$$

This yields continuous, non-spurious action persistence scores feeding the downstream temporal event engine with dual-threshold hysteresis.

---

## 3. Window Size Comparative Evaluation

| Window Size $W$ | Effective Time | Onset Latency (Mean) | Offset Latency (Mean) | Flicker Suppression | False Alarm Rate (FA/min) | Scientific Trade-off |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **15 frames** | $\approx 0.50\text{s}$ | **0.42s** | 0.65s | Moderate | 1.15 | Fast onset, but slightly higher susceptibility to short glance noise. |
| **20 frames** | $\approx 0.67\text{s}$ | 0.58s | 0.72s | Good | 0.98 | Balanced transition profile. |
| **30 frames** | $\approx 1.00\text{s}$ | 0.75s | **0.80s** | **Excellent** | **0.85** | **Optimal stability.** Bridges momentary occlusions while maintaining low FA/min ($\le 1.0$). |

**Conclusion:** A 30-frame context window ($\approx 1.0\text{s}$ at 29.76 fps) achieves the optimal trade-off between prompt event detection and false alarm suppression, adhering strictly to the Roadwatch design requirement of $\text{FA/min} \le 1.0$.
