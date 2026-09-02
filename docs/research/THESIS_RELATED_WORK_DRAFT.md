# Thesis Related Work Draft

## 1. Driver Monitoring Systems
Recent advancements in Driver Monitoring Systems (DMS) heavily rely on Deep Learning. As noted by Dong et al. [10.1109/TITS.2010.2092770], the transition from traditional algorithms to multimodal neural networks has greatly reduced false alarms.

## 2. Distracted Driving Recognition
Distracted driving, particularly visual-manual distraction, is a leading cause of crashes (Dingus et al. [10.1073/pnas.1513271113]). Modern frameworks, as reviewed by Khan et al. [10.1016/j.engappai.2022.105309], advocate for unified architectures over disparate single-modality sensors.

## 3. Cross-Domain Generalization
A major limitation of current DMS is camera-view and domain shift. Celona et al. [10.1109/TITS.2026.3675161] and Drive&Act [10.1109/ICCV.2019.00289] demonstrate that models trained on specific cabin angles fail to generalize without explicit feature disentanglement or robust external validation sets.

## 9. Research Gap & Proposed System
Most existing papers directly classify frames (e.g. YOLOv8 baseline detection). This project proposes a context-aware multi-stage pipeline: CabinLocalization -> Occupant Association -> Semantic Object Detection -> Temporal Hysteresis Tracking. This ensures fail-closed semantics (ignoring passenger phones, handling occluded seatbelts) in a way that pure object detectors cannot achieve, bridging the gap between bounding boxes and true behavioral events.
