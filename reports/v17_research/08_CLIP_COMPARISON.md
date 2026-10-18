# 08. DriveCLIP vs. Lightweight Action Representations

## 1. Comparative Analysis
* **DriveCLIP (ViT-B/16 Backbone):**
  * *Strengths:* Exceptional zero-shot domain transfer across different vehicle cabins, lighting conditions, and camera angles.
  * *Constraints:* High memory footprint (~350MB weights) and ~35ms latency on edge hardware.
* **Lightweight Causal E3 / ResNet Head:**
  * *Strengths:* <2ms latency, deterministic memory, deployable on mobile NPU.
  * *Constraints:* Prone to subject overfitting without explicit domain regularization.

## 2. Recommended V17 Hybrid Strategy
Use frozen CLIP image embeddings offline to generate pseudo-labels and feature distillation targets for a lightweight student temporal head, achieving DriveCLIP robustness at E3 runtime speeds.
