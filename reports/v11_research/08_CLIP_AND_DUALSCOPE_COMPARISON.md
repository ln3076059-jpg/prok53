# V11 CLIP and DualScope comparison

DualScope directly motivates critical behavior regions, dynamic differences, and fine-to-coarse time scales. V11 implements a small causal analogue with fixed region pooling and historical differences; it does not claim to reproduce DualScope's architecture, distillation, datasets, or published metrics. The initial ROI-plus-motion variant obtains pooled Macro-F1 56.60% but collapses on the worst subject, so it is rejected.

DriveCLIP motivates frozen semantic embeddings and task adapters. D4 was not run: no suitable locally verified checkpoint was admitted, ViT-L/14 would add significant download/RAM cost, and dataset rights forbid moving DMD media to an external service. Reporting a CLIP number from a different split or paper would be scientifically invalid.

The accepted development hypothesis is therefore the small E3 normalization-plus-causal representation, which uses existing frozen features and CPU compute. It offers an inspectable causal path and better subject balance, but cannot establish the richer semantic transfer claimed by CLIP or the full spatiotemporal design of DualScope.
