# Primary references checked for this stage

- [DualScope, AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/37812): behavior-region distillation and salient temporal dynamics motivate checking ROI detail and motion. This stage implements a simple motion sampler and spatial probes; it does not reproduce SAM distillation or Mamba.
- [DriveCLIP author code](https://github.com/zahid-isu/DriveCLIP): frame-based vision-language action recognition is an alternative representation direction. No CLIP weights were downloaded, trained, or integrated here.
- [Enhancing Driver Monitoring Systems Based on Novel Multi-Task Fusion Algorithm](https://pmc.ncbi.nlm.nih.gov/articles/PMC12610507/): activity, gaze and hand-wheel evidence motivate availability-aware fusion. This stage does not reproduce that paper or claim gaze/hand landmarks are available.
- [Ultralytics validation documentation](https://docs.ultralytics.com/modes/val/): AP validation is distinct from operating-point prediction. Installed version is 8.3.187; local validator defaults must be inspected for reproducibility. The spatial probe uses conf=0.10 and NMS IoU=0.70 and does not calculate AP.
