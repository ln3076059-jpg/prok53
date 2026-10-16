# V9 scientific literature review

All values below are `REPORTED_IN_PAPER`. None is a Roadwatch result.

| Work | Dataset/task | Architecture/method | Metric scope | Code/checkpoint | Roadwatch use |
| --- | --- | --- | --- | --- | --- |
| DualScope (AAAI 2026) | 7 DDAR datasets; video action classification | Behavior-centric SAM distillation + crops; motion/difference encoder + temporal-pyramid Mamba | Paper classification metrics; not Roadwatch event F1 | Paper + HF checkpoint card; linked GitHub returned 404 during audit | Test crop/motion/multi-scale hypotheses under Roadwatch protocol |
| Multi-Task Fusion (Sensors 2025) | DMD; AUC_V2/SAMDD robustness | 3D ResNet-18 + ConvLSTM/BiLSTM on BODY/FACE/HANDS; action/gaze/hand-wheel fusion | Paper attentiveness F1 98.28% at threshold 0.5; different task | Open article; data on request; no official code/weights located | Reliability-aware available-view fusion; gC-14 has BODY only |
| DriveCLIP (IEEE T-ITS 2024) | DMD, StateFarm, SynDD1, SAM-DD | CLIP visual-language frame classifier; separate VideoCLIP reference | Frame/video classification; not event F1 | MIT repo; five small .pkl artifacts; no frozen CLIP artifact installed locally | Controlled feature baseline after artifact/hash registration |
| DGMB-Net (Sensors 2024) | Bayonet-Drivers 10k images/3k labeled; State Farm re-annotated | Teacher-student RDB-Net with APL, HFPN and cascade head | COCO mAP; paper table reports 54.5 on Bayonet-Drivers at 30% labeled | Open paper/PMC; local code/checkpoint reproduction not established | Small-object pyramid relevant; pseudo-labeling blocked pending approved data |
| GM-YOLOv7 + RST-Net (Applied Sciences 2024) | Self-built seatbelt dataset | Ghost/G-ELAN/Mish localization; pruned ResNet + Triplet Attention | Paper RST-Net accuracy 98.25%, not Macro-F1 | Open article; no official code/checkpoint verified | Torso ROI/attention only after three-state labels are approved |
| Overhead YOLOv7 seatbelt (ACS 2025) | Day/night overhead traffic; four box classes | YOLOv7, augmentation, anchor optimization; 640 input, 100 epochs | Paper mAP50 97.46%, F1 95.37%, seatbelt AP 93.40% | CC BY 4.0 paper; generic YOLOv7 code, no task checkpoint/dataset | Relevant roadside geometry; Roadwatch lacks same-camera labeled data |

Primary sources: [DualScope](https://ojs.aaai.org/index.php/AAAI/article/view/37812),
[Multi-Task Fusion](https://pmc.ncbi.nlm.nih.gov/articles/PMC12610507/),
[DriveCLIP](https://github.com/zahid-isu/DriveCLIP),
[DGMB-Net](https://pmc.ncbi.nlm.nih.gov/articles/PMC10935380/),
[RST-Net](https://www.mdpi.com/2076-3417/14/8/3346), and
[overhead YOLOv7](https://ph.pollub.pl/index.php/acs/article/view/7594).

Transferable hypotheses are behavior-centric crops, explicit motion differences, multi-scale temporal
context, small-object feature pyramids, and reliability-aware fusion. Each requires a Roadwatch ablation.
