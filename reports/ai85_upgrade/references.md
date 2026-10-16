# Primary-source research review

Reviewed on 2026-10-16. Published benchmark scores below are source claims under their
own protocols, not Roadwatch measurements.

| Source | Verified scope | Use in this work |
| --- | --- | --- |
| [DualScope, AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/37812) | Behavior-centered spatial information and temporal modeling; seven DDAR datasets in the abstract. The PDF endpoint was inaccessible during this review, so no numerical score was transcribed. | Motivates testing temporal frame relevance separately from spatial pixel recovery. The small attention ablation is not a reproduction of DualScope. |
| [DriveCLIP author implementation](https://github.com/zahid-isu/DriveCLIP), [2024 article DOI](https://doi.org/10.1109/TITS.2024.3381175) | CLIP representations for driver behavior; author repository lists DMD, StateFarm, SynDD1 and SAM-DD. The DOI endpoint was inaccessible. | Representation/domain transfer is relevant, but the large visual-language model was not downloaded or substituted into the current ResNet pipeline. No unverified accuracy was imported. |
| [Enhancing Driver Monitoring Systems Based on Novel Multi-Task Fusion Algorithm](https://doi.org/10.3390/s25216799), [PubMed abstract](https://pubmed.ncbi.nlm.nih.gov/41229022/) | Abstract describes BODY/HANDS/FACE information for actions, gaze and hands-on-wheel tasks. | Supports auditing view availability and task evidence; gC-14 has BODY only. No task accuracy is equated with Roadwatch event F1. |
| [Seatbelt Detection Algorithm Improved with Lightweight Approach and Attention Mechanism](https://www.mdpi.com/2076-3417/14/8/3346) | Source reports RST-Net precision 98.49% and F1 98.25% on its seatbelt task. Its data were collected from a traffic department. | Cropped seatbelt features and lightweight attention are relevant, but its protocol does not establish three-state UNCERTAIN performance or Roadwatch generalization. New seatbelt training remains blocked by pending labels and zero UNCERTAIN train examples. |

The new experiments retain the existing evaluator, annotations, frozen model weights
and holdout exclusions. Their failed outcomes are preserved alongside the comparator.
