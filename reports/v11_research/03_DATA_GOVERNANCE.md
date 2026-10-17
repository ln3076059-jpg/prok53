# V11 data governance

Status: **INDEPENDENT_AI85_NOT_ESTABLISHED**. `RIGHTS_ACCEPTED=false` and `CANONICAL_ELIGIBLE=false`. Local temporal diagnostics are allowed, while new DMD fine-tuning and external uploads are blocked. Development subjects are `gB-9, gC-14, gZ-36`; each has been repeatedly inspected, so all new results are LEVEL 2 at most.

Protected subjects `gE-28, gZ-37` were neither enumerated nor opened by V11. They were not used for training, threshold choice, model selection, or reporting. There are no new approved subjects and no independent holdout. The reviewed 13-phone-FN lane remains diagnostic-only, with 12 valid reviewed auxiliary boxes and one visible-phone hard example without a valid box. Those selected examples were not converted into training labels.

Missing BODY/FACE/HANDS evidence is represented as UNKNOWN. An occluded belt remains `UNCERTAIN_OR_OCCLUDED`; it is never automatically relabeled `UNFASTENED`. Before any future training, the project must verify permission, provenance, human label approval, subject/camera groups, deduplication, and a locked split.
