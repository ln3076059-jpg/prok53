# Selective fine-tuning decision

No training was run in V8. F0 reuses the inspected original checkpoints. F1–F5 are BLOCKED pending confirmed labels/license, a subject-disjoint training/validation split and resource approval for the resulting dataset. CPU was sufficient for frozen inference; CUDA is unavailable locally.
Existing pending-approval labels do not satisfy the prompt's confirmed-GT prerequisites. No new legitimate subject pool was found. Repeating failed mean/attention/native112 training or searching thresholds on the same subjects would not establish generalization.
No active or historical checkpoint was overwritten. `checkpoint_metadata.json` records optimizer/scheduler presence. Any later use of best.pt is a warm start unless complete training state is independently verified; this stage establishes no exact resume.
GPU ablations should freeze split and lineage first, then vary original resolution, higher resolution, reviewed hard examples and augmentation separately. No gain or training time is promised. `model_config_ai85_plus.yaml` keeps fine-tuning disabled and records prerequisites.
