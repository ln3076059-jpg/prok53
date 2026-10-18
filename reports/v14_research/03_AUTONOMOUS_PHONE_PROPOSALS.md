# V14 autonomous phone proposals

`AutonomousPhonePipeline` accepts an original HxWx3 frame, non-oracle metadata, runtime driver evidence and a detector adapter. It rejects metadata keys for ground truth, reviewed or selected boxes, phone boxes and event bounds. Detector proposals retain raw confidence, model hash and source-frame coordinates. Missing proposals produce a source-resolution upper-body or face/hand fallback and remain UNKNOWN; no empty observation becomes safe evidence.

The current specialist and auxiliary checkpoints were hash-reconciled against the existing fully automatic V9 mining run. That run processed 52 frames across 13 FN-selected cases: specialist proposals occurred on 11 frames/5 cases; auxiliary proposals occurred on 48 frames/13 cases; cross-model agreement occurred in 2 cases. Reviewed boxes were not detector inputs. These counts are proposal yield, not precision, recall, AP or recovered events. A new YOLO pass was not run under current RAM constraints.
