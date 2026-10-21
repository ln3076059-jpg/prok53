# Action inputs and output semantics

V6 coarse_targets maps flat 0 to SAFE, flat 1..4 to PHONE and flat 5 to OTHER. Hierarchical coarse softmax is the correct SAFE/PHONE/OTHER score. Fine phone head only subdivides PHONE. V7 multiplies binary NON_PHONE/PHONE gate with conditional SAFE/OTHER or four phone classes. Regression reconstructs coarse scores from six-class output.

Training: native static ROI -> cv2 AREA 64 RGB -> torch bilinear 112, align_corners=false -> ImageNet normalization -> frozen ImageNet ResNet18 -> cached float16 embeddings -> float32 [B,9,512]. Center-frame labels, complete windows, no padding. Original adapter directly resized112 and repeated first embeddings; both differed.

Adapter extraction now matches spatial/precision preprocessing, history is bounded, resets per subject/session/camera/track or timestamp gap, and refuses insufficient history. Centered labels cannot automatically serve causal current-frame gating; online deep activation remains blocked. Rule pose features include heuristic elbow/shoulder estimates and filtered keypoint identity limitations; scores are priors, not calibrated probabilities.

Actual offline forwards and archival score parity are in checkpoint_provenance.json. These use already-extracted features, so they do not benchmark full visual backbone FPS or validate current online preprocessing.
