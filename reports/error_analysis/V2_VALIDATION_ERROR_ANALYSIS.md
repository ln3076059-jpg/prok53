# V2 Validation Error Analysis

## Phone Detector Error Modes
- **hand near face:** Some instances where the driver rests their hand near their face are confused with phone usage.
- **hand near ear:** Scratching the ear or adjusting glasses sometimes triggers a false positive.
- **mounted phone:** Phones mounted on the dashboard are generally ignored, but some large glare might trigger detection if occluded.
- **passenger phone:** Model still occasionally detects passenger phones if they lean into the driver's space.
- **dark cabin:** Low light conditions reduce recall significantly.

## Seatbelt Detector Error Modes
- **dark belt / same-color shirt:** High false negative rate when the driver wears black clothing with a black seatbelt.
- **occlusion:** Arms blocking the chest area lower the bounding box confidence.
- **seat/door edge:** Sometimes the B-pillar or seat edge is misidentified as a seatbelt.
- **rear passenger:** Model struggles to detect seatbelts for rear passengers accurately.

## Seatbelt Classifier Error Modes
- **fastened → unfastened:** Misclassifications happen when the belt blends into the clothing.
- **unfastened → fastened:** A diagonal strap from a bag or clothing pattern is sometimes mistaken for a fastened belt.
- **fastened/unfastened → uncertain:** Happens frequently in blurry frames or severe occlusion.

## Recommendations for V2.1 POST-BASELINE EXPERIMENT
- Introduce more synthetic augmentation for low light.
- Collect more hard negative examples of hands near the face without phones.
- Train the classifier with a specialized loss to penalize confident wrong predictions heavily.
