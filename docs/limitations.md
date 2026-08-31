# Limitations

No governed dataset, human approvals, trained weights, validation metrics, frozen-test metrics, or event ground truth exists yet. Subject isolation is not provable without trusted subject metadata. The runtime currently accepts declared vehicle/cabin crops; raw traffic scenes require an upstream vehicle detector/tracker and must fail closed until its ROIs are supplied. Cabin domains, belt visibility, occupant association, mounted phones, occlusion, night conditions, and camera geometry remain open risks. Detector mAP must never be described as system accuracy.
