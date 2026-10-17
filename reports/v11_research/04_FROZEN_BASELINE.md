# V11 frozen baseline

The frozen event comparator is unchanged: TP=16, FP=4, FN=13, precision=80.00%, recall=55.17%, Event F1=65.31%, and FA/min=0.1813. Ground truth and matching policy are unchanged.

The V11-A clip baseline reuses the local frozen ResNet representation with the pre-registered LOSO protocol. It obtains Macro-F1 17.56%, balanced accuracy 32.35%, SAFE recall 6.82%, and worst-subject SAFE recall 0.00%. Its subject silhouette is 0.4084, far larger than its action silhouette 0.0362, confirming strong subject/cabin bias.

These are two distinct tasks: the event comparator cannot be compared directly with clip Macro-F1. V11-A provides a controlled clip baseline only. It neither replaces the active model nor claims phone AP, mAP, event recovery, independent generalization, or production readiness.
