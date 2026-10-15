# V7 current baseline audit

V7 starts from the frozen V6 hierarchical LOSO result. The pooled coarse confusion matrix is `[[84, 196, 388], [74, 1111, 254], [9, 174, 314]]` in SAFE/PHONE/OTHER order. SAFE recall is 12.57%: 196 SAFE clips became PHONE and 388 became OTHER.

The V6 fusion baseline remains TP=16, FP=5, FN=13, F1=64.00%, FA/min=0.2266. No production configuration was changed.

Audit conclusion: SAFE collapse is primarily a coarse SAFE-versus-OTHER generalization problem. Fine phone loss alone cannot repair it.
