# V9 frozen phone proposal mining

This review-only experiment ran two unchanged checkpoints over
52 original-video frames from the registered 13 development
false negatives.
The specialist detector produced proposals on 11 frames;
the independent COCO cell-phone class produced proposals on
48 frames at the fixed 0.02 audit floor.

- Cases with specialist proposals: 5/13.
- Cases with auxiliary proposals: 13/13.
- Cases with cross-model IoU >= 0.25:
  2/13.
- Cases with auxiliary confidence >= 0.20 and hand/face proximity:
  11/13.
- Verified occupant tracks: 0/13.

These are proposal-coverage counts on FN-selected development data, not detector recall,
event recovery, or evidence for choosing an operating threshold. Recurrent cockpit objects can be
false cell-phone proposals. Human review must inspect the contact sheets and original video before
any box becomes a label.
No model weights, labels, event matcher, or production configuration changed.
