# Dataset and generalization

| Subject | Available views | Frame counts |
| --- | --- | --- |
| gB-9 | body, face, hands | 12129, 12141, 12122 |
| gZ-36 | body, face, hands | 15351, 15359, 15337 |
| gC-14 | body | 11916 |

- Development subjects: `gB-9, gZ-36, gC-14`.
- Consumed holdouts excluded from every new load: `gZ-37, gE-28`.
- New approved development subjects: none.
- New independent holdout: none.
- Source rights: `PENDING_PROJECT_REVIEW`; accepted =
  `false`.
- Splits: group-disjoint by source subject for every outer fold.

`AI85_INDEPENDENT_VALIDATION = NOT_ESTABLISHED`

The three subjects have different clothing, brightness, camera composition and view availability.
The subject-ID probe reaches 100%, while action silhouette is only 0.0486. The current dataset is
the binding research bottleneck. Additional architecture search on these same subjects would
increase development-set adaptation without establishing generalization.
