# Scientific validity

- Protocol: outer LOSO over three development subjects.
- Fixed decisions: event threshold 0.80, release ratio 0.65, merge gap 2.0 s, negative-evidence margin 0.15.
- Forbidden data: consumed holdouts `gZ-37`, `gE-28`; accessed count is zero.
- Hard negatives: labels and model scores from the two training subjects only; the outer subject is never accessed by the miner.
- Abstention: ambiguous clips remain in denominators, so false negatives are not hidden.
- Limitation: repeated design decisions on the same three subjects create adaptive overfitting risk even under LOSO.
- Governance: all checkpoints are `DEVELOPMENT_ONLY_NOT_VALIDATED`; no automatic violation and no production activation.
