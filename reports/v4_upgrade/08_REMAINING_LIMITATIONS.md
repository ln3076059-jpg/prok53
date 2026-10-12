# Roadwatch V4 Remaining Limitations

1. The current working tree is missing 3,558 tracked dataset/manifests/schema files that predate this
   upgrade. They were not restored because they may be intentional user changes.
2. Development V3 feature caches and source sequences are unavailable, blocking V4 ablation, root-
   cause row export and end-to-end runtime measurement.
3. No new independent holdout exists; V4 generalization is not established.
4. CPU-only PyTorch blocks GPU/FP16/TensorRT measurement.
5. Seatbelt uncertain-class recall is historically 4%, and the JSON support counts conflict with the
   checked-in confusion matrix.
6. The durable database queue provides persistent enqueue and atomic claim, but retry counters,
   heartbeat-based stale-job recovery and multi-GPU scheduling are not yet implemented.
7. Database schema migration tooling and request rate limiting remain absent.
8. V4 thresholds are inherited hypotheses, not newly calibrated values.
9. Historical full-pipeline CPU speed is far below real time.

Production blockers: missing V4 development evidence, missing new holdout, unvalidated candidate,
incomplete queue recovery, no GPU benchmark, and unresolved seatbelt provenance/performance.
