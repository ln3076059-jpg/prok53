# Remaining Limitations

- Only three development subjects are reusable; model/architecture selection can overfit this pool.
- No untouched new independent holdout was evaluated in V6.
- No valid roadside ground-truth dataset is available, so roadside accuracy, cabin localization and camera-height robustness are unknown.
- Event overmerge remains a material error mode; one-to-one matching can score a long prediction as false positive despite truth overlap.
- Seatbelt uncertain/occluded support is 25 with 4% recall; calibration and abstention estimates are fragile.
- The V6 action model is offline research code and is not integrated into the production backend.
- Camera streaming remains disabled by default and was not benchmarked as a continuous ingest service.
- BackgroundTasks is development-only; production must retain the durable database worker.
- Local hardware has about 6 GB RAM; training required fold isolation and is not a GPU reproduction of cited research.
- Large/private checkpoints remain local and need an approved artifact distribution channel.
- A clean Docker image build was not guaranteed if Docker CLI was unavailable; compose configuration status is in Section 10.

Recommended next evidence: add licensed development subjects, collect human-verified seatbelt uncertainty cases, freeze a new camera-independent holdout, correct temporal overmerge without outer-label tuning, then run a blinded evaluation and formal production approval.
