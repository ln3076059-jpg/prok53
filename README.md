# Roadwatch Driver Safety

Evidence-first driver safety research platform featuring an end-to-end multi-stage architecture for occupant-aware phone use reasoning, three-state seatbelt evaluation, continuous temporal event confirmation, tamper-evident evidence preservation, and human-in-the-loop review.

---

## 1. Overview
Roadwatch is an academic research and engineering prototype designed to reliably detect risky driver behaviors (handheld phone use and unfastened seatbelts) from vehicular cabin camera feeds while preventing premature false alarms. The system adheres strictly to scientific honesty:
- **ACADEMIC_PROJECT_COMPLETE:** `true`
- **ENGINEERING_COMPLETE:** `true`
- **DMD_PHONE_TEMPORAL_BENCHMARK:** `COMPLETE`
- **CANONICAL_SELF_CAPTURE_PILOT:** `NOT_COMPLETED` (Preserved as `OPTIONAL_FUTURE_SELF_CAPTURE_VALIDATION`)
- **FINAL_UNTOUCHED_EVENT_HOLDOUT:** `NOT_COMPLETED`
- **PRODUCTION_READY:** `false`

The system is fully operational and demonstrated across all layers: deep learning component detectors, continuous temporal state machine, SQLite/MySQL persistence, FastAPI backend, and React/TypeScript review dashboard.

---

## 2. Architecture
Roadwatch follows a decoupled, multi-stage, fail-closed pipeline rather than a naive monolithic detector:

```text
FRAME
  ↓
VEHICLE DETECTION & TRACKING (ByteTrack / YOLO11n)
  ↓
CABIN LOCALIZATION (Geometric / Windshield ROI)
  ↓
OCCUPANT ROLE ASSOCIATION (Driver vs. Passenger Assignment)
  ↓
┌─────────────────────────────────┬─────────────────────────────────┐
│ PHONE BRANCH                    │ SEATBELT BRANCH                 │
│ • YOLO11s Phone Detector (960p) │ • YOLO11s Torso/ROI Detector    │
│ • YOLO11n Pose Keypoint Binding │ • YOLO11s-cls 3-State Classifier│
│ • Hand/Face Proximity Context   │ • Fastened / Unfastened / Uncert│
└────────────────┬────────────────┴────────────────┬────────────────┘
                 ↓                                 ↓
                     TEMPORAL FUSION & HYSTERESIS
                                   ↓
                       EVENT ENGINE & COOLDOWN
                                   ↓
                       TAMPER-EVIDENT EVIDENCE
                                   ↓
                       HUMAN REVIEW WORKFLOW
```

---

## 3. Phone Semantics
A detected phone is **never** automatically treated as a distraction violation. The system evaluates:
- **Semantic States:** `NO_PHONE`, `PHONE_PRESENT_NOT_USED`, `MOUNTED_OR_STATIC_PHONE`, `PHONE_USE`, `UNKNOWN_PHONE_CONTEXT`.
- **Driver Role Binding:** Only phones associated with the resolved `driver` role can trigger `PHONE` violations. Passenger phone usage is strictly filtered out.
- **Pose Proximity Evidence:** Uses YOLO11n-pose wrist and facial keypoint Euclidean distance to distinguish handheld texting/calling from static mounts.
- **Fail-Closed Rule:** Detections with `UNKNOWN` occupant role route to `NEEDS_REVIEW` and never default to driver violations.

---

## 4. Seatbelt Semantics
Seatbelt evaluation operates on occupant torso ROIs using three-state classification:
- **`seatbelt_fastened`:** Confirmed fastened diagonal chest webbing.
- **`seatbelt_unfastened`:** Explicit visible absence of chest belt on occupant upper body.
- **`uncertain_or_occluded`:** Low contrast, dark clothing, bag straps, or occlusion.
- **Fail-Closed Principle:** `uncertain_or_occluded` is **never** converted to an unfastened violation. It routes safely to human review.
- **Dataset Boundary:** DMD Distraction RGB does not supply project-compliant seatbelt ground truth; seatbelt scientific evaluation is held at component validation/frozen test.

---

## 5. Installation

```bash
# Clone repository
git clone https://github.com/ln3076059-jpg/prok53.git
cd prok53

# Set up Python virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Configure environment variables
copy .env.example .env
```

---

## 6. Backend Launch

```powershell
# Create administrative reviewer account
python tools/create_admin.py reviewer@example.org a-strong-local-password

# Launch FastAPI development server
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```
API documentation is available at `http://localhost:8000/docs`.

---

## 7. Frontend Launch

```bash
cd frontend
npm install
npm run dev
```
The operations dashboard opens at `http://localhost:5173`. Build for production using:
```bash
npm run build
```

---

## 8. DMD Dataset Setup (Vicomtech Driver Monitoring)
The project utilizes the external Vicomtech DMD dataset for temporal development:
- **Dataset Role:** `TEMPORAL_DEVELOPMENT` (Canonical Eligible: False, Untouched Holdout: False).
- **Target Profile:** Session `s2`, Channel `RGB`, Stream `BODY`.
- **Target Labels:** `texting_right`, `texting_left`, `phonecall_right`, `phonecall_left`.
- **Acquisition & Retention:** Downloads archives, extracts only target `.mp4` and `.json`, computes SHA256 checksums, and removes archives to maintain disk capacity.

---

## 9. DMD Temporal Benchmark
The temporal benchmark operates on subject-disjoint DMD sequences:
- **Calibration Subjects:** `14` (`gC-14`), `37` (`gZ-37`)
- **Held-Subject Evaluation:** `36` (`gZ-36`)
- **Overlap:** `SUBJECT_OVERLAP = 0`, `SHA_OVERLAP = 0`

Execute calibration and held evaluation:
```powershell
# Run temporal calibration parameter sweep
python -m training.dmd.calibrate

# Run held-subject benchmark
python -m training.dmd.evaluate_temporal
```

---

## 10. Model Artifacts

| Component | Architecture | Weight Path | SHA256 Hash |
|---|---|---|---|
| **Phone Detector** | YOLO11s (960p) | `models/active/v2/phone_detector.pt` | `840a29cb2151b881279cdabe25b03b28c5dcf40a43464edd6b672c8851f77d54` |
| **Seatbelt Detector** | YOLO11s (1280p) | `models/active/v2/seatbelt_detector.pt` | `361436ab073c8fcc17a041f098285efdd0cf6775a8970be2ffada4e45c6bc500` |
| **Seatbelt Classifier** | YOLO11s-cls | `models/active/v2/seatbelt_classifier.pt` | `e55158e3f152922e710ad260295da16488ffc2c6a145b8dfdc5c4ce323392bd4` |
| **Vehicle Detector** | YOLO11n | `models/auxiliary/yolo11n.pt` | `3c8d3568c4d293ca5610bc13328e08d6c7075c32ab5fa626b9c9baeeffc24151` |
| **Pose Estimator** | YOLO11n-pose | `models/auxiliary/yolo11n-pose.pt` | `fa3eb05c4889c2598380d19a2e3cfcf1a9528bb5eb4fbaf9b9e67d2ca85806c9` |

---

## 11. Verified Component Metrics

### Validation Split (rtx5060ti_16gb / RTX 3090)
- **Phone Detector:** Precision 97.13%, Recall 82.73%, mAP50 94.48%, mAP50-95 70.03%
- **Seatbelt Detector:** Precision 92.57%, Recall 87.85%, mAP50 94.73%, mAP50-95 53.70%
- **Seatbelt Classifier:** Top-1 Accuracy 80.97% (~81.0%)

### Frozen Component Test (`FROZEN_TEST_RUN_COUNT = 1`, Preserved)
- **Phone Detector:** Precision 90.20%, Recall 86.20%, mAP50 90.50%, mAP50-95 65.50%
- **Seatbelt Detector:** Precision 93.80%, Recall 88.20%, mAP50 92.70%, mAP50-95 47.70%
- **Seatbelt Classifier:** Top-1 Accuracy 78.90%

---

## 12. Event Metrics (DMD Held-Subject Benchmark)

- **Benchmark Classification:** `DMD_EXTERNAL_DEVELOPMENT_BENCHMARK`
- **Phone Event Precision:** **85.7%**
- **Phone Event Recall:** **75.0%**
- **Phone Event F1 Score:** **80.0%**
- **False Alarms per Minute:** **0.55 / min**
- **Mean Event Start Latency:** **+0.72s** (Onset tolerance: $\le 2.50$s)
- **Mean Event End Latency:** **+1.45s** (Offset tolerance: $\le 3.50$s)
- **Seatbelt Temporal Independent Benchmark:** `NOT_AVAILABLE` (Component validated only)

---

## 13. Runtime Benchmark (Measured Target Hardware)

Measured on local CPU execution (`reports/V2_RUNTIME_BENCHMARK_FINAL.md`):
- **Video Decode:** **63.86 FPS** (p50: 13.89 ms)
- **Phone & Seatbelt Detector:** p50 1060.2 ms
- **Pose Estimator:** p50 122.5 ms
- **Seatbelt Classifier:** p50 49.5 ms
- **Temporal Fusion:** p50 0.025 ms
- **Total Serial Processing:** p50 1254.3 ms (0.65 FPS CPU)
- **Host Process RSS:** 233.1 MB RAM

---

## 14. Human Review Workflow
The platform features an append-only human review system:
- Explicit separation between `MODEL_PROPOSAL` and `HUMAN_DECISION`.
- States: `PENDING`, `CONFIRMED`, `REJECTED`, `NEEDS_REVIEW`.
- Tamper detection using SHA-256 bound audit logs.

---

## 15. Evidence System
Each confirmed event generates an immutable evidence bundle:
- Video snippet bounding box overlays
- Annotated keyframe image
- JSON metadata trace with model version and component confidences
- Stored under `evidence/` with path-traversal prevention.

---

## 16. Testing

```powershell
# Run full software test suite (450+ tests)
python -m pytest -q

# Run DMD pipeline unit tests
python -m pytest tests/test_dmd_pipeline.py -v

# Run frontend production build
cd frontend; npm run build; cd ..
```

---

## 17. Deployment
- **Database:** Defaults to SQLite for local development; supports MySQL via `DATABASE_URL`.
- **Security:** Reject development secrets in production (`APP_ENV=production`), password hashing via Passlib/Bcrypt, CORS domain restriction, and upload MIME validation.

---

## 18. Limitations
1. **CPU Real-Time Constraint:** Sequential CPU inference runs at 0.65 FPS; GPU or 5 FPS frame sampling is necessary for 30 FPS line-rate processing.
2. **Cabin Lighting & Extreme Glare:** Low IR contrast or direct sunlight glare can degrade detector confidence.
3. **Absence of Production Holdout:** Benchmarked against external DMD development sequences; field deployment certification requires independent real-world fleet validation.

---

## 19. Project Status
```text
ACADEMIC_PROJECT_COMPLETE = true
ENGINEERING_COMPLETE = true
SOFTWARE_TESTS = PASS (454 passed / 0 failed)
DMD_PHONE_TEMPORAL_BENCHMARK = COMPLETE
CANONICAL_SELF_CAPTURE_PILOT = NOT_COMPLETED
FINAL_UNTOUCHED_EVENT_HOLDOUT = NOT_COMPLETED
PRODUCTION_READY = false
```

---

## 20. Repository Map
- `backend/`: FastAPI application, AI auxiliary models, temporal engine, database entities
- `frontend/`: React/TypeScript operations dashboard with Vite and Carbon Design
- `models/`: Active locked weights (`models/active/v2/`), model configs, baseline lock manifests
- `training/dmd/`: DMD adapter, timeline validation, subject split, temporal calibration, benchmark runner
- `tools/`: Runtime benchmarking, annotation review, admin provisioning
- `reports/`: Verifiable metrics, truth audit, split audit, error analysis, runtime benchmarks
- `docs/`: System architecture, related work, literature comparisons, specifications
