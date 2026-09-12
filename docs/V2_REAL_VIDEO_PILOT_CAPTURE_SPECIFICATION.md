# FIELD CAPTURE SPECIFICATION: v2_sequence_002
## Internal Governed Real-Video Pilot Capture (Planning / Documentation Only)

**Status:** PROSPECTIVE SPECIFICATION / READY FOR REAL CAPTURE  
**Raw Video Count:** 0  
**Canonical Eligible:** false  
**Human Verified:** false  
**Production Ready:** false  
**Next Legitimate Action:** COMPLETE_REAL_FIELD_CAPTURE_WITH_PROSPECTIVE_EVIDENCE  

---

## 1. PILOT VIDEO TECHNICAL SPECIFICATION

| Parameter | Target Value | Acceptable Notes | Must Verify After Recording |
| :--- | :--- | :--- | :--- |
| **Container** | MP4 (`.mp4`) | ISO Base Media File Format | Container format matches `.mp4` stream markers |
| **Video Codec** | H.264 / AVC | Baseline / Main / High Profile | `h264` codec identifier via ffprobe/PyAV |
| **Resolution** | 1920 x 1080 (1080p Full HD) | 16:9 aspect ratio standard | Width = 1920, Height = 1080 pixels |
| **Frame Rate** | 30 fps (constant frame rate) | Standard constant timing (29.97 / 30.0 fps) | Verify average & real frame rate = 30 fps |
| **Orientation** | Landscape | Horizontal mount (never vertical/portrait) | Pixel aspect ratio 1:1, rotation tag = 0° |
| **Color Range** | SDR (Standard Dynamic Range) | Rec. 709 (BT.709) color space | Pixel format = `yuv420p`, SDR transfer curves |
| **HDR** | OFF | High Dynamic Range disabled | No HDR10, HLG, or Dolby Vision metadata |
| **Slow Motion** | OFF | Normal capture speed (1x) | Native timestamp continuity, no 120/240 fps flags |
| **Frame Rate Mode** | Constant (avoid VFR if possible) | Hardware constant cadence preferred | Timebase continuity, monotonically increasing PTS |
| **Digital Zoom** | OFF (1.0x native optical) | Preserves sensor MTF & native FoV | No digital interpolation artifacts |
| **Beauty / AI Filter**| OFF | Natural visual appearance only | No face smoothing, eye enhancement, or AI beauty |
| **Stabilization** | Normal device stabilization only | Optical or mild electronic stabilization | No aggressive cropping or motion warping |
| **Target Duration** | 45–90 seconds per clip | Complete behavioral cycle | Total stream duration between 45.0s and 90.0s |
| **Pre-Event Padding** | 3–5 seconds | Stable baseline before behavior starts | Visual stability verified prior to first transition |
| **Post-Event Padding**| 3–5 seconds | Stable baseline after behavior ends | Visual stability verified following last transition |

> [!IMPORTANT]
> **Preserve Raw Original File:** Do not transcode, trim, or re-encode files to force target parameters. If a device captures slightly different values, retain the exact raw bytes and extract the empirical metadata via `ffprobe`/`PyAV` during post-capture intake.

---

## 2. CAMERA PLACEMENT SPECIFICATION

- **Position:** Front windshield or dashboard, cabin-facing inward.
- **Capture Angle:** Approximately 30°–45° relative to the driver's sagittal plane.
  *(Note: 30°–45° is a project-specific capture configuration, not an industry-wide universal standard).*
- **Mounting Integrity:** Fixed, rigid mount (suction cup / clamp); not handheld; no vibrations.
- **Safety Mandate:** Must not obstruct driver field of view; must be completely outside airbag deployment zones.
- **Visual Inclusions (Field of View):**
  1. Driver head and face (orientation, gaze direction interpretable).
  2. Shoulders and upper torso.
  3. Both hands observable across normal driving/resting postures.
  4. Steering wheel rim and upper quadrant.
  5. Seatbelt shoulder sash path (from B-pillar anchor across collarbone to hip).
  6. Seatbelt buckle region (where practical).
  7. Hand-held interaction zone (chest/steering wheel height).
  8. Lap / center-console region (where resting phones are placed).
  9. Front passenger seating area clearly visible during Clip C6 (role separation).

---

## 3. PARTICIPANT / VEHICLE IDENTIFIERS

All identifiers must be established **prospectively** prior to recording. Never fabricate them post hoc from arbitrary filenames.

| Scope | Identifier | Purpose / Policy |
| :--- | :--- | :--- |
| **Source Entity** | `INTERNAL_PILOT_TEAM` | Prospective source group designation |
| **Camera Unit** | `CAM_001` | Fixed front-quarter interior camera unit |
| **Vehicle Group** | `VEH_001` | Staged pilot vehicle reference (parked test rig) |
| **Person 1 (Driver)** | `P001` | Participant assigned driver role across C1–C7 |
| **Person 2 (Passenger)** | `P002` | Participant assigned front passenger role in C6 |
| **Session Designation** | `SESSION_DEV_001` | Capture session batch identifier |
| **Dataset Role** | `TEMPORAL_DEVELOPMENT` | Strictly bounded to internal temporal evaluation |
| **Canonical Scope** | `CANONICAL_ELIGIBLE = false` | Cannot be promoted to external benchmark |

*Privacy Policy: Real names, personal phone numbers, and identity documents are prohibited from ordinary annotation and metadata files. Any consent agreements remain strictly in restricted governance evidence.*

---

## 4. PRE-CAPTURE CHECKLIST

Execute and check every item before pressing **Record**:

- [ ] `SOURCE_ID` assigned (`INTERNAL_PILOT_TEAM`)
- [ ] `CAMERA_ID` assigned (`CAM_001`)
- [ ] `VEHICLE_GROUP_ID` assigned (`VEH_001`)
- [ ] `PERSON_GROUP_IDS` assigned (`P001`, `P002`)
- [ ] `CAPTURE_SESSION_ID` assigned (`SESSION_DEV_001`)
- [ ] Dataset role confirmed as `TEMPORAL_DEVELOPMENT`
- [ ] Participant consent confirmed and filed in restricted evidence
- [ ] Project-use rights documentation confirmed
- [ ] Physical paper/digital capture session field log prepared
- [ ] Storage verified: minimum 15 GB free space on recording unit
- [ ] Battery level verified: >= 75% or connected to continuous DC power
- [ ] System clock, date, and local timezone verified on recording device
- [ ] Camera mode verified: Landscape orientation
- [ ] Video resolution verified: 1920x1080 (1080p)
- [ ] Frame rate verified: 30 fps
- [ ] HDR feature confirmed: OFF
- [ ] Slow-motion feature confirmed: OFF
- [ ] Lens physically inspected and wiped clean
- [ ] Camera mount locked firmly; zero play/wobble
- [ ] Airbag deployment zones verified completely unobstructed
- [ ] Driver torso, hands, shoulder belt path clearly visible in monitor
- [ ] Center console and lap regions visible in monitor
- [ ] Front passenger area verified visible in monitor for Clip C6
- [ ] 5-second framing test clip recorded and reviewed
- [ ] Physical privacy scrub: no visible personal documents, badges, or mail
- [ ] Vehicle privacy scrub: license plates, VIN plates, or registered papers out of view

---

## 5. C1–C7 CAPTURE TABLE (INDIVIDUAL RECORDING CARDS)

| Clip | Participant / Role | Vehicle State | Target Behavior / Sequence | Target Duration | Pass Criteria | Retake Conditions |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **C1** | `P001` (Driver) | `PARKED` | **Phone:** `NO_PHONE`<br>**Seatbelt:** `FASTENED`<br>*Negative control clip* | 45–60s | Stable fastened belt; natural driving hand positions; zero phone presence/interaction throughout. | Phone enters frame; belt unlatched; camera moves; hand outside view >3s. |
| **C2** | `P001` (Driver) | `PARKED` | **Phone:** `MOUNTED_OR_STATIC_PHONE`<br>**Seatbelt:** `FASTENED`<br>*Passive phone baseline* | 45–60s | Phone visible on dashboard mount or console; zero touching or holding; belt remains fastened. | Phone touched, held, or adjusted; phone falls; seatbelt occluded. |
| **C3** | `P001` (Driver) | `PARKED` | **Seatbelt Sequence:**<br>`UNFASTENED` (5s)<br>→ *Fastening transition*<br>→ `FASTENED` (stable remainder)<br>**Phone:** `NO_PHONE` | 45–60s | Observable reach for sash, latch insertion, belt settling, followed by stable fastened posture. | Fastening action cut off or hidden by body; buckle off-camera; jerky abort. |
| **C4** | `P001` (Driver) | `PARKED` | **Phone Sequence:**<br>`PHONE_PRESENT_NOT_USED` (5s)<br>→ *Pickup transition*<br>→ `PHONE_USE` (active holding/screen tap 15–20s)<br>→ *Put-down transition*<br>→ `PHONE_PRESENT_NOT_USED` (stable 5s) | 60–90s | Clear resting state; unambiguous hand reach, grip, and usage; clean return to resting surface. | Hand pickup hidden from camera; phone dropped; belt state fluctuates; no clear resting state. |
| **C5** | `P001` (Driver) | `PARKED` | **Seatbelt Sequence:**<br>`FASTENED` (5s)<br>→ *Unfastening transition*<br>→ `UNFASTENED` (stable remainder)<br>**Phone:** `NO_PHONE` | 45–60s | Stable fastened state, hand depress buckle button, sash retraction, stable unfastened state. | Unbuckling obscured by arm/clothing; sash fails to retract; clip stopped too early (<3s post-event). |
| **C6** | `P001` (Driver)<br>`P002` (Passenger) | `PARKED` | **P001 (Driver):** `NO_PHONE`, `FASTENED`<br>**P002 (Passenger):** `PHONE_USE`, `FASTENED`<br>*Role separation benchmark* | 60–90s | Driver holds wheel/lap with NO phone; passenger actively holds and uses phone in adjacent seat; both roles distinct. | Passenger hands cross into driver quadrant; driver touches passenger phone; role ambiguity. |
| **C7** | `P001` (Driver) | `PARKED` | **Observation State:**<br>`UNCERTAIN_OR_OCCLUDED`<br>Natural partial occlusion (jacket over buckle/sash or arm blocking phone view). | 45–60s | Realistic physical occlusion; ground truth strictly documented as visual uncertainty, NOT inferred. | Occlusion artificial/contrived beyond reality; entire occupant hidden; scene uninterpretable. |

---

## 6. VISUAL COVERAGE REQUIREMENTS

### Phone Subsystem Checklist:
- [ ] Driver left hand observable during normal steering wheel grip
- [ ] Driver right hand observable during normal steering wheel grip
- [ ] Resting location of phone clearly identifiable when present
- [ ] Reach-and-pickup trajectory continuously visible across transition
- [ ] Active holding and screen manipulation visually interpretable
- [ ] Put-down placement and hand release continuously visible
- [ ] Mounted/static resting state clearly distinguishable from handheld operation

### Seatbelt Subsystem Checklist:
- [ ] Shoulder belt sash diagonal path visible across chest and collarbone
- [ ] Upper torso and sternum area unobstructed
- [ ] Hand grasp and pull motion visible during fastening transition
- [ ] Settled diagonal line visible during stable `FASTENED` state
- [ ] Buckle press and belt release motion visible during unfastening transition
- [ ] Empty seat / retracted sash clearly visible during stable `UNFASTENED` state
- [ ] Genuine partial occlusion geometry captured in Clip C7

### Occupant Role Subsystem Checklist:
- [ ] Driver seat boundary clearly separated from center console and passenger seat
- [ ] Front passenger seat and occupant fully framed in Clip C6
- [ ] Phone interactions uniquely attributed to the correct occupant's body track

---

## 7. PER-CLIP FIELD LOG TEMPLATE

```yaml
capture_field_record:
  clip_id: "C01"                                # C01 to C07
  capture_session_id: "SESSION_DEV_001"
  source_id: "INTERNAL_PILOT_TEAM"
  camera_id: "CAM_001"
  vehicle_group_id: "VEH_001"
  person_group_ids: ["P001"]                    # ["P001", "P002"] for C06
  driver_person_id: "P001"
  passenger_person_id: null                     # "P002" for C06
  capture_datetime: "YYYY-MM-DDTHH:MM:SS"       # Local timestamp
  timezone: "+07:00"                            # Local timezone offset
  vehicle_state: "PARKED"                       # PARKED / CONTROLLED_TEST_RIG
  camera_view: "FRONT_RIGHT_CABIN_35DEG"
  lighting: "DAYLIGHT"                          # DAYLIGHT | LOW_LIGHT | NIGHT | MIXED | GLARE
  weather: "CLEAR"                              # Descriptive field
  cabin_light: "OFF"                            # ON | OFF
  window_state: "CLOSED"                        # OPEN | CLOSED | MIXED
  planned_phone_state: "NO_PHONE"
  planned_seatbelt_state: "FASTENED"
  planned_transition: "NONE"                    # e.g., FASTENING, PICKUP_PUTDOWN, UNFASTENING
  actual_take_number: 1                         # Increment if retaken
  recording_start_time: "HH:MM:SS"
  recording_end_time: "HH:MM:SS"
  operator_reference: "OP_01"                   # Administrative pseudonym
  consent_reference: "CONSENT_REF_001"          # Stored in restricted vault
  rights_evidence_reference: "RIGHTS_DOC_001"   # Stored in restricted vault
  lineage_evidence_reference: "LINEAGE_DEV_001"
  notes: ""
```

---

## 8. FILE NAMING RULE

Prospective filename format:
```text
SESSION_DEV_001_C01_TAKE01.mp4
SESSION_DEV_001_C02_TAKE01.mp4
SESSION_DEV_001_C03_TAKE01.mp4
SESSION_DEV_001_C04_TAKE01.mp4
SESSION_DEV_001_C05_TAKE01.mp4
SESSION_DEV_001_C06_TAKE01.mp4
SESSION_DEV_001_C07_TAKE01.mp4
```
*(If a retake occurs, increment take number: `..._TAKE02.mp4`)*

> [!CAUTION]
> Filename is administrative metadata only. It must NEVER be accepted as proof of physical lineage. Cryptographic integrity is established exclusively via raw file byte SHA256 checksums bound to the session manifest.

---

## 9. RETAKE CRITERIA

A take must be flagged as rejected and retaken immediately if any of the following occur:
1. **Camera Instability:** Mount slips, tilts, or shakes noticeably during recording.
2. **Key Occlusion:** Driver's hands or upper torso move completely outside the frame during a critical transition.
3. **Belt Framing Loss:** Seatbelt sash or upper torso is clipped out of the frame.
4. **Phone Tracking Loss:** Phone is picked up or put down outside the camera's observable field of view.
5. **Role Ambiguity in C6:** Passenger phone use cannot be visually disentangled from driver space.
6. **Optical / Focus Defects:** Severe autofocus hunting, lens flare, or extreme glare blinding the interaction area.
7. **Premature Termination:** Recording stopped before the required 3–5 seconds of post-event stable padding.
8. **Interrupted Sequence:** Driver aborts or pauses during the middle of a continuous transition.

*Note: Do NOT retake for mild natural variations in head posture or lighting, provided the behavior remains unambiguous and interpretable.*

---

## 10. RAW OFFLOAD PROCEDURE

Immediately upon completion of the physical capture session:

1. Stop recording cleanly on the capture device.
2. **Zero In-Device Editing:** Do NOT trim, filter, rotate, or touch the video on the camera or phone.
3. Connect device via direct hardware connection (USB / SD reader) to the ingest workstation.
4. Copy original unmodified files directly to incoming raw storage:
   `datasets/incoming/v2_sequence_002/raw/`
5. **Absolute Bit-Preservation:**
   - DO NOT rename before initial byte checksum calculation.
   - DO NOT transcode (no re-encoding to H.264/H.265).
   - DO NOT resize or change frame rate.
   - DO NOT remux or strip audio channels.
   - DO NOT apply software stabilization or color correction.
6. Compute cryptographic SHA256 checksum over exact raw binary bytes:
   ```bash
   Get-FileHash -Algorithm SHA256 <file>
   ```
7. Permanently bind:
   `[PSEUDONYM] <--> [CAPTURE_SESSION_ID] <--> [RAW_VIDEO_SHA256]`
8. Log exact raw file size in bytes and file modification timestamp.
9. Record offload operator pseudonym and machine ingest timestamp.
10. Archive capture sheet and consent evidence into restricted governance vault.

---

## 11. EVIDENCE BUNDLE CHECKLIST

The physical and digital evidence pack must contain:

- [ ] **A. Capture Session Record:** Signed field log detailing equipment, location, timestamps, and operators.
- [ ] **B. Rights / Project-Use Record:** Documented institutional or project legal authority to collect pilot footage.
- [ ] **C. Participant Consent Records:** Explicit written participant consent stored securely in restricted governance storage.
- [ ] **D. Raw Offload Record:** Intake log containing original device file paths, offload timestamps, and workstation environment.
- [ ] **E. Session Manifest:** Machine-readable manifest linking each `clip_id`, `participant_id`, `role`, and planned sequence.
- [ ] **F. Raw SHA256 Cryptographic Binding:** Complete SHA256 table linking filenames to verified bitstream hashes.

> [!WARNING]
> Governance Prohibition: Never auto-fabricate signatures, reviewer approvals, consent forms, or simulated timestamps. All records must originate from genuine physical actions.

---

## 12. POST-CAPTURE TECHNICAL VALIDATION PLAN

After offloading raw files to the incoming directory, technical intake must perform two-stage automated inspection:

### Stage 1: Container & Stream Header Inspection (via ffprobe / PyAV)
- [ ] `FILE_SIZE_BYTES`: Exact byte count matches filesystem allocation.
- [ ] `SHA256`: Cryptographic hash matches offload record.
- [ ] `CONTAINER`: Confirmed ISO Media MP4 (`isom` / `mp42`).
- [ ] `VIDEO_CODEC`: Confirmed `h264` (AVC).
- [ ] `AUDIO_CODEC`: Confirmed `aac` or empty/disabled.
- [ ] `WIDTH` & `HEIGHT`: 1920 x 1080.
- [ ] `PIXEL_FORMAT`: `yuv420p` SDR.
- [ ] `AVERAGE_FRAME_RATE`: 30.0 fps (or 29.97).
- [ ] `REAL_FRAME_RATE`: Matches timebase specification.
- [ ] `TIME_BASE`: Valid stream clock (e.g., 1/90000 or 1/30000).
- [ ] `DURATION`: Between 45.0s and 90.0s.
- [ ] `FRAME_COUNT`: Consistent with duration * frame rate.
- [ ] `START_TIME`: 0.000000.

### Stage 2: Sequential Frame Decode Validation
- [ ] Complete frame-by-frame decode loop from frame 0 to EOF.
- [ ] `DECODED_FRAME_COUNT`: Matches container declared packet count.
- [ ] `DECODE_ERRORS`: Zero corrupted NAL units or macroblock decode exceptions.
- [ ] `CORRUPT_FRAMES`: Zero dropped or unrenderable frames.
- [ ] `MISSING_PTS`: Zero frames with missing presentation timestamps.
- [ ] `NON_INCREASING_PRESENTATION_PTS`: Monotonically increasing PTS verified across all frames.
- [ ] `FIRST_PTS`: Verified as initial zero/baseline.
- [ ] `LAST_PTS`: Verified as stream terminal timestamp.
- [ ] `EOF_REACHED`: Stream terminates cleanly on trailer.

---

## 13. ONE-PAGE FIELD OPERATIONAL CHECKLIST

Keep this checklist open and visible throughout recording:

### A. Before Recording
- [ ] Confirm vehicle is in `PARKED` state in safe, non-traffic location.
- [ ] Check device battery (>75%) and storage space (>15 GB free).
- [ ] Clean camera lens with microfiber cloth.
- [ ] Open field capture sheet and pre-assign Session ID and Clip IDs.

### B. Camera Setup
- [ ] Mount camera firmly to windshield / dashboard at ~30°–45° angle to driver.
- [ ] Confirm mount does not cover airbags or block road view.
- [ ] Set camera to Landscape, 1080p, 30 fps, SDR (HDR OFF, Slow-mo OFF).
- [ ] Inspect framing: Driver head, torso, hands, seatbelt sash, and lap console visible.

### C. Person / Vehicle Setup
- [ ] Assign Driver as `P001`; assign Passenger as `P002` (for C6).
- [ ] Check vehicle cabin lighting and ensure no glare blinds camera lens.
- [ ] Ensure personal items, mail, or credentials are removed from view.

### D. Clip Scenarios (C1–C7)
- [ ] **C1:** Driver belted (`FASTENED`), zero phone presence (`NO_PHONE`). 45–60s baseline.
- [ ] **C2:** Driver belted (`FASTENED`), phone resting on static mount/console. No touching. 45–60s.
- [ ] **C3:** Start unbelted (5s) -> Reach, buckle sash -> Belted remainder. 45–60s.
- [ ] **C4:** Phone resting (5s) -> Pick up -> Tap/use phone (15–20s) -> Put down -> Rest (5s). 60–90s.
- [ ] **C5:** Start belted (5s) -> Press release button, retract sash -> Unbelted remainder. 45–60s.
- [ ] **C6:** Driver belted, NO phone. Passenger belted, ACTIVE phone use. 60–90s.
- [ ] **C7:** Partial natural occlusion (jacket/arm across belt or phone). Ambiguous observation. 45–60s.

### E. During Recording
- [ ] Press Record. Wait 3–5 seconds before participant begins action.
- [ ] Ensure participant performs transition steadily at normal human speed.
- [ ] Hold final stable state for 3–5 seconds after action finishes.

### F. After Each Clip
- [ ] Press Stop cleanly.
- [ ] Check clip duration (45–90s).
- [ ] Quick-scrub video on device to verify hands, belt, and phone remained in frame.
- [ ] Log take number, start/end timestamps, and lighting notes on field sheet.
- [ ] If retake is required, log reason and repeat immediately as `TAKE02`.

### G. After Session
- [ ] Double-check all 7 planned scenarios (C1–C7) have at least one valid take.
- [ ] Dismount camera equipment safely.
- [ ] Collect physical capture log sheets.

### H. Raw Offload
- [ ] Transfer original `.mp4` files directly via cable to `datasets/incoming/v2_sequence_002/raw/`.
- [ ] Do not trim, edit, transcode, or re-encode files.
- [ ] Calculate and record SHA256 checksums immediately.

### I. Required Evidence
- [ ] Retain original unedited `.mp4` files.
- [ ] Archive signed physical/digital capture log sheets.
- [ ] Archive participant consent records in restricted vault.

### J. Conditions That Require Retake
- [ ] Camera moved or vibrated during scene.
- [ ] Hand interaction or seatbelt action occurred outside visible frame.
- [ ] Recording cut off before 3–5 seconds of post-event padding.
- [ ] Sudden exposure blowout or glare obscure interaction.
