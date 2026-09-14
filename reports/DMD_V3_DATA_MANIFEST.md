# DMD V3 Data Governance and Source Manifest

**Project:** Roadwatch Driver Safety  
**Phase:** V3 Multi-View Research Improvement  
**Governance Standard:** Subject Isolation & Fail-Closed Integrity  
**Date:** September 2026  

---

## 1. Subject Role Allocation and Boundary Rules

| Subject ID | Group | Designated V3 Role | Modality / Session | Available Views | Status / Restrictions |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **gC-14** | `gC` | `TEMPORAL_DEVELOPMENT` | RGB / `s2` | BODY | Active development. Used for cross-subject development folds. |
| **gZ-36** | `gZ` | `TEMPORAL_DEVELOPMENT` | RGB / `s2` | BODY (FACE & HANDS extraction in progress) | Active development. Consumed historical Benchmark 001 subject. |
| **gB-9** | `gB` | `TEMPORAL_DEVELOPMENT` | RGB / `s2` | BODY, FACE, HANDS | Active development. Multi-view stream extraction in progress. |
| **gZ-37** | `gZ` | `CONSUMED_BENCHMARK_002` | RGB / `s2` | BODY | **PERMANENTLY FROZEN.** Consumed as held Benchmark 002. Never reused for tuning, feature selection, or retraining. |
| **gE-28** | `gE` | `FINAL_UNTOUCHED_EXTERNAL_DMD_HOLDOUT` | RGB / `s2` | Archive Only | **STRICTLY GUARDED.** Subjected to programmatic `HoldoutAccessError`. Zero frame decode, inspection, or inference prior to formal pre-holdout freeze. |

---

## 2. Technical Integrity & Source Hashes

All DMD media is governed under the Vicomtech Non-Commercial Academic Research License. Signed URLs and authorization query strings are excluded from repository tracking and permanent documentation.

### Subject gC-14 (Development)
- **Archive Filename:** `dmd-dataset-distraction-gC-14.tar.gz` (3,033,297,472 bytes)
- **Archive SHA256:** `cd3b661f53f5cdefc125f614fc8bf9366b7f6a64ff89b72b2e06ccebc829b61a`
- **Body Stream:** `datasets/external_dmd/subjects/gC-14/gC_14_s2_2019-03-04T11;48;02+01;00_rgb_body.mp4` (767,936,358 bytes)
- **Body SHA256:** `bb7c567eea366afba4f2b329142e181333ce62c7ac7085bccebd6daa9ec8694e`
- **Annotation:** `gC_14_s2_2019-03-04T11;48;02+01;00_rgb_ann_distraction.json` (3,933,974 bytes)
- **Annotation SHA256:** `25aa157587dd283e4550b9129321fa9a1fc1889cc7cc1819c7e58d74e3089089`
- **Decode Integrity:** `PASS` (H.264, 1280x720, 29.76 fps, 11,916 frames, monotonic PTS)

### Subject gZ-36 (Development — Multi-View Complete)
- **Archive Filename:** `dmd-dataset-distraction-gZ-36.tar.gz` (5,597,771,964 bytes)
- **Archive SHA256:** `536f71ffa30096e1f4c41f75ef35e4fd6520f7b10c9496ea8111002b115fad86`
- **Body Stream:** `datasets/external_dmd/subjects/gZ-36/gZ_36_s2_2019-04-09T10;39;38+02;00_rgb_body.mp4` (988,594,053 bytes)
- **Body SHA256:** `039d494e15f0f72864e815999c76aadbce84b2d58c0c011374975e4e77ec0ef4`
- **Face Stream:** `datasets/external_dmd/subjects/gZ-36/gZ_36_s2_2019-04-09T10;39;38+02;00_rgb_face.mp4` (987,600,900 bytes)
- **Face SHA256:** `b3d5af83e8061ae1b70302663d0a18f0268d359de99b55bc49fdfdda0787e33d`
- **Hands Stream:** `datasets/external_dmd/subjects/gZ-36/gZ_36_s2_2019-04-09T10;39;38+02;00_rgb_hands.mp4` (990,947,400 bytes)
- **Hands SHA256:** `285973b6ecbd0337fd88877459f26f1891f3dd161b0554d710b3aa5ef233a081`
- **Annotation:** `gZ_36_s2_2019-04-09T10;39;38+02;00_rgb_ann_distraction.json` (5,052,686 bytes)
- **Annotation SHA256:** `45c1d8257fc4a800d52748a7615a243fc222d0a8e84766987ada65a5b08a8291`
- **Decode Integrity:** `PASS` (H.264, 1280x720, 29.76 fps, 15,351 frames, synchronized PTS)
- **Archive Status:** Cleaned up / deleted after extraction to maintain D: safe disk space.

### Subject gZ-37 (Historical Benchmark 002 — Consumed)
- **Archive Filename:** `dmd-dataset-distraction-gZ-37.tar.gz` (4,882,642,095 bytes)
- **Body Stream:** `datasets/external_dmd/original/extracted/gZ_37_s2_2019-04-08T15;45;15+02;00_rgb_body.mp4` (842,305,772 bytes)
- **Body SHA256:** `cb7a6228c545428ec3ea88055c3a183996a7ccc194dc0673bc117d8de51896b9`
- **Annotation:** `gZ_37_s2_2019-04-08T15;45;15+02;00_rgb_ann_distraction.json` (4,241,121 bytes)
- **Annotation SHA256:** `9b91ba555d216b1c12eebe2cef9379ab3cb5a961d693b31ea8b2438953ca69ea`
- **Benchmark 002 Outcome:** Precision 66.7%, Recall 25.0%, F1 36.4%, FA/min 0.14.

### Subject gB-9 (Development — Multi-View Complete)
- **Archive Filename:** `dmd-dataset-distraction-gB-9.tar.gz` (4,700,871,711 bytes)
- **Archive SHA256:** `06398b3ac12abbb6a304f5f06833189280235af12552a44c2381d95500703caa`
- **Body Stream:** `datasets/external_dmd/subjects/gB-9/gB_9_s2_2019-03-07T16;21;20+01;00_rgb_body.mp4` (783,497,406 bytes)
- **Body SHA256:** `36bc2bcd2547033424833207fa6f4694fbf29fdfad2d305c30adb77639924b23`
- **Face Stream:** `datasets/external_dmd/subjects/gB-9/gB_9_s2_2019-03-07T16;21;20+01;00_rgb_face.mp4` (788,873,435 bytes)
- **Face SHA256:** `1c16cdfa966ada454737e691e2efa899d8692b78c17913ea8620f7b7e4ec24fa`
- **Hands Stream:** `datasets/external_dmd/subjects/gB-9/gB_9_s2_2019-03-07T16;21;20+01;00_rgb_hands.mp4` (781,617,679 bytes)
- **Hands SHA256:** `daf082de0e92932b0d29a03a6bc906806d595b606ba0c8153a3d3898b34f1ae0`
- **Annotation:** `gB_9_s2_2019-03-07T16;21;20+01;00_rgb_ann_distraction.json` (3,864,849 bytes)
- **Annotation SHA256:** `a6febae9d122092d1ebb360f51b83705378417b9b3289b9dea0d7bde1bdbd37e`
- **Decode Integrity:** `PASS` (H.264, 1280x720, 29.76 fps, 12,129 frames, synchronized PTS)
- **Archive Status:** Cleaned up / deleted after extraction to maintain D: safe disk space.

### Subject gE-28 (Final Clean Untouched Holdout)
- **Archive Expected Size:** 4,765,223,355 bytes
- **Designated Location:** `datasets/external_dmd/holdout/gE-28/archive_only_before_freeze/`
- **Isolation Policy:** Guarded by `training.dmd.holdout_guard.assert_holdout_untouched`. Any call to decode, inspect, or infer gE-28 raises `HoldoutAccessError`.
