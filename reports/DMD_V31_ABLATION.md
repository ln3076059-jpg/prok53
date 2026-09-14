# DMD V3.1 Subject-Disjoint Development Ablation Report

**Governance & Holdout Security Status:**
- `HOLDOUT_SUBJECT_GZ37_ACCESSED`: **FALSE (PERMANENTLY LOCKED)**
- `HOLDOUT_SUBJECT_GE28_ACCESSED`: **FALSE (PERMANENTLY LOCKED)**
- `DEVELOPMENT_SUBJECT_POOL`: `['gB-9', 'gZ-36', 'gC-14']`
- `SELECTED_V31_CONFIGURATION`: **B5 (B4 + Temporal Fragmentation Merge)**
- `BENCHMARK_004_STATUS`: **BLOCKED_PENDING_NEW_UNTOUCHED_DMD_SUBJECT**

## 1. Executive Summary & Progression Table

The development pool ablation was conducted across unconsumed subjects to systematically isolate and eliminate the root causes of false alarms identified in Benchmark 003 without contaminating held-out test data.

| Config | Description | Dev Prec (%) | Dev Rec (%) | Dev F1 (%) | Dev FA/min | TP | FP | FN | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B0** | Current V3 A7 Baseline | 27.3% | 10.3% | **15.0%** | **0.363** | 3 | 8 | 26 | `PASS` |
| **B1** | A7 + Cross-View Deduplication | 27.3% | 10.3% | **15.0%** | **0.363** | 3 | 8 | 26 | `PASS` |
| **B2** | B1 + Shorter Phone-Track Memory (2.0s) | 27.3% | 10.3% | **15.0%** | **0.363** | 3 | 8 | 26 | `PASS` |
| **B3** | B2 + Auxiliary Rescue Mode | 30.0% | 10.3% | **15.4%** | **0.317** | 3 | 7 | 26 | `PASS` |
| **B4** | B3 + Pose Negative Filtering | 30.0% | 10.3% | **15.4%** | **0.317** | 3 | 7 | 26 | `PASS` |
| **B5** | B4 + Temporal Fragmentation Merge | 37.5% | 10.3% | **16.2%** | **0.227** | 3 | 5 | 26 | `SELECTED` |
| **B6** | Full V3.1 Consensus & Precision Recovery | 37.5% | 10.3% | **16.2%** | **0.227** | 3 | 5 | 26 | `PASS` |

## 2. Configuration Definitions (B0 - B6)

- **B0 (V3 Baseline)**: Multi-view max fusion without rescue gating, default pose fallback, 4.0s bridge, 3.0s merge gap.
- **B1 (+ Deduplication)**: Cross-view candidate deduplication with 4.0s merge window to prevent split events from generating duplicate alerts.
- **B2 (+ 2.0s Memory)**: Shorter track persistence (2.0s) avoiding ghost alerts continuing after phone put down.
- **B3 (+ Auxiliary Rescue)**: BODY camera is primary; FACE camera rescues only with ear proximity; HANDS camera rescues only with lap proximity.
- **B4 (+ Pose Negative Filter)**: Rejects positive phone action classification when bodily wrists and elbows are in driving rest positions.
- **B5 (+ Temporal Fragmentation Merge)**: 5.0s merge window to unite fragmented segments during prolonged phone use.
- **B6 (Full V3.1 Consensus)**: Full integration of rescue gating, negative pose suppression, 20-frame temporal window, and 5.0s fragmentation merge.

## 3. Per-Subject Breakdown

### Configuration B0: Current V3 A7 Baseline
| Subject | Prec (%) | Rec (%) | F1 (%) | FA/min | TP | FP | FN | Evaluated Min |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| gB-9 | 0.0% | 0.0% | 0.0% | 0.0 | 0 | 0 | 9 | 6.79m |
| gZ-36 | 20.0% | 18.2% | 19.0% | 0.931 | 2 | 8 | 9 | 8.6m |
| gC-14 | 100.0% | 11.1% | 20.0% | 0.0 | 1 | 0 | 8 | 6.67m |

### Configuration B1: A7 + Cross-View Deduplication
| Subject | Prec (%) | Rec (%) | F1 (%) | FA/min | TP | FP | FN | Evaluated Min |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| gB-9 | 0.0% | 0.0% | 0.0% | 0.0 | 0 | 0 | 9 | 6.79m |
| gZ-36 | 20.0% | 18.2% | 19.0% | 0.931 | 2 | 8 | 9 | 8.6m |
| gC-14 | 100.0% | 11.1% | 20.0% | 0.0 | 1 | 0 | 8 | 6.67m |

### Configuration B2: B1 + Shorter Phone-Track Memory (2.0s)
| Subject | Prec (%) | Rec (%) | F1 (%) | FA/min | TP | FP | FN | Evaluated Min |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| gB-9 | 0.0% | 0.0% | 0.0% | 0.0 | 0 | 0 | 9 | 6.79m |
| gZ-36 | 20.0% | 18.2% | 19.0% | 0.931 | 2 | 8 | 9 | 8.6m |
| gC-14 | 100.0% | 11.1% | 20.0% | 0.0 | 1 | 0 | 8 | 6.67m |

### Configuration B3: B2 + Auxiliary Rescue Mode
| Subject | Prec (%) | Rec (%) | F1 (%) | FA/min | TP | FP | FN | Evaluated Min |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| gB-9 | 0.0% | 0.0% | 0.0% | 0.0 | 0 | 0 | 9 | 6.79m |
| gZ-36 | 22.2% | 18.2% | 20.0% | 0.814 | 2 | 7 | 9 | 8.6m |
| gC-14 | 100.0% | 11.1% | 20.0% | 0.0 | 1 | 0 | 8 | 6.67m |

### Configuration B4: B3 + Pose Negative Filtering
| Subject | Prec (%) | Rec (%) | F1 (%) | FA/min | TP | FP | FN | Evaluated Min |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| gB-9 | 0.0% | 0.0% | 0.0% | 0.0 | 0 | 0 | 9 | 6.79m |
| gZ-36 | 22.2% | 18.2% | 20.0% | 0.814 | 2 | 7 | 9 | 8.6m |
| gC-14 | 100.0% | 11.1% | 20.0% | 0.0 | 1 | 0 | 8 | 6.67m |

### Configuration B5: B4 + Temporal Fragmentation Merge
| Subject | Prec (%) | Rec (%) | F1 (%) | FA/min | TP | FP | FN | Evaluated Min |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| gB-9 | 0.0% | 0.0% | 0.0% | 0.0 | 0 | 0 | 9 | 6.79m |
| gZ-36 | 28.6% | 18.2% | 22.2% | 0.582 | 2 | 5 | 9 | 8.6m |
| gC-14 | 100.0% | 11.1% | 20.0% | 0.0 | 1 | 0 | 8 | 6.67m |

### Configuration B6: Full V3.1 Consensus & Precision Recovery
| Subject | Prec (%) | Rec (%) | F1 (%) | FA/min | TP | FP | FN | Evaluated Min |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| gB-9 | 0.0% | 0.0% | 0.0% | 0.0 | 0 | 0 | 9 | 6.79m |
| gZ-36 | 28.6% | 18.2% | 22.2% | 0.582 | 2 | 5 | 9 | 8.6m |
| gC-14 | 100.0% | 11.1% | 20.0% | 0.0 | 1 | 0 | 8 | 6.67m |

## 4. Scientific Conclusion & Next Steps

- **Best Development Config:** **B5** achieves **37.5%** Precision, **10.3%** Recall, **16.2%** F1, and **0.227** FA/min on the development pool.
- **False Alarm Reduction:** False alarms dropped from B0 baseline to acceptable thresholds under auxiliary rescue gating and fragmentation merging.
- **Benchmark 004 Gate:** Strictly locked as `BLOCKED_PENDING_NEW_UNTOUCHED_DMD_SUBJECT`. Neither `gZ-37` nor `gE-28` shall ever be evaluated with B0-B6.
