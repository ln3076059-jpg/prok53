from pathlib import Path
from backend.ai.events import TemporalEventEngine, Observation
from training.export_event_predictions import export_predictions_to_csv
from training.evaluate_events import evaluate, _read


def _context_for_prediction(prediction: dict[str, str], **overrides: str) -> dict[str, str]:
    context = {
        "video_id": prediction["video_id"],
        "occupant_id": prediction["occupant_id"],
        "occupant_role": "driver",
        "vehicle_id": prediction["vehicle_id"],
        "cabin_id": prediction["cabin_id"],
        "start_seconds": "0",
        "end_seconds": "60",
        "inside_vehicle": "true",
        "outside_vehicle_person": "false",
        "motorcycle_flag": "false",
        "phone_state": "NO_PHONE",
        "seatbelt_state": "FASTENED",
    }
    context.update(overrides)
    return context

def test_end_to_end_prediction_evaluation(tmp_path: Path):
    # 1. Generate runtime EventCandidates using TemporalEventEngine
    engine = TemporalEventEngine(window_seconds=2.0)
    
    # We will simulate a PHONE event for a driver
    observations = [
        Observation(
            timestamp=1.0, 
            class_name="phone", 
            confidence=0.8, 
            track_id=1, 
            occupant_role="driver", 
            vehicle_context_id="v1",
            cabin_id="c1",
            visibility="clear",
            outside_vehicle_person="false",
            behavior_label="PHONE_USE",
            vehicle_type="car",
            phone_context="HANDHELD",
            occupant_id="v1:occupant-track:1"
        ),
        Observation(
            timestamp=1.5, 
            class_name="phone", 
            confidence=0.85, 
            track_id=1, 
            occupant_role="driver", 
            vehicle_context_id="v1",
            cabin_id="c1",
            visibility="clear",
            outside_vehicle_person="false",
            behavior_label="PHONE_USE",
            vehicle_type="car",
            phone_context="HANDHELD",
            occupant_id="v1:occupant-track:1"
        ),
        Observation(
            timestamp=2.0, 
            class_name="phone", 
            confidence=0.9, 
            track_id=1, 
            occupant_role="driver", 
            vehicle_context_id="v1",
            cabin_id="c1",
            visibility="clear",
            outside_vehicle_person="false",
            behavior_label="PHONE_USE",
            vehicle_type="car",
            phone_context="HANDHELD",
            occupant_id="v1:occupant-track:1"
        ),
        Observation(
            timestamp=2.5, 
            class_name="phone", 
            confidence=0.95, 
            track_id=1, 
            occupant_role="driver", 
            vehicle_context_id="v1",
            cabin_id="c1",
            visibility="clear",
            outside_vehicle_person="false",
            behavior_label="PHONE_USE",
            vehicle_type="car",
            phone_context="HANDHELD",
            occupant_id="v1:occupant-track:1"
        ),
    ]
    
    candidates = []
    for obs in observations:
        candidates.extend(engine.add(obs))
        
    assert len(candidates) > 0, "TemporalEventEngine should produce at least one candidate"
    
    # 2. Export candidates to CSV
    csv_path = tmp_path / "predictions.csv"
    export_predictions_to_csv(candidates, video_id="video_1", output_path=csv_path)
    
    assert csv_path.exists()
    
    # 3. Read back using evaluate_events _read logic
    prediction_rows, pred_fields = _read(csv_path)
    assert len(prediction_rows) == 1
    
    assert prediction_rows[0]["observation_count"] == "4"
    assert prediction_rows[0]["start_seconds"] == "1.000"
    assert prediction_rows[0]["end_seconds"] == "2.500"
    assert prediction_rows[0]["occupant_id"] == "v1:occupant-track:1"
    assert prediction_rows[0]["track_id"] == "1"
    
    # 5. Mock Ground Truth
    truth_rows = [
        {
            "video_id": "video_1",
            "event_type": "PHONE",
            "occupant_id": "v1:occupant-track:1",
            "occupant_role": "driver",
            "vehicle_id": "v1",
            "cabin_id": "c1",
            "inside_vehicle": "true",
            "outside_vehicle_person": "false",
            "motorcycle_flag": "false",
            "label": "PHONE_USE",
            "start_seconds": "1.0",
            "end_seconds": "2.5"
        }
    ]
    
    # 6. Evaluate
    report = evaluate(
        truth_rows=truth_rows,
        prediction_rows=prediction_rows,
        video_minutes=1.0,
        prediction_fieldnames=pred_fields,
        context_rows=[_context_for_prediction(prediction_rows[0])],
    )
    
    # 7. Assert contract integrity
    assert report["safety_invariant_counters"] != "NOT_EVALUABLE", "Metadata from exporter failed validation"
    assert report["event_types"]["PHONE"]["true_positives"] >= 1, "Expected matching TP for the predicted event"
    assert report["safety_invariant_counters"]["single_frame_violation_count"] == 0, "Should not be a single-frame violation"

def test_video_analyzer_metadata_contract():
    from backend.services.video_analyzer import VideoAnalyzer, InferenceContext
    from backend.ai.detector import NormalizedDetection
    from types import SimpleNamespace
    import numpy as np
    from collections import defaultdict

    analyzer = VideoAnalyzer.__new__(VideoAnalyzer)
    analyzer.detector = SimpleNamespace(
        predict=lambda *args, **kwargs: [NormalizedDetection(0, "phone", 0.9, (10, 10, 20, 20), 1)],
        model_version="V2_TEST"
    )
    analyzer.local_tracker = SimpleNamespace(
        update=lambda *args: [NormalizedDetection(0, "phone", 0.9, (10, 10, 20, 20), 1)]
    )
    analyzer.occupants = SimpleNamespace(
        assign_upper_body=lambda *args: SimpleNamespace(role="driver", confidence=0.9, method="test", occupant_track_id=42),
        assign_object=lambda *args: SimpleNamespace(role="driver", confidence=0.9, method="test", occupant_track_id=42)
    )
    analyzer.pose_interval = 100
    analyzer.use_fusion = False
    analyzer.sequence = SimpleNamespace(
        add=lambda *args: SimpleNamespace(ratio=1.0, positive_ratio=1.0, duration_seconds=1.0, maximum_gap_seconds=0.0)
    )
    analyzer._feature_history = defaultdict(list)
    analyzer.runtime_status = lambda: {"fusion": {"fusion_mode": "DISABLED"}}

    observations_captured = []
    engine = SimpleNamespace(
        add=lambda obs: observations_captured.append(obs) or []
    )

    session = SimpleNamespace(add=lambda x: None, flush=lambda: None)
    job = SimpleNamespace(id="j1")
    video = SimpleNamespace(id="v1")
    context = InferenceContext(
        context_id="v1:car-1:cabin:0",
        vehicle_id="v1:car-1",
        cabin_id="v1:car-1:cabin:0",
        cabin=np.zeros((10, 10, 3)),
        offset=(0, 0),
        vehicle_track_id=1,
        vehicle_type="car",
        vehicle_class_id=1,
        vehicle_confidence=0.9,
        cabin_confidence=0.9,
        cabin_method="KNOWN_CABIN",
        vehicle_bbox=(0.0, 0.0, 100.0, 100.0),
        cabin_bbox=(10.0, 10.0, 90.0, 90.0)
    )

    analyzer._analyze_context(
        session, job, video, np.zeros((100, 100, 3)), 0, 1.0,
        context, engine, SimpleNamespace(register=lambda *args: None), {}
    )

    assert len(observations_captured) == 1
    obs = observations_captured[0]
    
    assert obs.cabin_id == "v1:car-1:cabin:0"
    assert obs.visibility == "unknown"
    assert obs.outside_vehicle_person == ""
    assert obs.behavior_label == "PHONE_USE"
    assert obs.occupant_id == "v1:car-1:cabin:0:occupant-track:42"

def test_end_to_end_seatbelt_evaluation(tmp_path: Path):
    engine = TemporalEventEngine(window_seconds=2.0)
    
    observations = [
        Observation(
            timestamp=1.0, 
            class_name="seatbelt_unfastened", 
            confidence=0.8, 
            track_id=1, 
            occupant_role="driver", 
            vehicle_context_id="v1",
            cabin_id="c1",
            visibility="clear",
            outside_vehicle_person="false",
            behavior_label="UNFASTENED",
            vehicle_type="car",
            seatbelt_probabilities=[0.1, 0.8, 0.1],
            occupant_id="v1:occupant-track:1"
        ),
        Observation(
            timestamp=1.5, 
            class_name="seatbelt_unfastened", 
            confidence=0.85, 
            track_id=1, 
            occupant_role="driver", 
            vehicle_context_id="v1",
            cabin_id="c1",
            visibility="clear",
            outside_vehicle_person="false",
            behavior_label="UNFASTENED",
            vehicle_type="car",
            seatbelt_probabilities=[0.1, 0.85, 0.05],
            occupant_id="v1:occupant-track:1"
        ),
        Observation(
            timestamp=2.0, 
            class_name="seatbelt_unfastened", 
            confidence=0.9, 
            track_id=1, 
            occupant_role="driver", 
            vehicle_context_id="v1",
            cabin_id="c1",
            visibility="clear",
            outside_vehicle_person="false",
            behavior_label="UNFASTENED",
            vehicle_type="car",
            seatbelt_probabilities=[0.05, 0.9, 0.05],
            occupant_id="v1:occupant-track:1"
        ),
        Observation(
            timestamp=2.5, 
            class_name="seatbelt_unfastened", 
            confidence=0.95, 
            track_id=1, 
            occupant_role="driver", 
            vehicle_context_id="v1",
            cabin_id="c1",
            visibility="clear",
            outside_vehicle_person="false",
            behavior_label="UNFASTENED",
            vehicle_type="car",
            seatbelt_probabilities=[0.01, 0.95, 0.04],
            occupant_id="v1:occupant-track:1"
        ),
    ]
    
    candidates = []
    for obs in observations:
        candidates.extend(engine.add(obs))
        
    assert len(candidates) > 0, "TemporalEventEngine should produce at least one candidate"
    assert candidates[0].event_type == "NO_SEATBELT"
    assert candidates[0].behavior_label == "UNFASTENED"
    
    csv_path = tmp_path / "predictions_seatbelt.csv"
    export_predictions_to_csv(candidates, video_id="video_1", output_path=csv_path)
    
    assert csv_path.exists()
    
    prediction_rows, pred_fields = _read(csv_path)
    assert len(prediction_rows) == 1
    assert prediction_rows[0]["event_type"] == "NO_SEATBELT"
    assert prediction_rows[0]["label"] == "UNFASTENED"
    
    truth_rows = [
        {
            "video_id": "video_1",
            "event_type": "NO_SEATBELT",
            "occupant_id": "v1:occupant-track:1",
            "occupant_role": "driver",
            "vehicle_id": "v1",
            "cabin_id": "c1",
            "inside_vehicle": "true",
            "outside_vehicle_person": "false",
            "motorcycle_flag": "false",
            "label": "UNFASTENED",
            "start_seconds": "1.0",
            "end_seconds": "2.5"
        }
    ]
    
    report = evaluate(
        truth_rows=truth_rows,
        prediction_rows=prediction_rows,
        video_minutes=1.0,
        prediction_fieldnames=pred_fields,
        context_rows=[_context_for_prediction(prediction_rows[0])],
    )
    
    assert report["safety_invariant_counters"] != "NOT_EVALUABLE", "Metadata from exporter failed validation"
    assert report["event_types"]["NO_SEATBELT"]["true_positives"] >= 1, "Expected matching TP for the predicted event"

def test_role_misclassification_identity_preservation():
    # 3. Giữ identity ổn định qua role misclassification
    engine = TemporalEventEngine(window_seconds=2.0)
    
    # Upper-body track ID is 42. Behavior (phone) track is 7 then 9. 
    # Role is passenger then driver. 
    # occupant_id should be stable based on 42.
    obs1 = Observation(
        timestamp=1.0, class_name="phone", confidence=0.9, track_id=7, occupant_role="front_passenger", 
        vehicle_context_id="v1", cabin_id="c1", phone_context="HANDHELD", behavior_label="PHONE_USE", 
        occupant_id="c1:occupant-track:42"
    )
    obs2 = Observation(
        timestamp=2.0, class_name="phone", confidence=0.9, track_id=9, occupant_role="driver", 
        vehicle_context_id="v1", cabin_id="c1", phone_context="HANDHELD", behavior_label="PHONE_USE", 
        occupant_id="c1:occupant-track:42"
    )
    obs3 = Observation(
        timestamp=3.0, class_name="phone", confidence=0.9, track_id=9, occupant_role="driver", 
        vehicle_context_id="v1", cabin_id="c1", phone_context="HANDHELD", behavior_label="PHONE_USE", 
        occupant_id="c1:occupant-track:42"
    )
    obs4 = Observation(
        timestamp=4.0, class_name="phone", confidence=0.9, track_id=9, occupant_role="driver", 
        vehicle_context_id="v1", cabin_id="c1", phone_context="HANDHELD", behavior_label="PHONE_USE", 
        occupant_id="c1:occupant-track:42"
    )

    # Engine processes these
    engine.add(obs1)
    engine.add(obs2)
    engine.add(obs3)
    candidates = engine.add(obs4)
    
    assert len(candidates) > 0
    candidate = candidates[0]
    
    # identity preserved regardless of the last assigned role or the behavior track changing from 7 to 9
    assert candidate.occupant_id == "c1:occupant-track:42"
    assert candidate.occupant_role == "driver"

def test_engine_reset_vehicle():
    engine = TemporalEventEngine(window_seconds=2.0)
    obs1 = Observation(
        timestamp=1.0, class_name="phone", confidence=0.9, track_id=1, occupant_role="driver", 
        vehicle_context_id="v1", cabin_id="c1", phone_context="HANDHELD", behavior_label="PHONE_USE", 
        occupant_id="c1:occupant-track:1"
    )
    engine.add(obs1)
    
    # Verify state exists
    assert any(key[0] == "v1" for key in engine.windows)
    
    # Reset vehicle
    engine.reset_vehicle("v1")
    
    # Verify state cleared
    assert not any(key[0] == "v1" for key in engine.windows)
    assert not any(key[0] == "v1" for key in engine.smoothed)
    
def test_needs_review_not_promoted(tmp_path: Path):
    engine = TemporalEventEngine(window_seconds=2.0)
    
    observations = [
        Observation(
            timestamp=1.0, class_name="seatbelt_unfastened", confidence=0.8, track_id=2, occupant_role="driver", 
            vehicle_context_id="v1", cabin_id="c1", phone_context="UNKNOWN", behavior_label="UNFASTENED", occupant_id="c1:occupant-track:2",
            evidence_source="DETECTOR_CLASSIFIER_CONFLICT"
        ),
        Observation(
            timestamp=2.0, class_name="seatbelt_unfastened", confidence=0.85, track_id=2, occupant_role="driver", 
            vehicle_context_id="v1", cabin_id="c1", phone_context="UNKNOWN", behavior_label="UNFASTENED", occupant_id="c1:occupant-track:2",
            evidence_source="DETECTOR_CLASSIFIER_CONFLICT"
        ),
        Observation(
            timestamp=3.0, class_name="seatbelt_unfastened", confidence=0.9, track_id=2, occupant_role="driver", 
            vehicle_context_id="v1", cabin_id="c1", phone_context="UNKNOWN", behavior_label="UNFASTENED", occupant_id="c1:occupant-track:2",
            evidence_source="DETECTOR_CLASSIFIER_CONFLICT"
        )
    ]
    
    candidates = []
    for obs in observations:
        candidates.extend(engine.add(obs))
        
    assert candidates[0].review_status == "NEEDS_REVIEW"
    
    csv_path = tmp_path / "predictions_needs_review.csv"
    export_predictions_to_csv(candidates, video_id="video_1", output_path=csv_path)
    
    prediction_rows, pred_fields = _read(csv_path)
    # Exporter should exclude NEEDS_REVIEW entirely
    assert len(prediction_rows) == 0

