from pathlib import Path
from backend.ai.events import TemporalEventEngine, Observation
from training.export_event_predictions import export_predictions_to_csv
from training.evaluate_events import evaluate, _read

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
            phone_context="HANDHELD"
        ),
        Observation(
            timestamp=1.5, 
            class_name="phone", 
            confidence=0.85, 
            track_id=1, 
            occupant_role="driver", 
            vehicle_context_id="v1",
            phone_context="HANDHELD"
        ),
        Observation(
            timestamp=2.0, 
            class_name="phone", 
            confidence=0.9, 
            track_id=1, 
            occupant_role="driver", 
            vehicle_context_id="v1",
            phone_context="HANDHELD"
        ),
        Observation(
            timestamp=2.5, 
            class_name="phone", 
            confidence=0.95, 
            track_id=1, 
            occupant_role="driver", 
            vehicle_context_id="v1",
            phone_context="HANDHELD"
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
    
    # 4. Mock Ground Truth
    truth_rows = [
        {
            "video_id": "video_1",
            "event_type": "PHONE",
            "occupant_role": "driver",
            "vehicle_id": "v1",
            "cabin_id": "v1",
            "start_seconds": "1.0",
            "end_seconds": "2.5"
        }
    ]
    
    # 5. Evaluate
    report = evaluate(
        truth_rows=truth_rows,
        prediction_rows=prediction_rows,
        video_minutes=1.0,
        prediction_fieldnames=pred_fields
    )
    
    # 6. Assert contract integrity
    assert report["safety_invariant_counters"] != "NOT_EVALUABLE", "Metadata from exporter failed validation"
    assert report["event_types"]["PHONE"]["true_positives"] >= 1, "Expected matching TP for the predicted event"
