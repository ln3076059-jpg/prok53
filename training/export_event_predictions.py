import csv
from collections.abc import Iterable
from pathlib import Path
from backend.ai.events import EventCandidate

def export_predictions_to_csv(
    candidates: Iterable[EventCandidate], 
    video_id: str,
    output_path: Path,
) -> None:
    """
    Exports a stream of runtime EventCandidate objects into an evaluator-compatible 
    CSV format. Provides fallback and synthetic fields where the runtime does not 
    yet track them natively.
    """
    fieldnames = [
        "video_id", "event_type", "occupant_role", "vehicle_id", "cabin_id", 
        "start_seconds", "end_seconds", "label", "visibility", 
        "outside_vehicle_person", "motorcycle_flag", "observation_count"
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        
        for candidate in candidates:
            if candidate.review_status == "NEEDS_REVIEW":
                continue
                
            if candidate.vehicle_type == "motorcycle":
                motorcycle_flag = "true"
            elif candidate.vehicle_type != "unknown" and candidate.vehicle_type != "":
                motorcycle_flag = "false"
            else:
                motorcycle_flag = ""
            
            writer.writerow({
                "video_id": video_id,
                "event_type": candidate.event_type,
                "occupant_role": candidate.occupant_role,
                "vehicle_id": candidate.vehicle_context_id,
                "cabin_id": candidate.cabin_id,
                "start_seconds": f"{candidate.start_timestamp:.3f}",
                "end_seconds": f"{candidate.end_timestamp:.3f}",
                "label": candidate.behavior_label,
                "visibility": candidate.visibility,
                "outside_vehicle_person": candidate.outside_vehicle_person,
                "motorcycle_flag": motorcycle_flag,
                "observation_count": candidate.observation_count,
            })
