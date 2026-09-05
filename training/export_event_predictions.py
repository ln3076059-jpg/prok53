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
            # 1. Provide safe defaults for labels
            if candidate.event_type == "PHONE":
                label = "PHONE_USE"
            elif candidate.event_type == "NO_SEATBELT":
                label = "UNFASTENED"
            else:
                label = "UNKNOWN"
                
            # 2. Derive motorcycle flag from runtime vehicle type
            motorcycle_flag = "true" if candidate.vehicle_type == "motorcycle" else "false"
            
            # 3. Handle single-frame evaluation duration since candidates are instantaneous
            # We set observation_count = 1 to signal true single-frame semantics to evaluator
            
            writer.writerow({
                "video_id": video_id,
                "event_type": candidate.event_type,
                "occupant_role": candidate.occupant_role,
                "vehicle_id": candidate.vehicle_context_id,
                "cabin_id": candidate.vehicle_context_id,  # Runtime doesn't split these natively yet
                "start_seconds": f"{candidate.timestamp:.3f}",
                "end_seconds": f"{candidate.timestamp:.3f}",
                "label": label,
                "visibility": "clear",
                "outside_vehicle_person": "false",
                "motorcycle_flag": motorcycle_flag,
                "observation_count": 1,
            })
