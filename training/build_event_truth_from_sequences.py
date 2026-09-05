import json
import csv
import argparse
import sys
import os
from pathlib import Path
from training.validate_event_sequence_annotations import validate_annotation, load_schema

def convert_sequence_to_events(annotation):
    emitted_events = []
    
    video_id = annotation.get("video_id")
    sequence_id = annotation.get("sequence_id", "unknown_seq")
    fps = annotation.get("fps", 30.0)
    
    if not video_id:
        return emitted_events
        
    context = annotation.get("context", {})
    # Fail-closed semantics for context
    motorcycle_flag = context.get("motorcycle_flag")
    inside_vehicle = context.get("inside_vehicle")
    outside_vehicle_person = context.get("outside_vehicle_person")
    
    vehicle_id = annotation.get("vehicle_id") or "UNKNOWN"
    cabin_id = "UNKNOWN"
    
    provenance = annotation.get("review_provenance", {})
    prov_status = provenance.get("status", "")
    human_review_status = "APPROVED" if prov_status == "HUMAN_APPROVED" else prov_status
    reviewer_id = provenance.get("reviewer_id", "")
    reviewer_type = provenance.get("reviewer_type", "")
    reviewed_at = provenance.get("reviewed_at", "")
    adjudication_status = provenance.get("adjudication_status")
    
    if adjudication_status != "FINAL":
        raise ValueError(f"Annotation adjudication_status is not FINAL (got {adjudication_status})")
        
    notes = ""
    visibility = context.get("day_night", "unknown")
    conditions = context.get("occlusion", "none")
    
    occupants = annotation.get("occupants", [])
    occ_map = {occ["occupant_id"]: occ for occ in occupants}
    
    valid_roles = {"driver", "front_passenger", "rear_left", "rear_center", "rear_right"}
    
    events = annotation.get("events", [])
    
    for idx, evt in enumerate(events):
        event_type = evt.get("event_type")
        label = evt.get("label")
        occ_id = evt.get("occupant_id")
        
        start_frame = evt.get("start_frame", 0)
        end_frame = evt.get("end_frame", 0)
        
        start_sec = evt.get("start_time_sec")
        if start_sec is None:
            start_sec = start_frame / fps if fps > 0 else 0
            
        end_sec = evt.get("end_time_sec")
        if end_sec is None:
            end_sec = end_frame / fps if fps > 0 else 0
        
        occ = occ_map.get(occ_id, {})
        role = occ.get("role", "unknown")
        
        final_event_type = None
        
        if event_type == "PHONE":
            if label in ("PHONE_USE", "MOUNTED_OR_STATIC_PHONE", "PHONE_PRESENT_NOT_USED", "NO_PHONE", "UNKNOWN"):
                final_event_type = "PHONE"
        elif event_type == "SEATBELT":
            if label in ("FASTENED", "UNFASTENED", "UNCERTAIN_OR_OCCLUDED", "NOT_APPLICABLE"):
                final_event_type = "NO_SEATBELT"
                
        if final_event_type:
            event_id = f"{video_id}_{sequence_id}_evt_{idx}"
            emitted_events.append({
                "video_id": video_id,
                "event_id": event_id,
                "event_type": final_event_type,
                "start_seconds": start_sec,
                "end_seconds": end_sec,
                "vehicle_id": vehicle_id,
                "cabin_id": cabin_id,
                "occupant_role": role,
                "inside_vehicle": str(inside_vehicle).lower() if inside_vehicle is not None else "",
                "outside_vehicle_person": str(outside_vehicle_person).lower() if outside_vehicle_person is not None else "",
                "motorcycle_flag": str(motorcycle_flag).lower() if motorcycle_flag is not None else "",
                "label": label,
                "visibility": visibility,
                "conditions": conditions,
                "human_review_status": human_review_status,
                "reviewer_id": reviewer_id,
                "reviewer_type": reviewer_type,
                "reviewed_at": reviewed_at,
                "adjudication_status": adjudication_status,
                "notes": notes
            })
            
    return emitted_events

def process_file(json_path, csv_writer, schema):
    with open(json_path, 'r', encoding='utf-8') as f:
        annotation = json.load(f)
        
    errors = validate_annotation(annotation, schema)
    if errors:
        print(f"Skipping {json_path} due to schema/semantic validation errors:")
        for e in errors:
            print(f" - {e}")
        return False
        
    try:
        events = convert_sequence_to_events(annotation)
    except ValueError as e:
        print(f"Skipping {json_path} due to conversion error: {e}")
        return False
        
    for evt in events:
        csv_writer.writerow([
            evt["video_id"],
            evt["event_id"],
            evt["event_type"],
            evt["start_seconds"],
            evt["end_seconds"],
            evt["vehicle_id"],
            evt["cabin_id"],
            evt["occupant_role"],
            evt["inside_vehicle"],
            evt["outside_vehicle_person"],
            evt["motorcycle_flag"],
            evt["label"],
            evt["visibility"],
            evt["conditions"],
            evt["human_review_status"],
            evt["reviewer_id"],
            evt["reviewer_type"],
            evt["reviewed_at"],
            evt["adjudication_status"],
            evt["notes"]
        ])
    return True

def main():
    parser = argparse.ArgumentParser(description="Convert sequence JSON annotations to final event CSV")
    parser.add_argument("input_path", help="Path to input JSON file or directory")
    parser.add_argument("output_csv", help="Path to output CSV file")
    parser.add_argument("--schema", default="datasets/schemas/v2_event_sequence_annotation.schema.json", help="Path to sequence schema JSON")
    
    args = parser.parse_args()
    
    input_p = Path(args.input_path)
    if not input_p.exists():
        print(f"Error: {args.input_path} does not exist.")
        sys.exit(1)
        
    schema = load_schema(args.schema)
        
    files_to_process = []
    if input_p.is_file():
        files_to_process.append(input_p)
    else:
        files_to_process.extend(input_p.glob("*.json"))
        
    success_count = 0
    temp_csv = str(args.output_csv) + ".tmp"
    with open(temp_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "video_id", "event_id", "event_type", "start_seconds", "end_seconds",
            "vehicle_id", "cabin_id", "occupant_role",
            "inside_vehicle", "outside_vehicle_person",
            "motorcycle_flag", "label",
            "visibility", "conditions",
            "human_review_status", "reviewer_id", "reviewer_type", "reviewed_at",
            "adjudication_status", "notes"
        ])
        
        for p in files_to_process:
            if process_file(p, writer, schema):
                success_count += 1
            
    if success_count < len(files_to_process):
        print(f"Failed to process {len(files_to_process) - success_count} files. Aborting export.")
        if os.path.exists(temp_csv):
            os.remove(temp_csv)
        sys.exit(1)
    else:
        print(f"Successfully converted {success_count}/{len(files_to_process)} files to {args.output_csv}")
        os.replace(temp_csv, args.output_csv)

if __name__ == "__main__":
    main()
