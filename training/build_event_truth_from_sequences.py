import json
import csv
import argparse
import uuid
import sys
from pathlib import Path

def convert_sequence_to_events(annotation):
    emitted_events = []
    
    video_id = annotation.get("video_id")
    if not video_id:
        return emitted_events
        
    context = annotation.get("context", {})
    motorcycle_flag = context.get("motorcycle_flag", False)
    inside_vehicle = context.get("inside_vehicle", True)
    
    occupants = annotation.get("occupants", [])
    occ_map = {occ["occupant_id"]: occ for occ in occupants}
    
    valid_roles = {"driver", "front_passenger", "rear_left", "rear_center", "rear_right"}
    
    events = annotation.get("events", [])
    
    for idx, evt in enumerate(events):
        event_type = evt.get("event_type")
        label = evt.get("label")
        occ_id = evt.get("occupant_id")
        start_sec = evt.get("start_time_sec")
        end_sec = evt.get("end_time_sec")
        
        occ = occ_map.get(occ_id, {})
        role = occ.get("role", "unknown")
        
        final_event_type = None
        
        if event_type == "PHONE":
            if label == "PHONE_USE" and role == "driver":
                final_event_type = "PHONE"
        elif event_type == "SEATBELT":
            if label == "UNFASTENED":
                if not motorcycle_flag and inside_vehicle and role in valid_roles:
                    final_event_type = "NO_SEATBELT"
                    
        if final_event_type:
            event_id = f"{video_id}_evt_{idx}"
            emitted_events.append({
                "video_id": video_id,
                "event_id": event_id,
                "event_type": final_event_type,
                "start_seconds": start_sec,
                "end_seconds": end_sec
            })
            
    return emitted_events

def process_file(json_path, csv_writer):
    with open(json_path, 'r', encoding='utf-8') as f:
        annotation = json.load(f)
        
    events = convert_sequence_to_events(annotation)
    for evt in events:
        csv_writer.writerow([
            evt["video_id"],
            evt["event_id"],
            evt["event_type"],
            evt["start_seconds"],
            evt["end_seconds"]
        ])
        
def main():
    parser = argparse.ArgumentParser(description="Convert sequence JSON annotations to final event CSV")
    parser.add_argument("input_path", help="Path to input JSON file or directory")
    parser.add_argument("output_csv", help="Path to output CSV file")
    
    args = parser.parse_args()
    
    input_p = Path(args.input_path)
    if not input_p.exists():
        print(f"Error: {args.input_path} does not exist.")
        sys.exit(1)
        
    files_to_process = []
    if input_p.is_file():
        files_to_process.append(input_p)
    else:
        files_to_process.extend(input_p.glob("*.json"))
        
    with open(args.output_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["video_id", "event_id", "event_type", "start_seconds", "end_seconds"])
        
        for p in files_to_process:
            process_file(p, writer)
            
    print(f"Converted {len(files_to_process)} files to {args.output_csv}")

if __name__ == "__main__":
    main()
