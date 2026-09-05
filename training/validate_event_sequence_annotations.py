import json
import argparse
from pathlib import Path
from jsonschema import Draft7Validator, FormatChecker
import dateutil.parser

def load_schema(schema_path):
    with open(schema_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def validate_annotation(annotation, schema):
    errors = []
    
    # 1. Schema Validation
    validator = Draft7Validator(schema, format_checker=FormatChecker())
    schema_errors = list(validator.iter_errors(annotation))
    if schema_errors:
        for err in schema_errors:
            path_str = ".".join(str(p) for p in err.absolute_path) if err.absolute_path else "root"
            errors.append(f"Schema error: {err.message} at path {path_str}")
        return errors
        
    # 2. Semantic Validation
    fps = annotation.get('fps', 0)
    frame_count = annotation.get('frame_count', 0)
    
    if fps <= 0:
        errors.append(f"Semantic error: fps must be > 0, got {fps}")
    if frame_count <= 0:
        errors.append(f"Semantic error: frame_count must be > 0, got {frame_count}")
        
    # Sequence time validation
    start_time_str = annotation.get('start_time')
    end_time_str = annotation.get('end_time')
    if start_time_str and end_time_str:
        try:
            st = dateutil.parser.parse(start_time_str)
            et = dateutil.parser.parse(end_time_str)
            if st >= et:
                errors.append(f"Semantic error: sequence start_time {start_time_str} must be < end_time {end_time_str}")
        except Exception:
            errors.append(f"Semantic error: invalid date format in start_time or end_time")
        
    occupants = annotation.get('occupants', [])
    valid_occupant_ids = set()
    for occ in occupants:
        oid = occ.get('occupant_id')
        if oid in valid_occupant_ids:
            errors.append(f"Semantic error: duplicate occupant_id '{oid}'")
        else:
            valid_occupant_ids.add(oid)
            
        role_conf = occ.get('role_confidence')
        if role_conf is not None:
            if not (0 <= role_conf <= 1):
                errors.append(f"Semantic error: occupant {oid} role_confidence must be between 0 and 1, got {role_conf}")
                
    events = annotation.get('events', [])
    duration_sec = frame_count / fps if fps > 0 else 0
    
    for idx, event in enumerate(events):
        event_type = event.get('event_type')
        occ_id = event.get('occupant_id')
        start_frame = event.get('start_frame')
        end_frame = event.get('end_frame')
        start_time_sec = event.get('start_time_sec')
        end_time_sec = event.get('end_time_sec')
        
        if occ_id not in valid_occupant_ids:
            errors.append(f"Semantic error: event {idx} occupant_id '{occ_id}' not found in occupants")
            
        if start_frame is not None and end_frame is not None:
            if not (0 <= start_frame <= end_frame):
                errors.append(f"Semantic error: event {idx} frames invalid (start_frame {start_frame} > end_frame {end_frame})")
            if end_frame >= frame_count:
                errors.append(f"Semantic error: event {idx} frames invalid (end_frame {end_frame} >= frame_count {frame_count})")
                
        if start_time_sec is not None and end_time_sec is not None:
            if not (0 <= start_time_sec <= end_time_sec):
                errors.append(f"Semantic error: event {idx} times invalid (start_time {start_time_sec} > end_time {end_time_sec})")
            if end_time_sec > duration_sec + 1e-4:
                errors.append(f"Semantic error: event {idx} times invalid (end_time_sec {end_time_sec} > duration {duration_sec})")
                
        # Check frame boundaries vs time boundaries
        if start_frame is not None and start_time_sec is not None and fps > 0:
            expected_start_time = start_frame / fps
            if abs(expected_start_time - start_time_sec) > 0.1:
                errors.append(f"Semantic error: event {idx} start_time_sec {start_time_sec} does not match start_frame {start_frame} at {fps} fps")
                
        if end_frame is not None and end_time_sec is not None and fps > 0:
            expected_end_time = end_frame / fps
            if abs(expected_end_time - end_time_sec) > 0.1:
                errors.append(f"Semantic error: event {idx} end_time_sec {end_time_sec} does not match end_frame {end_frame} at {fps} fps")

    # 3. Provenance Validation
    provenance = annotation.get('review_provenance', {})
    if provenance.get('status') != 'HUMAN_APPROVED':
        errors.append(f"Semantic error: review_provenance status must be HUMAN_APPROVED, got {provenance.get('status')}. AI proposals are not valid for governed calibration.")
    else:
        required_fields = ['reviewer_id', 'reviewed_at', 'evidence_hash']
        for field in required_fields:
            if not provenance.get(field):
                errors.append(f"Semantic error: review_provenance missing {field} for HUMAN_APPROVED")

    return errors

def main():
    parser = argparse.ArgumentParser(description="Validate event sequence annotations.")
    parser.add_argument('annotation_file', help="Path to annotation JSON file")
    parser.add_argument('--schema', default='datasets/schemas/v2_event_sequence_annotation.schema.json', help="Path to schema JSON file")
    args = parser.parse_args()
    
    schema = load_schema(args.schema)
    with open(args.annotation_file, 'r', encoding='utf-8') as f:
        annotation = json.load(f)
        
    errors = validate_annotation(annotation, schema)
    if errors:
        print("Validation failed with the following errors:")
        for err in errors:
            print(" -", err)
        exit(1)
    else:
        print("Validation PASS")
        exit(0)

if __name__ == "__main__":
    main()
