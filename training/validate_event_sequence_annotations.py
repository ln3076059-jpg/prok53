import json
import argparse
from pathlib import Path
from jsonschema import Draft7Validator, FormatChecker

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
        
    occupants = annotation.get('occupants', [])
    valid_occupant_ids = {occ['occupant_id'] for occ in occupants}
    
    for occ in occupants:
        role_conf = occ.get('role_confidence')
        if role_conf is not None:
            if not (0 <= role_conf <= 1):
                errors.append(f"Semantic error: occupant {occ['occupant_id']} role_confidence must be between 0 and 1, got {role_conf}")
                
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
            if not (0 <= start_frame <= end_frame < frame_count):
                errors.append(f"Semantic error: event {idx} frames invalid (0 <= {start_frame} <= {end_frame} < {frame_count} is False)")
                
        if start_time_sec is not None and end_time_sec is not None:
            # allow some float tolerance for end_time_sec <= duration_sec
            if not (0 <= start_time_sec <= end_time_sec and end_time_sec <= duration_sec + 1e-4):
                errors.append(f"Semantic error: event {idx} times invalid (0 <= {start_time_sec} <= {end_time_sec} <= {duration_sec} is False)")

    # 3. Provenance Validation
    provenance = annotation.get('review_provenance', {})
    if provenance.get('status') != 'HUMAN_APPROVED':
        errors.append(f"Semantic error: review_provenance status must be HUMAN_APPROVED, got {provenance.get('status')}")
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
