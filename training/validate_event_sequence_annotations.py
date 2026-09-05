import argparse
import json

import dateutil.parser
from jsonschema import Draft7Validator, FormatChecker

UNKNOWN_IDENTITIES = {"", "UNKNOWN"}


def load_schema(schema_path):
    with open(schema_path, encoding="utf-8") as f:
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
    fps = annotation.get("fps", 0)
    frame_count = annotation.get("frame_count", 0)

    if fps <= 0:
        errors.append(f"Semantic error: fps must be > 0, got {fps}")
    if frame_count <= 0:
        errors.append(f"Semantic error: frame_count must be > 0, got {frame_count}")

    # Sequence time validation
    start_time_str = annotation.get("start_time")
    end_time_str = annotation.get("end_time")
    if start_time_str and end_time_str:
        try:
            st = dateutil.parser.parse(start_time_str)
            et = dateutil.parser.parse(end_time_str)
            if st >= et:
                errors.append(
                    f"Semantic error: sequence start_time {start_time_str} must be < "
                    f"end_time {end_time_str}"
                )
        except Exception:
            errors.append("Semantic error: invalid date format in start_time or end_time")

    context = annotation.get("context", {})
    if context.get("inside_vehicle") is True and context.get("outside_vehicle_person") is True:
        errors.append(
            "Semantic error: contradictory context - inside_vehicle and "
            "outside_vehicle_person cannot both be True"
        )

    occupants = annotation.get("occupants", [])
    valid_occupant_ids = set()
    for occ in occupants:
        oid = occ.get("occupant_id")
        if oid in valid_occupant_ids:
            errors.append(f"Semantic error: duplicate occupant_id '{oid}'")
        else:
            valid_occupant_ids.add(oid)

        role_conf = occ.get("role_confidence")
        if role_conf is not None:
            if not (0 <= role_conf <= 1):
                errors.append(
                    f"Semantic error: occupant {oid} role_confidence must be between "
                    f"0 and 1, got {role_conf}"
                )

    for field in ("vehicle_id", "cabin_id"):
        if str(annotation.get(field, "")).strip().upper() in UNKNOWN_IDENTITIES:
            errors.append(f"Semantic error: {field} must be a known stable identity")

    events = annotation.get("events", [])
    duration_sec = frame_count / fps if fps > 0 else 0

    for idx, event in enumerate(events):
        occ_id = event.get("occupant_id")
        start_frame = event.get("start_frame")
        end_frame = event.get("end_frame")
        start_time_sec = event.get("start_time_sec")
        end_time_sec = event.get("end_time_sec")

        if occ_id not in valid_occupant_ids:
            errors.append(
                f"Semantic error: event {idx} occupant_id '{occ_id}' not found in occupants"
            )

        if start_frame is not None and end_frame is not None:
            if not (0 <= start_frame <= end_frame):
                errors.append(
                    f"Semantic error: event {idx} frames invalid "
                    f"(start_frame {start_frame} > end_frame {end_frame})"
                )
            if end_frame >= frame_count:
                errors.append(
                    f"Semantic error: event {idx} frames invalid "
                    f"(end_frame {end_frame} >= frame_count {frame_count})"
                )

        if start_time_sec is not None and end_time_sec is not None:
            if not (0 <= start_time_sec <= end_time_sec):
                errors.append(
                    f"Semantic error: event {idx} times invalid "
                    f"(start_time {start_time_sec} > end_time {end_time_sec})"
                )
            if end_time_sec > duration_sec + 1e-4:
                errors.append(
                    f"Semantic error: event {idx} times invalid "
                    f"(end_time_sec {end_time_sec} > duration {duration_sec})"
                )

        # Check frame boundaries vs time boundaries
        if start_frame is not None and start_time_sec is not None and fps > 0:
            expected_start_time = start_frame / fps
            if abs(expected_start_time - start_time_sec) > 0.1:
                errors.append(
                    f"Semantic error: event {idx} start_time_sec {start_time_sec} "
                    f"does not match start_frame {start_frame} at {fps} fps"
                )

        if end_frame is not None and end_time_sec is not None and fps > 0:
            expected_end_time = end_frame / fps
            if abs(expected_end_time - end_time_sec) > 0.1:
                errors.append(
                    f"Semantic error: event {idx} end_time_sec {end_time_sec} "
                    f"does not match end_frame {end_frame} at {fps} fps"
                )

    # Context intervals use half-open frame ranges and must cover every declared
    # occupant for the complete sequence. This makes background false positives
    # independently evaluable instead of borrowing context from sparse events.
    intervals_by_occupant = {occupant_id: [] for occupant_id in valid_occupant_ids}
    context_ids = set()
    for idx, interval in enumerate(annotation.get("context_intervals", [])):
        context_id = interval.get("context_id")
        occupant_id = interval.get("occupant_id")
        start_frame = interval.get("start_frame")
        end_frame = interval.get("end_frame")

        if context_id in context_ids:
            errors.append(f"Semantic error: duplicate context_id '{context_id}'")
        context_ids.add(context_id)
        if occupant_id not in valid_occupant_ids:
            errors.append(
                f"Semantic error: context interval {idx} occupant_id '{occupant_id}' "
                "not found in occupants"
            )
            continue
        if not isinstance(start_frame, int) or not isinstance(end_frame, int):
            continue
        if not (0 <= start_frame < end_frame <= frame_count):
            errors.append(
                f"Semantic error: context interval {idx} must satisfy "
                f"0 <= start_frame < end_frame <= frame_count ({frame_count})"
            )
            continue
        if (
            interval.get("inside_vehicle") is True
            and interval.get("outside_vehicle_person") is True
        ):
            errors.append(
                f"Semantic error: context interval {idx} cannot be both inside and outside vehicle"
            )
        intervals_by_occupant[occupant_id].append((start_frame, end_frame, context_id))

    for occupant_id, intervals in intervals_by_occupant.items():
        cursor = 0
        for start_frame, end_frame, context_id in sorted(intervals):
            if start_frame != cursor:
                relation = "overlap" if start_frame < cursor else "gap"
                errors.append(
                    f"Semantic error: {relation} in context coverage for occupant "
                    f"'{occupant_id}' before context_id '{context_id}': expected start_frame "
                    f"{cursor}, got {start_frame}"
                )
            cursor = max(cursor, end_frame)
        if cursor != frame_count:
            errors.append(
                f"Semantic error: incomplete context coverage for occupant '{occupant_id}': "
                f"ends at frame {cursor}, expected {frame_count}"
            )

    raw_context_intervals = annotation.get("context_intervals", [])
    for idx, event in enumerate(events):
        occupant_id = event.get("occupant_id")
        start_frame = event.get("start_frame")
        end_frame = event.get("end_frame")
        label = event.get("label")
        if not isinstance(start_frame, int) or not isinstance(end_frame, int):
            continue
        state_field = "phone_state" if event.get("event_type") == "PHONE" else "seatbelt_state"
        overlapping_context = [
            interval
            for interval in raw_context_intervals
            if interval.get("occupant_id") == occupant_id
            and interval.get("start_frame", frame_count) < end_frame + 1
            and interval.get("end_frame", 0) > start_frame
        ]
        if any(interval.get(state_field) != label for interval in overlapping_context):
            errors.append(
                f"Semantic error: event {idx} label {label!r} conflicts with "
                f"{state_field} in overlapping context intervals"
            )

    # 3. Provenance Validation
    provenance = annotation.get("review_provenance", {})
    if provenance.get("status") != "HUMAN_APPROVED":
        errors.append(
            "Semantic error: review_provenance status must be HUMAN_APPROVED, "
            f"got {provenance.get('status')}. AI proposals are not valid for "
            "governed calibration."
        )
    else:
        required_fields = ["reviewer_id", "reviewed_at", "evidence_hash"]
        for field in required_fields:
            if not provenance.get(field):
                errors.append(
                    f"Semantic error: review_provenance missing {field} for HUMAN_APPROVED"
                )

    return errors


def main():
    parser = argparse.ArgumentParser(description="Validate event sequence annotations.")
    parser.add_argument("annotation_file", help="Path to annotation JSON file")
    parser.add_argument(
        "--schema",
        default="datasets/schemas/v2_event_sequence_annotation.schema.json",
        help="Path to schema JSON file",
    )
    args = parser.parse_args()

    schema = load_schema(args.schema)
    with open(args.annotation_file, encoding="utf-8") as f:
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
