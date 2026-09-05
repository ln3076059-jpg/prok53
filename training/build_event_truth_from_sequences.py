import argparse
import csv
import json
import os
import sys
from pathlib import Path

from training.validate_event_sequence_annotations import load_schema, validate_annotation

EVENT_FIELDNAMES = [
    "video_id",
    "event_id",
    "event_type",
    "start_seconds",
    "end_seconds",
    "occupant_id",
    "vehicle_id",
    "cabin_id",
    "occupant_role",
    "inside_vehicle",
    "outside_vehicle_person",
    "motorcycle_flag",
    "label",
    "visibility",
    "conditions",
    "human_review_status",
    "reviewer_id",
    "reviewer_type",
    "reviewed_at",
    "adjudication_status",
    "notes",
    "identity_manifest_sha256",
]

CONTEXT_FIELDNAMES = [
    "video_id",
    "context_id",
    "occupant_id",
    "vehicle_id",
    "cabin_id",
    "occupant_role",
    "start_seconds",
    "end_seconds",
    "timeline_end_seconds",
    "inside_vehicle",
    "outside_vehicle_person",
    "motorcycle_flag",
    "phone_state",
    "seatbelt_state",
    "visibility",
    "conditions",
    "human_review_status",
    "reviewer_id",
    "reviewer_type",
    "reviewed_at",
    "adjudication_status",
    "notes",
    "identity_manifest_sha256",
]


def _review_fields(annotation):
    provenance = annotation.get("review_provenance", {})
    prov_status = provenance.get("status", "")
    return {
        "human_review_status": "APPROVED" if prov_status == "HUMAN_APPROVED" else prov_status,
        "reviewer_id": provenance.get("reviewer_id", ""),
        "reviewer_type": provenance.get("reviewer_type", ""),
        "reviewed_at": provenance.get("reviewed_at", ""),
        "adjudication_status": provenance.get("adjudication_status"),
        "identity_manifest_sha256": provenance.get("identity_manifest_sha256", ""),
    }


def convert_sequence_to_events(annotation):
    emitted_events = []

    video_id = annotation.get("video_id")
    sequence_id = annotation.get("sequence_id", "unknown_seq")
    fps = annotation.get("fps", 30.0)

    if not video_id:
        return emitted_events

    vehicle_id = annotation.get("vehicle_id", "")
    cabin_id = annotation.get("cabin_id", "")

    review = _review_fields(annotation)
    adjudication_status = review["adjudication_status"]

    if adjudication_status != "FINAL":
        raise ValueError(f"Annotation adjudication_status is not FINAL (got {adjudication_status})")

    occupants = annotation.get("occupants", [])
    occ_map = {occ["occupant_id"]: occ for occ in occupants}

    events = annotation.get("events", [])

    for idx, evt in enumerate(events):
        event_type = evt.get("event_type")
        label = evt.get("label")
        occ_id = evt.get("occupant_id")

        start_frame = evt.get("start_frame", 0)
        end_frame = evt.get("end_frame", 0)

        overlapping_context = [
            interval
            for interval in annotation.get("context_intervals", [])
            if interval.get("occupant_id") == occ_id
            and interval.get("start_frame", annotation["frame_count"]) < end_frame + 1
            and interval.get("end_frame", 0) > start_frame
        ]
        context_signatures = {
            (
                interval["inside_vehicle"],
                interval["outside_vehicle_person"],
                interval["motorcycle_flag"],
                interval["visibility"],
                interval["conditions"],
            )
            for interval in overlapping_context
        }
        if len(context_signatures) != 1:
            raise ValueError(f"event {idx} does not have one unambiguous context signature")
        event_context = overlapping_context[0]

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
            if label in (
                "PHONE_USE",
                "MOUNTED_OR_STATIC_PHONE",
                "PHONE_PRESENT_NOT_USED",
                "NO_PHONE",
                "UNKNOWN",
            ):
                final_event_type = "PHONE"
        elif event_type == "SEATBELT":
            if label in ("FASTENED", "UNFASTENED", "UNCERTAIN_OR_OCCLUDED", "NOT_APPLICABLE"):
                final_event_type = "NO_SEATBELT"

        if final_event_type:
            event_id = f"{video_id}_{sequence_id}_evt_{idx}"
            emitted_events.append(
                {
                    "video_id": video_id,
                    "event_id": event_id,
                    "event_type": final_event_type,
                    "start_seconds": start_sec,
                    "end_seconds": end_sec,
                    "occupant_id": occ_id,
                    "vehicle_id": vehicle_id,
                    "cabin_id": cabin_id,
                    "occupant_role": role,
                    "inside_vehicle": str(event_context["inside_vehicle"]).lower(),
                    "outside_vehicle_person": str(event_context["outside_vehicle_person"]).lower(),
                    "motorcycle_flag": str(event_context["motorcycle_flag"]).lower(),
                    "label": label,
                    "visibility": event_context["visibility"],
                    "conditions": event_context["conditions"],
                    "human_review_status": review["human_review_status"],
                    "reviewer_id": review["reviewer_id"],
                    "reviewer_type": review["reviewer_type"],
                    "reviewed_at": review["reviewed_at"],
                    "adjudication_status": adjudication_status,
                    "notes": event_context.get("notes", ""),
                    "identity_manifest_sha256": review["identity_manifest_sha256"],
                }
            )

    return emitted_events


def convert_sequence_to_context(annotation):
    """Convert reviewed half-open frame intervals into full-timeline context rows."""
    video_id = annotation["video_id"]
    fps = float(annotation["fps"])
    frame_count = int(annotation["frame_count"])
    vehicle_id = annotation["vehicle_id"]
    cabin_id = annotation["cabin_id"]
    review = _review_fields(annotation)
    if review["adjudication_status"] != "FINAL":
        raise ValueError(
            f"Annotation adjudication_status is not FINAL (got {review['adjudication_status']})"
        )
    roles = {occupant["occupant_id"]: occupant["role"] for occupant in annotation["occupants"]}
    timeline_end = frame_count / fps
    rows = []
    for interval in annotation["context_intervals"]:
        occupant_id = interval["occupant_id"]
        rows.append(
            {
                "video_id": video_id,
                "context_id": interval["context_id"],
                "occupant_id": occupant_id,
                "vehicle_id": vehicle_id,
                "cabin_id": cabin_id,
                "occupant_role": roles[occupant_id],
                "start_seconds": interval["start_frame"] / fps,
                "end_seconds": interval["end_frame"] / fps,
                "timeline_end_seconds": timeline_end,
                "inside_vehicle": str(interval["inside_vehicle"]).lower(),
                "outside_vehicle_person": str(interval["outside_vehicle_person"]).lower(),
                "motorcycle_flag": str(interval["motorcycle_flag"]).lower(),
                "phone_state": interval["phone_state"],
                "seatbelt_state": interval["seatbelt_state"],
                "visibility": interval["visibility"],
                "conditions": interval["conditions"],
                **review,
                "notes": interval.get("notes", ""),
            }
        )
    return rows


def process_file(json_path, csv_writer, schema, context_writer=None):
    with open(json_path, encoding="utf-8") as f:
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
        if hasattr(csv_writer, "fieldnames"):
            csv_writer.writerow(evt)
        else:
            csv_writer.writerow([evt[field] for field in EVENT_FIELDNAMES])
    if context_writer is not None:
        try:
            context_rows = convert_sequence_to_context(annotation)
        except ValueError as e:
            print(f"Skipping {json_path} due to context conversion error: {e}")
            return False
        for row in context_rows:
            context_writer.writerow(row)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Convert sequence JSON annotations to final event CSV"
    )
    parser.add_argument("input_path", help="Path to input JSON file or directory")
    parser.add_argument("output_csv", help="Path to output CSV file")
    parser.add_argument(
        "--context-output",
        required=True,
        help="Path to the full-timeline context truth CSV",
    )
    parser.add_argument(
        "--schema",
        default="datasets/schemas/v2_event_sequence_annotation.schema.json",
        help="Path to sequence schema JSON",
    )

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
    temp_context_csv = str(args.context_output) + ".tmp"
    with (
        open(temp_csv, "w", encoding="utf-8", newline="") as f,
        open(temp_context_csv, "w", encoding="utf-8", newline="") as context_file,
    ):
        writer = csv.DictWriter(f, fieldnames=EVENT_FIELDNAMES)
        writer.writeheader()
        context_writer = csv.DictWriter(context_file, fieldnames=CONTEXT_FIELDNAMES)
        context_writer.writeheader()
        for p in files_to_process:
            if process_file(p, writer, schema, context_writer):
                success_count += 1

    if success_count < len(files_to_process):
        print(f"Failed to process {len(files_to_process) - success_count} files. Aborting export.")
        if os.path.exists(temp_csv):
            os.remove(temp_csv)
        if os.path.exists(temp_context_csv):
            os.remove(temp_context_csv)
        sys.exit(1)
    else:
        print(
            f"Successfully converted {success_count}/{len(files_to_process)} files "
            f"to {args.output_csv} and {args.context_output}"
        )
        os.replace(temp_csv, args.output_csv)
        os.replace(temp_context_csv, args.context_output)


if __name__ == "__main__":
    main()
