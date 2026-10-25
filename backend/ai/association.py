from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Any

from backend.ai.detector import NormalizedDetection


@dataclass(frozen=True)
class OccupantRegion:
    role: str
    normalized_xyxy: list[float]
    enabled: bool = True
    min_overlap: float = 0.5


@dataclass(frozen=True)
class OccupantAssignment:
    role: str
    confidence: float
    method: str
    occupant_track_id: int | None = None

    @property
    def known(self) -> bool:
        return self.role != "unknown"


def normalized_roi_to_pixels(
    roi: list[float], width: int, height: int
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = roi
    return x1 * width, y1 * height, x2 * width, y2 * height


def intersection_over_box(
    box: tuple[float, float, float, float], roi: tuple[float, float, float, float]
) -> float:
    x1, y1, x2, y2 = box
    rx1, ry1, rx2, ry2 = roi
    intersection = max(0.0, min(x2, rx2) - max(x1, rx1)) * max(0.0, min(y2, ry2) - max(y1, ry1))
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return 0.0 if area == 0 else intersection / area


def associate_driver(
    detection: NormalizedDetection,
    frame_width: int,
    frame_height: int,
    roi: list[float],
    min_overlap: float = 0.5,
) -> bool:
    return (
        intersection_over_box(
            detection.xyxy, normalized_roi_to_pixels(roi, frame_width, frame_height)
        )
        >= min_overlap
    )


def associate_occupant(
    detection: NormalizedDetection,
    frame_width: int,
    frame_height: int,
    regions: list[dict],
) -> str | None:
    """Return the best calibrated occupant region for a detection."""
    best_role: str | None = None
    best_overlap = 0.0
    for item in regions:
        region = OccupantRegion(**item)
        if not region.enabled:
            continue
        overlap = intersection_over_box(
            detection.xyxy,
            normalized_roi_to_pixels(region.normalized_xyxy, frame_width, frame_height),
        )
        if overlap >= region.min_overlap and overlap > best_overlap:
            best_role = region.role
            best_overlap = overlap
    return best_role


def _pose_steering_and_gaze_score(
    box: tuple[float, float, float, float],
    poses: list[Any],
    frame_width: int,
    frame_height: int,
) -> tuple[float, float, Any | None]:
    """Compute (driver_posture_score, gaze_dx, matched_pose) for an occupant upper-body box.

    This is invariant to horizontal mirroring (anh nguoc):
    - In both front-facing dashcams (e.g. AnywayLabs 1363d5897d, 24da7148e6, 2976fd4270) and
      side-profile multi-occupant cameras (e.g. Mendeley 0008661593, AnywayLabs 6ff10c723b) or
      their horizontally flipped versions, the driver occupies the primary foreground seat with
      broader shoulder span, higher head placement, and steering-wheel arm extension, whereas
      the passenger sits clipped at the edge or recessed in the adjacent/background seat.
    """
    x1, y1, x2, y2 = box
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    fw = max(1.0, float(frame_width))
    fh = max(1.0, float(frame_height))
    area_norm = (bw * bh) / (fw * fh)
    cx = (x1 + x2) / 2.0 / fw
    cy = (y1 + y2) / 2.0 / fh

    best_pose = None
    best_ov = 0.0
    best_match = -1.0
    box_cx = (x1 + x2) / 2.0
    for p in poses or []:
        p_box = getattr(p, "xyxy", None)
        if p_box is None:
            continue
        ov = max(intersection_over_box(box, p_box), intersection_over_box(p_box, box))
        if ov < 0.20:
            continue
        core_pts = list(getattr(p, "face_points", ()) or ()) + list(
            getattr(p, "shoulder_points", ()) or ()
        )
        p_ax = (
            sum(pt[0] for pt in core_pts) / len(core_pts)
            if core_pts
            else (p_box[0] + p_box[2]) / 2.0
        )
        anchor_prox = max(0.0, 1.0 - abs(box_cx - p_ax) / (0.25 * fw))
        match_score = 0.45 * ov + 0.55 * anchor_prox
        if match_score > best_match:
            best_match = match_score
            best_ov = ov
            best_pose = p

    arm_reach = 0.0
    steering_reach_bonus = 0.0
    gaze_dx = 0.0
    shoulder_span_norm = 0.0
    torso_w = bw
    head_height_bonus = max(0.0, 0.55 - (y1 / fh)) * 0.45

    if best_pose is not None and best_ov >= 0.20:
        face_pts = getattr(best_pose, "face_points", ()) or ()
        hand_pts = getattr(best_pose, "hand_points", ()) or ()
        shoulder_pts = getattr(best_pose, "shoulder_points", ()) or ()

        if len(face_pts) >= 2:
            nose_x = face_pts[0][0]
            other_x = sum(pt[0] for pt in face_pts[1:]) / max(1, len(face_pts) - 1)
            gaze_dx = (nose_x - other_x) / max(80.0, bw)
            min_fy = min(pt[1] for pt in face_pts) / fh
            head_height_bonus = max(0.0, 0.60 - min_fy) * 0.55

        if len(shoulder_pts) >= 2:
            s_span_px = hypot(
                shoulder_pts[0][0] - shoulder_pts[1][0],
                shoulder_pts[0][1] - shoulder_pts[1][1],
            )
            shoulder_span_norm = s_span_px / fw
            torso_w = min(bw, max(120.0, 2.2 * s_span_px))

        ref_x = (
            sum(pt[0] for pt in shoulder_pts) / len(shoulder_pts)
            if shoulder_pts
            else (x1 + x2) / 2.0
        )
        if hand_pts:
            min_sx = min((pt[0] for pt in shoulder_pts), default=x1)
            max_sx = max((pt[0] for pt in shoulder_pts), default=x2)
            min_sy = min((pt[1] for pt in shoulder_pts), default=y1 + 0.30 * bh)
            max_sy = max((pt[1] for pt in shoulder_pts), default=y1 + 0.45 * bh)
            # Check if a hand is extended horizontally toward the steering wheel (and not at the ear, raised above shoulders, or waving outside the window)
            for hx, hy in hand_pts:
                near_ear = False
                if face_pts:
                    min_f = min(hypot(hx - fx, hy - fy) for fx, fy in face_pts)
                    if min_f < 0.28 * bh:
                        near_ear = True
                raised_above_shoulders = hy < (min_sy + 10.0)
                low_on_seat = hy > (0.72 * fh)
                waving_outside_window = (
                    (ref_x < 0.45 * fw and hx < min_sx - 0.03 * fw and hy < max_sy)
                    or (ref_x > 0.55 * fw and hx > max_sx + 0.03 * fw and hy < max_sy)
                )
                reach = min(0.45, abs(hx - ref_x) / fw)
                if not near_ear and not raised_above_shoulders and not low_on_seat and not waving_outside_window:
                    arm_reach = max(arm_reach, reach)
                    if reach >= 0.20 and (hx - ref_x) * gaze_dx > 0:
                        steering_reach_bonus = max(steering_reach_bonus, 0.48)
                else:
                    arm_reach = max(arm_reach, 0.08 * reach)

    area_norm = (torso_w * bh) / (fw * fh)

    # Penalty if clipped tightly against the extreme left/right edge in a lower foreground corner
    # (typical of near-side A-pillar passenger in Mendeley / AnywayLabs side camera)
    edge_clipped = (x1 <= 0.025 * fw or x2 >= 0.975 * fw) and (bw < 0.36 * fw and cy > 0.52)
    edge_penalty = -0.45 if edge_clipped else 0.0
    centrality = max(0.0, 1.0 - abs(cx - 0.50) * 1.4)

    driver_score = (
        1.35 * area_norm
        + 0.65 * arm_reach
        + steering_reach_bonus
        + 0.75 * shoulder_span_norm
        + head_height_bonus
        + 0.25 * centrality
        + edge_penalty
    )
    return driver_score, gaze_dx, best_pose


class OccupantAssociator:
    """Associate upper bodies and objects with seats using explicit confidence.

    Supports mirror-aware (anh nguoc) and side-camera multi-occupant reasoning so that:
    1. A single occupant in the cabin (even if horizontally flipped to x > 0.5) is not
       misclassified as front_passenger.
    2. Two occupants in side-view or mirrored cameras are distinguished using pose steering
       reach, gaze direction, and seat prominence before falling back to image_driver_side.
    3. Objects (phones) bind to the occupant whose hand/ear pose and upper-body box are
       anatomically closest, never via a rigid x >= 0.52 cutoff.
    """

    VALID_ROLES = {
        "driver",
        "front_passenger",
        "rear_left",
        "rear_center",
        "rear_right",
        "unknown",
    }

    def __init__(
        self,
        driver_side: str = "left",
        minimum_confidence: float = 0.55,
        rear_seats_enabled: bool = True,
        calibrated_regions: list[dict] | None = None,
    ):
        if driver_side not in {"left", "right"}:
            raise ValueError("driver_side must be left or right in image coordinates")
        self.driver_side = driver_side
        self.minimum_confidence = float(minimum_confidence)
        self.rear_seats_enabled = bool(rear_seats_enabled)
        self.calibrated_regions = calibrated_regions or []

    def assign_cabin_upper_bodies(
        self,
        upper_bodies: list[NormalizedDetection],
        frame_width: int,
        frame_height: int,
        poses: list[Any] | None = None,
    ) -> list[tuple[NormalizedDetection, OccupantAssignment]]:
        """Jointly assign roles to all detected upper bodies in a cabin frame (mirror-aware).

        Supports flexible occupant counts and roles:
        - 1 driver + 0 passengers (or 0 drivers + 1 passenger when clipped/passenger-only ROI)
        - 1 driver + 1 front_passenger in the same cabin (front-facing, side-camera, or mirrored)
        - 2 drivers in split-screen / dual-driver frames when both exhibit strong independent
          steering reach in separate non-overlapping halves
        """
        if not upper_bodies:
            return []

        poses = poses or []
        fw = max(1.0, float(frame_width))
        fh = max(1.0, float(frame_height))

        # Step 0: If > 2 upper_bodies were passed (e.g. duplicate pose on same person or empty
        # headrest ghost), filter out ghost poses and deduplicate by anatomical head/shoulder center.
        cleaned_ubs: list[NormalizedDetection] = []
        if len(upper_bodies) > 2:
            scored_candidates: list[tuple[float, float, NormalizedDetection]] = []
            for ub in upper_bodies:
                sc, _, mp = _pose_steering_and_gaze_score(ub.xyxy, poses, frame_width, frame_height)
                has_limbs = bool(
                    mp is not None
                    and (
                        getattr(mp, "hand_points", ())
                        or getattr(mp, "elbow_points", ())
                        or len(getattr(mp, "face_points", ()) or ()) >= 3
                    )
                )
                quality = float(ub.confidence) + (0.35 if has_limbs else -0.35) + 0.25 * sc
                anchor_x = (ub.xyxy[0] + ub.xyxy[2]) / 2.0
                if mp is not None:
                    f_pts = getattr(mp, "face_points", ()) or ()
                    s_pts = getattr(mp, "shoulder_points", ()) or ()
                    ref_pts = f_pts or s_pts
                    if ref_pts:
                        anchor_x = sum(pt[0] for pt in ref_pts) / len(ref_pts)
                scored_candidates.append((quality, anchor_x, ub))
            scored_candidates.sort(key=lambda item: item[0], reverse=True)
            kept: list[tuple[float, NormalizedDetection]] = []
            for _, ax, ub in scored_candidates:
                if all(abs(ax - k_ax) >= 0.14 * fw for k_ax, _ in kept):
                    kept.append((ax, ub))
                if len(kept) == 2:
                    break
            cleaned_ubs = [ub for _, ub in kept]
        else:
            cleaned_ubs = list(upper_bodies)

        # Case 1: Single occupant in the cabin
        if len(cleaned_ubs) == 1:
            ub = cleaned_ubs[0]
            x1, y1, x2, y2 = ub.xyxy
            cx = ((x1 + x2) / 2.0) / fw
            cy = ((y1 + y2) / 2.0) / fh
            if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
                return [(ub, OccupantAssignment("unknown", 0.0, "OUT_OF_CABIN", ub.track_id))]
            only_passenger_roi = (
                bool(self.calibrated_regions)
                and all(
                    r.get("role") == "front_passenger"
                    for r in self.calibrated_regions
                    if r.get("enabled", True)
                )
            )
            if only_passenger_roi:
                return [
                    (
                        ub,
                        OccupantAssignment(
                            "front_passenger",
                            max(0.85, float(ub.confidence)),
                            "SINGLE_OCCUPANT_PASSENGER_ROI",
                            ub.track_id,
                        ),
                    )
                ]
            sc, _, mp_single = _pose_steering_and_gaze_score(ub.xyxy, poses, frame_width, frame_height)
            bw = max(1.0, x2 - x1)
            bh = max(1.0, y2 - y1)
            s_pts_single = (
                getattr(mp_single, "shoulder_points", ()) or ()
                if mp_single is not None
                else ()
            )
            sy_single = [pt[1] for pt in s_pts_single]
            window_framed_passenger = (
                mp_single is not None
                and len(getattr(mp_single, "face_points", ()) or ()) == 5
                and len(s_pts_single) >= 2
                and len(getattr(mp_single, "hand_points", ()) or ()) == 0
                and len(getattr(mp_single, "elbow_points", ()) or ()) == 0
                and bh >= 0.42 * fh
                and min(sy_single) >= 0.50 * fh
                and (y2 - max(sy_single)) < 0.22 * fh
                and x2 <= 0.62 * fw
            )
            if window_framed_passenger:
                return [
                    (
                        ub,
                        OccupantAssignment(
                            "front_passenger",
                            max(0.88, float(ub.confidence)),
                            "SINGLE_OCCUPANT_WINDOW_PASSENGER",
                            ub.track_id,
                        ),
                    )
                ]
            exterior_windshield_passenger = (
                self.driver_side == "left"
                and cx < 0.46
                and y2 <= 0.72 * fh
                and bh <= 0.50 * fh
                and mp_single is not None
                and bool(getattr(mp_single, "hand_points", ()))
                and any(
                    hy < min((pt[1] for pt in (getattr(mp_single, "shoulder_points", ()) or [(0.0, y1 + 0.35 * bh)])))
                    for _, hy in (getattr(mp_single, "hand_points", ()) or ())
                )
            )
            if exterior_windshield_passenger:
                return [
                    (
                        ub,
                        OccupantAssignment(
                            "front_passenger",
                            max(0.88, float(ub.confidence)),
                            "EXTERIOR_WINDSHIELD_SINGLE_PASSENGER",
                            ub.track_id,
                        ),
                    )
                ]
            edge_clipped_passenger = (
                (x1 <= 0.025 * fw or x2 >= 0.975 * fw)
                and bw < 0.32 * fw
                and cy > 0.56
                and sc < 0.05
            )
            if edge_clipped_passenger:
                return [
                    (
                        ub,
                        OccupantAssignment(
                            "front_passenger",
                            max(0.85, float(ub.confidence)),
                            "SINGLE_OCCUPANT_CLIPPED_PASSENGER",
                            ub.track_id,
                        ),
                    )
                ]
            method = (
                "SINGLE_OCCUPANT_MIRROR_TOLERANT"
                if (self.driver_side == "left" and cx >= 0.52)
                or (self.driver_side == "right" and cx < 0.48)
                else "SINGLE_OCCUPANT_PRIMARY"
            )
            return [
                (
                    ub,
                    OccupantAssignment(
                        "driver",
                        max(0.88, min(0.96, float(ub.confidence))),
                        method,
                        ub.track_id,
                    ),
                )
            ]

        # Case 2: Two occupants in the frame
        sorted_ubs = sorted(cleaned_ubs, key=lambda d: (d.xyxy[0] + d.xyxy[2]) / 2.0)
        left_ub = sorted_ubs[0]
        right_ub = sorted_ubs[-1]

        l_score, l_gaze, l_pose = _pose_steering_and_gaze_score(
            left_ub.xyxy, poses, frame_width, frame_height
        )
        r_score, r_gaze, r_pose = _pose_steering_and_gaze_score(
            right_ub.xyxy, poses, frame_width, frame_height
        )

        # Check if BOTH occupants are independent drivers in a split-screen / dual-driver frame:
        # Both sit strictly in their own half (x2_left <= 0.52*w, x1_right >= 0.48*w) and both have
        # strong independent driver posture + extended steering hand reach.
        split_halves = left_ub.xyxy[2] <= 0.52 * fw and right_ub.xyxy[0] >= 0.48 * fw
        both_driver_rois = (
            len([r for r in self.calibrated_regions if r.get("enabled", True) and r.get("role") == "driver"]) >= 2
        )
        if both_driver_rois or (
            split_halves
            and l_score >= 0.92
            and r_score >= 0.92
            and abs(r_score - l_score) <= 0.14
            and l_pose is not None
            and r_pose is not None
            and len(getattr(l_pose, "hand_points", ()) or ()) >= 2
            and len(getattr(r_pose, "hand_points", ()) or ()) >= 2
        ):
            return [
                (
                    left_ub,
                    OccupantAssignment(
                        "driver",
                        max(0.88, min(0.96, float(left_ub.confidence))),
                        "DUAL_DRIVER_INDEPENDENT",
                        left_ub.track_id,
                    ),
                ),
                (
                    right_ub,
                    OccupantAssignment(
                        "driver",
                        max(0.88, min(0.96, float(right_ub.confidence))),
                        "DUAL_DRIVER_INDEPENDENT",
                        right_ub.track_id,
                    ),
                ),
            ]

        # Shared cabin with 2 occupants: compare pose steering, shoulder span, head height, and prominence
        score_diff = r_score - l_score
        max_ub_h = max(left_ub.xyxy[3] - left_ub.xyxy[1], right_ub.xyxy[3] - right_ub.xyxy[1])
        is_exterior_windshield_view = (
            (
                max(left_ub.xyxy[3], right_ub.xyxy[3]) <= 0.48 * fh
                and max_ub_h <= 0.40 * fh
            )
            or (
                max_ub_h <= 0.52 * fh
                and max(left_ub.xyxy[3], right_ub.xyxy[3]) <= 0.72 * fh
                and abs(left_ub.xyxy[3] - right_ub.xyxy[3]) <= 0.05 * fh
            )
        )
        # Check if 2 occupants overlap horizontally in a right-facing side-profile camera
        l_cx = (left_ub.xyxy[0] + left_ub.xyxy[2]) / 2.0
        r_cx = (right_ub.xyxy[0] + right_ub.xyxy[2]) / 2.0
        is_right_facing_side_camera = (
            abs(r_cx - l_cx) < 0.32 * fw
            and (l_gaze > 0.02 and r_gaze >= -0.01 or (l_gaze + r_gaze) > 0.04)
        )
        diff_thresh = 0.05 if max_ub_h <= 0.38 * fh else 0.10
        if is_exterior_windshield_view:
            driver_is_left = False if self.driver_side == "left" else True
            method = "EXTERIOR_WINDSHIELD_VIEW"
        elif is_right_facing_side_camera:
            driver_is_left = False if self.driver_side == "left" else True
            method = "SIDE_PROFILE_RIGHT_FACING"
        elif abs(score_diff) >= diff_thresh:
            driver_is_left = score_diff < 0
            method = "POSE_STEERING_MIRROR_AWARE"
        else:
            driver_is_left = self.driver_side == "left"
            method = "DUAL_OCCUPANT_GEOMETRY"

        left_role = "driver" if driver_is_left else "front_passenger"
        right_role = "front_passenger" if driver_is_left else "driver"

        results: list[tuple[NormalizedDetection, OccupantAssignment]] = [
            (
                left_ub,
                OccupantAssignment(
                    left_role,
                    max(0.88, min(0.96, float(left_ub.confidence))),
                    method,
                    left_ub.track_id,
                ),
            ),
            (
                right_ub,
                OccupantAssignment(
                    right_role,
                    max(0.88, min(0.96, float(right_ub.confidence))),
                    method,
                    right_ub.track_id,
                ),
            ),
        ]
        return results

    def assign_upper_body(
        self,
        detection: NormalizedDetection,
        frame_width: int,
        frame_height: int,
    ) -> OccupantAssignment:
        calibrated = self._calibrated(detection, frame_width, frame_height)
        if calibrated.confidence >= self.minimum_confidence:
            return calibrated

        x1, y1, x2, y2 = detection.xyxy
        center_x = ((x1 + x2) / 2) / max(frame_width, 1)
        center_y = ((y1 + y2) / 2) / max(frame_height, 1)
        if not 0.0 <= center_x <= 1.0 or not 0.0 <= center_y <= 1.0:
            return OccupantAssignment("unknown", 0.0, "OUT_OF_CABIN", detection.track_id)

        if self.rear_seats_enabled and center_y >= 0.68:
            if center_x < 0.34:
                role, anchor = "rear_left", (0.18, 0.78)
            elif center_x > 0.66:
                role, anchor = "rear_right", (0.82, 0.78)
            else:
                role, anchor = "rear_center", (0.50, 0.78)
        else:
            driver_on_left = self.driver_side == "left"
            if abs(center_x - 0.5) < 0.08:
                return OccupantAssignment(
                    "unknown", 0.45, "CABIN_GEOMETRY_AMBIGUOUS", detection.track_id
                )
            is_left = center_x < 0.5
            role = "driver" if is_left == driver_on_left else "front_passenger"
            anchor = ((0.25 if is_left else 0.75), 0.48)

        distance = hypot(center_x - anchor[0], center_y - anchor[1])
        confidence = max(0.0, min(0.85, 0.85 - distance))
        if confidence < self.minimum_confidence:
            return OccupantAssignment(
                "unknown", confidence, "CABIN_GEOMETRY_LOW_CONFIDENCE", detection.track_id
            )
        return OccupantAssignment(role, confidence, "CABIN_GEOMETRY", detection.track_id)

    def assign_object(
        self,
        detection: NormalizedDetection,
        occupants: list[tuple[NormalizedDetection, OccupantAssignment]],
        frame_width: int,
        frame_height: int,
        poses: list[Any] | None = None,
    ) -> OccupantAssignment:
        if not occupants:
            # In a close-up POV cabin shot (where only hands/arms and steering wheel are in view
            # and no full occupant upper-body box is present), bind a handheld phone to the driver
            # if pose keypoints show hands/arms holding or overlapping the phone, or if the phone
            # is a high-confidence close-up macro shot filling the foreground.
            if detection.class_name == "phone":
                bw = max(1.0, float(detection.xyxy[2] - detection.xyxy[0]))
                bh = max(1.0, float(detection.xyxy[3] - detection.xyxy[1]))
                is_closeup_phone = (
                    float(detection.confidence) >= 0.60
                    and (bw >= 0.22 * float(frame_width) or bh >= 0.32 * float(frame_height))
                )
                has_hand_overlap = False
                if poses:
                    for p in poses:
                        if getattr(p, "xyxy", None) is None:
                            continue
                        h_pts = list(getattr(p, "hand_points", ()) or ())
                        e_pts = list(getattr(p, "elbow_points", ()) or ())
                        arm_pts = h_pts + e_pts
                        ov = max(
                            intersection_over_box(detection.xyxy, p.xyxy),
                            intersection_over_box(p.xyxy, detection.xyxy),
                        )
                        if arm_pts and ov >= 0.20:
                            has_hand_overlap = True
                            break
                if has_hand_overlap or is_closeup_phone:
                    return OccupantAssignment(
                        "driver",
                        0.90,
                        "CLOSEUP_POV_DRIVER_HAND_PHONE",
                        1,
                    )
            calibrated = self._calibrated(detection, frame_width, frame_height)
            if calibrated.confidence >= self.minimum_confidence:
                return OccupantAssignment(
                    calibrated.role,
                    calibrated.confidence,
                    "CAMERA_CALIBRATION_NO_OCCUPANT_TRACK",
                    None,
                )
            return OccupantAssignment("unknown", calibrated.confidence, "NO_OCCUPANT_MATCH", None)

        # If there is only 1 known occupant in the cabin, bind the object to that occupant
        # unless it is outside reasonable reach of their upper body.
        if len(occupants) == 1:
            occ_det, occ_assign = occupants[0]
            if occ_assign.known:
                ox1, oy1, ox2, oy2 = occ_det.xyxy
                ow = max(1.0, ox2 - ox1)
                oh = max(1.0, oy2 - oy1)
                pcx = (detection.xyxy[0] + detection.xyxy[2]) / 2.0
                pcy = (detection.xyxy[1] + detection.xyxy[3]) / 2.0
                # Allow generous reach around the single occupant (including outstretched hand near steering wheel)
                if (ox1 - 0.85 * ow) <= pcx <= (ox2 + 0.85 * ow) and (oy1 - 0.35 * oh) <= pcy <= (
                    oy2 + 0.45 * oh
                ):
                    return OccupantAssignment(
                        occ_assign.role,
                        max(0.88, float(occ_assign.confidence)),
                        "SINGLE_OCCUPANT_OBJECT_BINDING",
                        occ_det.track_id,
                    )

        object_center = (
            (detection.xyxy[0] + detection.xyxy[2]) / 2,
            (detection.xyxy[1] + detection.xyxy[3]) / 2,
        )
        diagonal = max(hypot(frame_width, frame_height), 1.0)
        candidates: list[tuple[float, NormalizedDetection, OccupantAssignment]] = []
        for occupant, assignment in occupants:
            eff_box = occupant.xyxy
            center = (
                (occupant.xyxy[0] + occupant.xyxy[2]) / 2,
                (occupant.xyxy[1] + occupant.xyxy[3]) / 2,
            )
            anat_bonus = 0.0
            if poses:
                _, _, matched_pose = _pose_steering_and_gaze_score(
                    occupant.xyxy, poses, frame_width, frame_height
                )
                if matched_pose is not None:
                    f_pts = list(getattr(matched_pose, "face_points", ()) or ())
                    s_pts = list(getattr(matched_pose, "shoulder_points", ()) or ())
                    h_pts = list(getattr(matched_pose, "hand_points", ()) or ())
                    e_pts = list(getattr(matched_pose, "elbow_points", ()) or ())
                    min_sy_occ = min((pt[1] for pt in s_pts), default=occupant.xyxy[1] + 150.0)
                    min_sx_occ = min((pt[0] for pt in s_pts), default=occupant.xyxy[0])
                    dx1, dy1, dx2, dy2 = detection.xyxy
                    if detection.class_name == "phone":
                        # 1) Wrists genuinely close to the phone bounding box (within 0.12*diagonal of box edge)
                        active_wrists = [
                            (hx, hy)
                            for hx, hy in h_pts
                            if hypot(max(dx1 - hx, 0.0, hx - dx2), max(dy1 - hy, 0.0, hy - dy2))
                            <= 0.12 * diagonal
                        ]
                        # 1b) Raised forearm/wrist at phone elevation between shoulders and phone (selfie posture)
                        raised_selfie_wrists = [
                            (hx, hy)
                            for hx, hy in h_pts
                            if hy < (min_sy_occ - 20.0)
                            and abs(hy - object_center[1]) <= 0.14 * frame_height
                            and (min_sx_occ - 30.0) <= hx <= (dx2 + 25.0)
                        ]
                        # 2) Elbows supporting the forearm below/near the phone when the wrist is occluded
                        has_far_steering_wrist = any(
                            hx > dx2 + 0.04 * frame_width and hy >= dy1 - 0.05 * frame_height
                            for hx, hy in h_pts
                        )
                        supporting_elbows = (
                            [
                                (ex, ey)
                                for ex, ey in e_pts
                                if -0.05 * frame_height <= (ey - dy2) <= 0.42 * frame_height
                                and abs(ex - object_center[0]) <= 0.26 * frame_width
                                and ey <= min_sy_occ + 0.22 * frame_height
                            ]
                            if not has_far_steering_wrist
                            else []
                        )
                        active_arm_pts = active_wrists + raised_selfie_wrists + supporting_elbows
                        pts = active_arm_pts if object_center[1] < 0.42 * frame_height else (active_arm_pts + f_pts)
                        if raised_selfie_wrists:
                            anat_bonus += 0.95
                        elif not active_wrists and has_far_steering_wrist and len(occupants) >= 2:
                            anat_bonus -= 0.75
                        elif not active_wrists and object_center[1] < 0.42 * frame_height:
                            anat_bonus += -0.18 if has_far_steering_wrist else (0.28 if len(occupants) >= 2 else 0.0)
                    else:
                        active_arm_pts = h_pts + e_pts
                        pts = active_arm_pts + f_pts
                    core_pts = f_pts + s_pts
                    if core_pts:
                        cx_anat = sum(p[0] for p in core_pts) / len(core_pts)
                        cy_anat = sum(p[1] for p in core_pts) / len(core_pts)
                        center = (cx_anat, cy_anat)
                        if len(occupants) >= 2 and len(core_pts) >= 2:
                            pad_x = 0.08 * frame_width
                            pad_y = 0.10 * frame_height
                            eff_box = (
                                max(occupant.xyxy[0], min(p[0] for p in core_pts) - pad_x),
                                max(occupant.xyxy[1], min(p[1] for p in core_pts) - pad_y),
                                min(occupant.xyxy[2], max(p[0] for p in core_pts) + pad_x),
                                min(occupant.xyxy[3], max(p[1] for p in core_pts) + pad_y),
                            )
                    if pts:
                        min_kp_dist = (
                            min(
                                hypot(max(dx1 - px, 0.0, px - dx2), max(dy1 - py, 0.0, py - dy2))
                                for px, py in pts
                            )
                            / diagonal
                        )
                        anat_bonus += 0.65 * max(0.0, 1.0 - min_kp_dist * 4.5)
                        if active_arm_pts:
                            min_hand_dist = (
                                min(
                                    hypot(max(dx1 - hx, 0.0, hx - dx2), max(dy1 - hy, 0.0, hy - dy2))
                                    for hx, hy in active_arm_pts
                                )
                                / diagonal
                            )
                            anat_bonus += 0.75 * max(0.0, 1.0 - min_hand_dist * 4.5)
                            if object_center[1] < 0.32 * frame_height:
                                anat_bonus += 0.45
            overlap = intersection_over_box(detection.xyxy, eff_box)
            distance = hypot(object_center[0] - center[0], object_center[1] - center[1]) / diagonal
            score = 0.42 * overlap + 0.35 * max(0.0, 1.0 - distance * 2.5) + anat_bonus
            candidates.append((score, occupant, assignment))
        score, occupant, assignment = max(candidates, key=lambda item: item[0])
        confidence = min(assignment.confidence, max(0.85, score) if score >= 0.25 else score)
        if score < 0.22 or not assignment.known or confidence < self.minimum_confidence:
            calibrated = self._calibrated(detection, frame_width, frame_height)
            if calibrated.known and calibrated.confidence >= self.minimum_confidence:
                return OccupantAssignment(
                    calibrated.role,
                    calibrated.confidence,
                    "CAMERA_CALIBRATION_FALLBACK",
                    occupant.track_id,
                )
            return OccupantAssignment(
                "unknown", confidence, "OBJECT_OCCUPANT_LOW_CONFIDENCE", occupant.track_id
            )
        return OccupantAssignment(
            assignment.role, confidence, "OBJECT_TO_OCCUPANT", occupant.track_id
        )

    def _calibrated(
        self,
        detection: NormalizedDetection,
        frame_width: int,
        frame_height: int,
    ) -> OccupantAssignment:
        best_role = "unknown"
        best_overlap = 0.0
        for item in self.calibrated_regions:
            region = OccupantRegion(**item)
            if not region.enabled:
                continue
            overlap = intersection_over_box(
                detection.xyxy,
                normalized_roi_to_pixels(region.normalized_xyxy, frame_width, frame_height),
            )
            if overlap >= region.min_overlap and overlap > best_overlap:
                best_role, best_overlap = region.role, overlap
        if best_role not in self.VALID_ROLES:
            return OccupantAssignment("unknown", 0.0, "INVALID_CALIBRATION", detection.track_id)
        return OccupantAssignment(best_role, best_overlap, "CAMERA_CALIBRATION", detection.track_id)
