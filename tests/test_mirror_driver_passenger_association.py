from __future__ import annotations

from backend.ai.association import OccupantAssociator
from backend.ai.auxiliary import PoseEvidence, evaluate_occupant_violation_policy
from backend.ai.detector import NormalizedDetection


def test_single_driver_mirrored_frame_stays_driver_and_keeps_right_side_phone():
    """When a single-driver frame is horizontally flipped (anh nguoc) so the driver and phone
    sit at x > 0.52, the occupant must still be classified as 'driver' and their phone as VIOLATION.
    """
    assoc = OccupantAssociator(driver_side="left", minimum_confidence=0.35)
    w, h = 1280, 720

    # Mirrored single driver on the right half of the image (cx = 0.68)
    mirrored_driver_ub = NormalizedDetection(
        0, "occupant_upper_body", 0.93, (650.0, 140.0, 1100.0, 660.0), track_id=1
    )
    mirrored_pose = PoseEvidence(
        confidence=0.90,
        face_points=((860.0, 230.0), (880.0, 215.0), (910.0, 225.0)),
        hand_points=((620.0, 390.0), (790.0, 310.0)),
        xyxy=(650.0, 140.0, 1100.0, 660.0),
        shoulder_points=((820.0, 330.0), (960.0, 335.0)),
    )
    assigned = assoc.assign_cabin_upper_bodies([mirrored_driver_ub], w, h, poses=[mirrored_pose])
    assert len(assigned) == 1
    assert assigned[0][1].role == "driver"

    # Phone held at x = 760..830 (phone_cx = 0.62 >= 0.52)
    phone_det = NormalizedDetection(0, "phone", 0.91, (760.0, 270.0, 830.0, 360.0), track_id=10)
    phone_assign = assoc.assign_object(phone_det, assigned, w, h, poses=[mirrored_pose])
    assert phone_assign.role == "driver"
    is_viol, code, _ = evaluate_occupant_violation_policy("phone", phone_assign.role, "HANDHELD_PHONE_USE")
    assert is_viol is True
    assert code == "VIOLATION_DRIVER_PHONE"


def test_side_camera_two_occupants_passenger_left_driver_right_and_flipped():
    """In side-camera multi-occupant cabin images (e.g. Mendeley M[0], AnywayLabs 6ff10c723b),
    the near-side passenger is clipped at the left border (x < 0.30) while the driver sits in the
    main cabin seat on the right (x > 0.40) reaching for the steering wheel.
    Verify correct driver/passenger assignment on BOTH original and horizontally flipped images.
    """
    assoc = OccupantAssociator(driver_side="left", minimum_confidence=0.35)
    w, h = 1280, 720

    # 1) Original side-camera frame: Passenger clipped on left, Driver holding wheel on right
    pass_left_ub = NormalizedDetection(
        0, "occupant_upper_body", 0.89, (8.0, 410.0, 320.0, 710.0), track_id=1
    )
    drv_right_ub = NormalizedDetection(
        0, "occupant_upper_body", 0.95, (430.0, 150.0, 1120.0, 690.0), track_id=2
    )
    pass_pose = PoseEvidence(
        confidence=0.86,
        face_points=((190.0, 460.0), (165.0, 450.0)),
        hand_points=((240.0, 560.0), (260.0, 575.0)),
        xyxy=(8.0, 410.0, 320.0, 710.0),
        shoulder_points=((140.0, 520.0), (210.0, 525.0)),
    )
    drv_pose = PoseEvidence(
        confidence=0.94,
        face_points=((690.0, 240.0), (650.0, 230.0)),
        hand_points=((1060.0, 350.0), (630.0, 270.0)),  # Left hand extended to steering wheel, right hand at ear
        xyxy=(430.0, 150.0, 1120.0, 690.0),
        shoulder_points=((640.0, 350.0), (720.0, 355.0)),
    )
    assigned_orig = assoc.assign_cabin_upper_bodies(
        [pass_left_ub, drv_right_ub], w, h, poses=[pass_pose, drv_pose]
    )
    roles_by_tid = {ub.track_id: a.role for ub, a in assigned_orig}
    assert roles_by_tid[1] == "front_passenger"
    assert roles_by_tid[2] == "driver"

    # Driver's phone at ear (x=600..660, on the right half) must bind to driver -> VIOLATION
    drv_phone = NormalizedDetection(0, "phone", 0.92, (600.0, 235.0, 660.0, 315.0))
    drv_phone_assign = assoc.assign_object(
        drv_phone, assigned_orig, w, h, poses=[pass_pose, drv_pose]
    )
    assert drv_phone_assign.role == "driver"

    # Passenger's phone in hand (x=220..280, on the left half) must bind to front_passenger -> EXEMPT
    pass_phone = NormalizedDetection(0, "phone", 0.90, (220.0, 530.0, 280.0, 600.0))
    pass_phone_assign = assoc.assign_object(
        pass_phone, assigned_orig, w, h, poses=[pass_pose, drv_pose]
    )
    assert pass_phone_assign.role == "front_passenger"

    # 2) Horizontally flipped version of the same frame (x -> w - x)
    drv_left_flipped = NormalizedDetection(
        0, "occupant_upper_body", 0.95, (w - 1120.0, 150.0, w - 430.0, 690.0), track_id=2
    )
    pass_right_flipped = NormalizedDetection(
        0, "occupant_upper_body", 0.89, (w - 320.0, 410.0, w - 8.0, 710.0), track_id=1
    )
    drv_pose_flipped = PoseEvidence(
        confidence=0.94,
        face_points=((w - 690.0, 240.0), (w - 650.0, 230.0)),
        hand_points=((w - 1060.0, 350.0), (w - 630.0, 270.0)),
        xyxy=(w - 1120.0, 150.0, w - 430.0, 690.0),
        shoulder_points=((w - 720.0, 355.0), (w - 640.0, 350.0)),
    )
    pass_pose_flipped = PoseEvidence(
        confidence=0.86,
        face_points=((w - 190.0, 460.0), (w - 165.0, 450.0)),
        hand_points=((w - 240.0, 560.0), (w - 260.0, 575.0)),
        xyxy=(w - 320.0, 410.0, w - 8.0, 710.0),
        shoulder_points=((w - 210.0, 525.0), (w - 140.0, 520.0)),
    )
    assigned_flip = assoc.assign_cabin_upper_bodies(
        [drv_left_flipped, pass_right_flipped], w, h, poses=[drv_pose_flipped, pass_pose_flipped]
    )
    roles_flip = {ub.track_id: a.role for ub, a in assigned_flip}
    assert roles_flip[2] == "driver"
    assert roles_flip[1] == "front_passenger"


def test_same_frame_three_poses_filters_ghost_and_keeps_true_passenger_and_driver():
    """In front-facing 2-occupant cabin frames like AnywayLabs 2976fd4270, YOLO11-pose can emit
    a 3rd ghost pose on the window/headrest (0..270) alongside the true passenger (205..460) and
    driver (437..1090). Verify that the ghost is filtered out and the true passenger and driver
    are accurately assigned.
    """
    assoc = OccupantAssociator(driver_side="left", minimum_confidence=0.35)
    w, h = 1376, 768

    ub_pass = NormalizedDetection(
        0, "occupant_upper_body", 0.681, (205.6, 309.6, 459.8, 754.5), track_id=1
    )
    ub_drv = NormalizedDetection(
        0, "occupant_upper_body", 0.755, (437.1, 163.1, 1090.8, 741.6), track_id=2
    )
    ub_ghost = NormalizedDetection(
        0, "occupant_upper_body", 0.669, (0.0, 261.4, 270.4, 766.2), track_id=3
    )
    pose_pass = PoseEvidence(
        confidence=0.681,
        face_points=((269.9, 425.8), (293.0, 404.4), (254.9, 404.4), (339.4, 409.0), (244.8, 406.3)),
        hand_points=((356.3, 768.0),),
        xyxy=(205.6, 309.6, 459.8, 754.5),
        shoulder_points=((400.4, 520.1), (210.3, 513.5)),
        elbow_points=((416.0, 693.8),),
    )
    pose_drv = PoseEvidence(
        confidence=0.755,
        face_points=((826.9, 361.9), (855.0, 328.5), (776.9, 310.9), (656.9, 332.2)),
        hand_points=((1008.1, 680.6), (785.5, 694.9)),
        xyxy=(437.1, 163.1, 1090.8, 741.6),
        shoulder_points=((923.1, 587.1), (525.6, 576.5)),
        elbow_points=((1051.9, 768.0), (495.4, 768.0)),
    )
    pose_ghost = PoseEvidence(
        confidence=0.669,
        face_points=((70.9, 400.5), (192.9, 402.1)),
        hand_points=(),
        xyxy=(0.0, 261.4, 270.4, 766.2),
        shoulder_points=((25.4, 532.2), (246.1, 540.5)),
        elbow_points=(),
    )
    assigned = assoc.assign_cabin_upper_bodies(
        [ub_pass, ub_drv, ub_ghost], w, h, poses=[pose_pass, pose_drv, pose_ghost]
    )
    roles_by_tid = {ub.track_id: a.role for ub, a in assigned}
    assert 3 not in roles_by_tid
    assert roles_by_tid[1] == "front_passenger"
    assert roles_by_tid[2] == "driver"


def test_flexible_occupant_counts_two_drivers_or_no_driver():
    """Verify that the associator supports:
    1) Two independent drivers in a split-screen / dual-driver frame ('co the co 2 driver')
    2) A single clipped passenger without a driver ('hoac tuy ko co')
    """
    assoc = OccupantAssociator(driver_side="left", minimum_confidence=0.35)
    w, h = 1280, 720

    # 1) Dual drivers in split halves, both with extended steering reach
    drv1_ub = NormalizedDetection(
        0, "occupant_upper_body", 0.94, (40.0, 120.0, 600.0, 680.0), track_id=1
    )
    drv2_ub = NormalizedDetection(
        0, "occupant_upper_body", 0.94, (680.0, 120.0, 1240.0, 680.0), track_id=2
    )
    drv1_pose = PoseEvidence(
        confidence=0.94,
        face_points=((260.0, 220.0), (240.0, 205.0), (280.0, 205.0)),
        hand_points=((560.0, 340.0), (290.0, 280.0)),
        xyxy=(40.0, 120.0, 600.0, 680.0),
        shoulder_points=((180.0, 340.0), (360.0, 345.0)),
    )
    drv2_pose = PoseEvidence(
        confidence=0.94,
        face_points=((900.0, 220.0), (880.0, 205.0), (920.0, 205.0)),
        hand_points=((1200.0, 340.0), (930.0, 280.0)),
        xyxy=(680.0, 120.0, 1240.0, 680.0),
        shoulder_points=((820.0, 340.0), (1000.0, 345.0)),
    )
    assigned_dual = assoc.assign_cabin_upper_bodies(
        [drv1_ub, drv2_ub], w, h, poses=[drv1_pose, drv2_pose]
    )
    assert [a.role for _, a in assigned_dual] == ["driver", "driver"]

    # 2) Single clipped passenger at edge with no driver in frame
    pass_only_ub = NormalizedDetection(
        0, "occupant_upper_body", 0.88, (0.0, 380.0, 240.0, 710.0), track_id=5
    )
    pass_only_pose = PoseEvidence(
        confidence=0.82,
        face_points=((60.0, 440.0), (40.0, 430.0)),
        hand_points=((90.0, 560.0),),
        xyxy=(0.0, 380.0, 240.0, 710.0),
        shoulder_points=((20.0, 500.0), (95.0, 510.0)),
    )
    assigned_pass_only = assoc.assign_cabin_upper_bodies(
        [pass_only_ub], w, h, poses=[pass_only_pose]
    )
    assert assigned_pass_only[0][1].role == "front_passenger"

