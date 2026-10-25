"""ROADWATCH V21 — Action Recognition Inference Adapter.

Integrates:
1. RuleGuidedPoseClassifier (geometric keypoint prior, fast & high SAFE_DRIVING recall)
2. HierarchicalResNetGRUClassifier (V6 pretrained hierarchical action classifier)
3. BinaryGateResNetGRUClassifier (V7 pretrained binary-gated action classifier)

Strict Compliance:
- No synthetic or fabricated scores.
- Clear calibration status (UNCALIBRATED_OVR_SIGMOID or RULE_GUIDED_PRIOR).
- Respects LOSO cross-validation folds (does not load model trained on the test subject).
- Outputs explicit SAFE_DRIVING, PHONE_USE, OTHER_ACTIVITY probabilities and uncertainty.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torchvision.models import ResNet18_Weights, resnet18

from backend.ai.auxiliary import PoseEvidence
from training.dmd.action_checkpoint import (
    CheckpointSafetyError,
    sha256_file,
    validate_checkpoint_manifest,
)
from training.dmd.action_model import RuleGuidedPoseClassifier
from training.dmd.v6_action_hierarchy import (
    HierarchicalResNetGRUClassifier,
)
from training.dmd.v7_action_models import (
    BinaryGateResNetGRUClassifier,
    binary_gate_probabilities,
)

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"


@dataclass(frozen=True)
class ActionScores:
    safe_driving: float
    phone_use: float
    other_activity: float
    uncertainty: float
    predicted_action: str
    provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ActionInferenceAdapter:
    """Multi-tiered action inference adapter for Roadwatch V21."""

    def __init__(
        self,
        mode: str = "HYBRID_RULE_V6",  # "RULE_ONLY", "V6_HIERARCHICAL", "V7_BINARY_GATE", "HYBRID_RULE_V6"
        heldout_subject: str = "gC-14",
        device: str = "cpu",
        missing_checkpoint_policy: str = "RULE_ONLY",
        manifest_path: Path | None = None,
        causal_mode: str = "STRICT_CAUSAL_ONLINE",
    ):
        if mode not in {"RULE_ONLY", "V6_HIERARCHICAL", "V7_BINARY_GATE", "HYBRID_RULE_V6"}:
            raise ValueError(f"Unsupported action mode: {mode}")
        if missing_checkpoint_policy not in {"RULE_ONLY", "ERROR"}:
            raise ValueError("missing_checkpoint_policy must be RULE_ONLY or ERROR")
        if causal_mode not in {"STRICT_CAUSAL_ONLINE", "DECLARED_LATENCY_CENTERED"}:
            raise ValueError(f"Unsupported causal mode: {causal_mode}")
        self.mode = mode
        self.requested_mode = mode
        self.actual_mode = "RULE_ONLY"
        self.causal_mode = causal_mode
        self.declared_latency_seconds = 0.0
        self.forward_count = 0
        self.fallback_reason = None
        self.checkpoint_path: Path | None = None
        self.manifest_path = manifest_path
        self.contract: dict[str, Any] = {}
        self._stream_key = None
        self._last_timestamp = None
        self.heldout_subject = heldout_subject
        self.device = device

        self.rule_classifier = RuleGuidedPoseClassifier()
        self._resnet_backbone = None
        self._deep_model = None
        self._deep_model_sha = "NONE"
        self._deep_model_name = "NONE"

        # Temporal embedding buffer for deep temporal models (seq_len=9)
        self.embedding_buffer: deque[np.ndarray] = deque(maxlen=9)
        self.timestamp_buffer: deque[float] = deque(maxlen=9)

        try:
            if mode in ("V6_HIERARCHICAL", "HYBRID_RULE_V6"):
                self._init_v6_model()
            elif mode == "V7_BINARY_GATE":
                self._init_v7_model()
        except CheckpointSafetyError as exc:
            self.fallback_reason = str(exc)
            if missing_checkpoint_policy == "ERROR":
                raise

    def _get_backbone(self) -> torch.nn.Module:
        if self._resnet_backbone is None:
            weights = ResNet18_Weights.DEFAULT
            weights_path = Path(torch.hub.get_dir()) / "checkpoints" / Path(weights.url).name
            if not weights_path.is_file() or sha256_file(weights_path) != self.contract.get(
                "backbone_sha256"
            ):
                raise CheckpointSafetyError("FEATURE_EXTRACTOR_SHA_MISMATCH")
            model = resnet18(weights=weights)
            model.fc = torch.nn.Identity()
            model.eval()
            for param in model.parameters():
                param.requires_grad_(False)
            self._resnet_backbone = model.to(self.device)
        return self._resnet_backbone

    def _init_v6_model(self) -> None:
        self._load_verified_model(
            MODELS_DIR
            / "candidate"
            / "v6_action_ablation_001"
            / f"hierarchical_loss_only_loso_{self.heldout_subject}.pt",
            HierarchicalResNetGRUClassifier,
        )

    def _init_v7_model(self) -> None:
        self._load_verified_model(
            MODELS_DIR
            / "candidate"
            / "v7_research"
            / f"v7_d_binary_gate_loso_{self.heldout_subject}.pt",
            BinaryGateResNetGRUClassifier,
        )

    def _load_verified_model(self, ckpt_path: Path, model_class: type) -> None:
        self.checkpoint_path = ckpt_path
        self.contract = validate_checkpoint_manifest(
            ckpt_path,
            self.manifest_path or ckpt_path.with_suffix(".manifest.json"),
            self.heldout_subject,
            model_class.__name__,
        )
        data = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        if data.get("heldout_subject") != self.heldout_subject or set(
            data.get("training_subjects", [])
        ) != set(self.contract["training_subject_ids"]):
            raise CheckpointSafetyError("CHECKPOINT_EMBEDDED_SPLIT_MISMATCH")
        model = model_class()
        try:
            model.load_state_dict(data["model_state_dict"], strict=True)
        except (RuntimeError, KeyError) as exc:
            raise CheckpointSafetyError("INCOMPATIBLE_CHECKPOINT_ARCHITECTURE") from exc
        # A centered training label cannot be presented as a causal current-frame score.
        # Offline head diagnostics or declared-latency inference remain possible.
        if (
            self.contract["feature_preprocessing"]["label_position"] == "CENTER"
            and self.causal_mode != "DECLARED_LATENCY_CENTERED"
        ):
            raise CheckpointSafetyError("CENTERED_TRAINING_NOT_VALIDATED_FOR_CAUSAL_GATING")
        if self.causal_mode == "DECLARED_LATENCY_CENTERED":
            seq_len = int(self.contract["feature_preprocessing"].get("sequence_length", 9))
            period = float(self.contract.get("sample_period_seconds", 0.5))
            self.declared_latency_seconds = (seq_len // 2) * period
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self._deep_model = model.to(self.device)
        self._deep_model_sha = self.contract["sha256"]
        self._deep_model_name = ckpt_path.name
        self.actual_mode = self.requested_mode

    def reset(self) -> None:
        self.embedding_buffer.clear()
        self.timestamp_buffer.clear()
        self._stream_key = None
        self._last_timestamp = None

    def observe_embedding(
        self, embedding: np.ndarray, stream_key: tuple, timestamp: float
    ) -> np.ndarray | None:
        if embedding.shape != (512,) or not np.isfinite(embedding).all():
            raise ValueError("GRU requires a finite 512-dimensional embedding")
        period = float(self.contract.get("sample_period_seconds", 0.5))
        if stream_key == self._stream_key and self._last_timestamp is not None:
            if timestamp == self._last_timestamp:
                # Idempotent duplicate delivery: never append or forward twice.
                return None
            if timestamp < self._last_timestamp:
                raise CheckpointSafetyError("OUT_OF_ORDER_TEMPORAL_FEATURE")
        if stream_key != self._stream_key or (
            self._last_timestamp is not None
            and timestamp - self._last_timestamp > 1.5 * period
        ):
            self.reset()
        self._stream_key = stream_key
        self._last_timestamp = timestamp
        self.embedding_buffer.append(embedding)
        self.timestamp_buffer.append(float(timestamp))
        return np.stack(self.embedding_buffer) if len(self.embedding_buffer) == 9 else None

    def current_sequence_timestamps(self) -> tuple[float, ...]:
        """Return timestamps for the current complete temporal window."""
        return tuple(self.timestamp_buffer) if len(self.timestamp_buffer) == 9 else ()

    def provenance(self) -> dict[str, Any]:
        return {
            "requested_mode": self.requested_mode,
            "actual_mode": self.actual_mode,
            "causal_mode": self.causal_mode,
            "declared_latency_seconds": self.declared_latency_seconds,
            "forward_count": self.forward_count,
            "checkpoint_path": str(self.checkpoint_path) if self.checkpoint_path else None,
            "checkpoint_sha": self._deep_model_sha if self._deep_model is not None else None,
            "model_loaded": self._deep_model is not None,
            "fallback_reason": self.fallback_reason,
            "calibration_status": "UNCALIBRATED_SCORES",
        }

    def extract_crop_embedding(self, driver_crop: np.ndarray) -> np.ndarray:
        """Extract frozen ResNet-18 512-dim embedding from driver crop."""
        backbone = self._get_backbone()
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

        # Preprocess to 112x112 RGB
        resized = cv2.resize(driver_crop, (64, 64), interpolation=cv2.INTER_AREA)
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = (
            torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().div_(255.0).to(self.device)
        )
        tensor = torch.nn.functional.interpolate(
            tensor, size=(112, 112), mode="bilinear", align_corners=False
        )
        normed = (tensor - mean) / std

        with torch.inference_mode():
            emb = backbone(normed).cpu().numpy()[0]
        return emb.astype(np.float16).astype(np.float32)

    def _extract_pose_feature_vector(
        self, pose: PoseEvidence | None, phone_det_box: list[float] | None
    ) -> np.ndarray | None:
        """Construct 16-element pose geometry feature vector for RuleGuidedPoseClassifier."""
        if not pose or not pose.face_points or not pose.hand_points:
            return None

        # Approximate keypoints from available PoseEvidence
        # FACE_INDICES: nose (0), l_eye (1), r_eye (2), l_ear (3), r_ear (4)
        # HAND_INDICES: l_wrist (9), r_wrist (10)
        face_pts = pose.face_points
        hand_pts = pose.hand_points

        # Default centers
        nose = face_pts[0] if len(face_pts) > 0 else (0.0, 0.0)
        l_ear = face_pts[3] if len(face_pts) > 3 else (face_pts[1] if len(face_pts) > 1 else nose)
        r_ear = face_pts[4] if len(face_pts) > 4 else (face_pts[2] if len(face_pts) > 2 else nose)

        l_wrist = hand_pts[0] if len(hand_pts) > 0 else (0.0, 0.0)
        r_wrist = hand_pts[1] if len(hand_pts) > 1 else l_wrist

        # Normalize distances by pose bounding box diagonal if available
        p_box = pose.xyxy or (0.0, 0.0, 100.0, 100.0)
        diag = max(((p_box[2] - p_box[0]) ** 2 + (p_box[3] - p_box[1]) ** 2) ** 0.5, 1.0)

        def norm_dist(p1: tuple[float, float], p2: tuple[float, float]) -> float:
            return float(((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5) / diag

        lw_le = norm_dist(l_wrist, l_ear)
        lw_re = norm_dist(l_wrist, r_ear)
        rw_re = norm_dist(r_wrist, r_ear)
        rw_le = norm_dist(r_wrist, l_ear)
        lw_nose = norm_dist(l_wrist, nose)
        rw_nose = norm_dist(r_wrist, nose)

        # Phone distance to wrists and face
        if phone_det_box:
            ph_center = (
                (phone_det_box[0] + phone_det_box[2]) / 2,
                (phone_det_box[1] + phone_det_box[3]) / 2,
            )
            ph_lw = norm_dist(ph_center, l_wrist)
            ph_rw = norm_dist(ph_center, r_wrist)
            ph_face = norm_dist(ph_center, nose)
        else:
            ph_lw = 2.0
            ph_rw = 2.0
            ph_face = 2.0

        feat = np.array(
            [
                lw_le,
                lw_re,
                rw_re,
                rw_le,
                lw_nose,
                rw_nose,
                0.5,
                0.5,  # lw_ls, rw_rs placeholder
                60.0 if min(lw_le, lw_nose) < 0.3 else 120.0,  # l_elbow estimate
                60.0 if min(rw_re, rw_nose) < 0.3 else 120.0,  # r_elbow estimate
                ph_lw,
                ph_rw,
                ph_face,
                len(face_pts) + len(hand_pts),
                pose.confidence,
                diag,
            ],
            dtype=np.float32,
        )
        return feat

    def predict(
        self,
        driver_crop: np.ndarray | None,
        pose: PoseEvidence | None,
        phone_conf: float = 0.0,
        phone_box_local: list[float] | None = None,
        *,
        session_id: str = "UNSPECIFIED",
        camera_id: str = "UNSPECIFIED",
        occupant_track_id: str = "UNKNOWN",
        timestamp: float | None = None,
        embedding: np.ndarray | None = None,
        sequence: np.ndarray | None = None,
    ) -> ActionScores:
        """Computes comprehensive action scores."""
        # 1. Rule-guided pose classification
        pose_feat = self._extract_pose_feature_vector(pose, phone_box_local)
        rule_probs = self.rule_classifier.predict_probabilities(
            pose_feat, phone_detector_conf=phone_conf
        )

        rule_safe = float(rule_probs.normal)
        rule_phone = float(rule_probs.phonecall + rule_probs.texting)
        rule_other = float(rule_probs.other_gesture)

        # If only rule-based mode requested or deep model absent
        if self.mode == "RULE_ONLY" or self._deep_model is None:
            uncertainty = 1.0 - max(rule_safe, rule_phone, rule_other)
            best_action = (
                "SAFE_DRIVING"
                if rule_safe >= max(rule_phone, rule_other)
                else ("PHONE_USE" if rule_phone >= rule_other else "OTHER_ACTIVITY")
            )
            return ActionScores(
                safe_driving=rule_safe,
                phone_use=rule_phone,
                other_activity=rule_other,
                uncertainty=round(uncertainty, 4),
                predicted_action=best_action,
                provenance={
                    **self.provenance(),
                    "model_type": "RuleGuidedPoseClassifier",
                    "checkpoint_sha256": "DETERMINISTIC_PRIOR",
                    "calibration": "RULE_GUIDED_PRIOR",
                    "rule_phonecall": float(rule_probs.phonecall),
                    "rule_texting": float(rule_probs.texting),
                    "rule_predicted_class": rule_probs.predicted_class,
                },
            )

        # 2. Deep temporal sequence classification
        if sequence is not None:
            seq_array = np.asarray(sequence, dtype=np.float32)
            if seq_array.shape != (9, 512) or not np.isfinite(seq_array).all():
                raise ValueError("sequence must be a finite [9, 512] array")
        else:
            if embedding is not None:
                emb = embedding
            elif driver_crop is not None:
                emb = self.extract_crop_embedding(driver_crop)
            else:
                raise ValueError("Either driver_crop, embedding, or sequence must be provided")

            if (
                timestamp is None
                or session_id == "UNSPECIFIED"
                or camera_id == "UNSPECIFIED"
                or occupant_track_id == "UNKNOWN"
            ):
                raise CheckpointSafetyError("SEQUENCE_IDENTITY_AND_TIMING_REQUIRED")
            seq_array = self.observe_embedding(
                emb, (self.heldout_subject, session_id, camera_id, occupant_track_id), timestamp
            )
        if seq_array is None:
            return ActionScores(
                0.0,
                0.0,
                0.0,
                1.0,
                "UNKNOWN",
                {
                    **self.provenance(),
                    "reason": "INSUFFICIENT_SEQUENCE_HISTORY",
                },
            )

        seq_tensor = torch.from_numpy(seq_array).unsqueeze(0).float().to(self.device)

        with torch.inference_mode():
            self.forward_count += 1
            if isinstance(self._deep_model, HierarchicalResNetGRUClassifier):
                coarse_logits, phone_logits = self._deep_model(seq_tensor)
                coarse_probs = torch.softmax(coarse_logits, dim=1).cpu().numpy()[0]
                deep_safe = float(coarse_probs[0])
                deep_phone = float(coarse_probs[1])
                deep_other = float(coarse_probs[2])
            elif isinstance(self._deep_model, BinaryGateResNetGRUClassifier):
                b_logits, np_logits, p_logits = self._deep_model(seq_tensor)
                p_all = binary_gate_probabilities(b_logits, np_logits, p_logits).cpu().numpy()[0]
                deep_safe = float(p_all[0])
                deep_phone = float(p_all[1:5].sum())
                deep_other = float(p_all[5])
            else:
                deep_safe, deep_phone, deep_other = rule_safe, rule_phone, rule_other

        # 3. Hybrid fusion between Rule-guided prior and Deep ResNet-GRU
        # Rule classifier is strong at geometry (detecting hands on steering wheel = NORMAL = SAFE_DRIVING)
        # Deep model is strong at visual scene representations
        if self.mode == "HYBRID_RULE_V6":
            # If rule classifier strongly says NORMAL (steering wheel hands) and phone_conf is low:
            # Enforce SAFE_DRIVING to reject hard negatives!
            if rule_safe > 0.70 and phone_conf < 0.25:
                fused_safe = max(rule_safe, deep_safe)
                fused_phone = min(rule_phone, deep_phone)
                fused_other = rule_other
            else:
                fused_safe = 0.5 * rule_safe + 0.5 * deep_safe
                fused_phone = 0.5 * rule_phone + 0.5 * deep_phone
                fused_other = 0.5 * rule_other + 0.5 * deep_other

            # Normalize
            total = fused_safe + fused_phone + fused_other
            if total > 0:
                fused_safe /= total
                fused_phone /= total
                fused_other /= total
        else:
            fused_safe, fused_phone, fused_other = deep_safe, deep_phone, deep_other

        uncertainty = 1.0 - max(fused_safe, fused_phone, fused_other)
        best_action = (
            "SAFE_DRIVING"
            if fused_safe >= max(fused_phone, fused_other)
            else ("PHONE_USE" if fused_phone >= fused_other else "OTHER_ACTIVITY")
        )

        return ActionScores(
            safe_driving=round(fused_safe, 4),
            phone_use=round(fused_phone, 4),
            other_activity=round(fused_other, 4),
            uncertainty=round(uncertainty, 4),
            predicted_action=best_action,
            provenance={
                **self.provenance(),
                "model_type": f"{self.mode}:{self._deep_model_name}",
                "checkpoint_sha256": self._deep_model_sha,
                "heldout_subject": self.heldout_subject,
                "calibration": "UNCALIBRATED_SOFTMAX_SCORES",
                "rule_phonecall": float(rule_probs.phonecall),
                "rule_texting": float(rule_probs.texting),
                "rule_predicted_class": rule_probs.predicted_class,
            },
        )

    def predict_from_embedding(
        self,
        embedding: np.ndarray,
        pose: PoseEvidence | None,
        phone_conf: float = 0.0,
        phone_box_local: list[float] | None = None,
        *,
        session_id: str = "UNSPECIFIED",
        camera_id: str = "UNSPECIFIED",
        occupant_track_id: str = "UNKNOWN",
        timestamp: float | None = None,
    ) -> ActionScores:
        """Public API to evaluate action from precomputed/cached embedding."""
        return self.predict(
            driver_crop=None,
            pose=pose,
            phone_conf=phone_conf,
            phone_box_local=phone_box_local,
            session_id=session_id,
            camera_id=camera_id,
            occupant_track_id=occupant_track_id,
            timestamp=timestamp,
            embedding=embedding,
        )

    def predict_from_sequence(
        self,
        sequence: np.ndarray,
        pose: PoseEvidence | None,
        phone_conf: float = 0.0,
        phone_box_local: list[float] | None = None,
        *,
        session_id: str,
        camera_id: str,
        occupant_track_id: str,
        sequence_timestamps: tuple[float, ...],
        available_at_timestamp: float,
    ) -> ActionScores:
        """Score an already-observed window without mutating temporal state again.

        Center-labelled checkpoints target the middle element of the window.  The
        score is only available after the last element has arrived, so both times
        are recorded and must not be presented as an early causal decision.
        """
        if len(sequence_timestamps) != 9:
            raise ValueError("sequence_timestamps must contain 9 entries")
        if any(
            right <= left
            for left, right in zip(sequence_timestamps, sequence_timestamps[1:], strict=False)
        ):
            raise ValueError("sequence timestamps must be strictly increasing")
        # Stored feature timestamps retain full FPS precision while legacy traces
        # round query timestamps to milliseconds.  One millisecond is treated as
        # the same source frame only after frame identity has been validated.
        if available_at_timestamp + 1e-3 < sequence_timestamps[-1]:
            raise ValueError("score cannot be available before the final feature")
        scores = self.predict(
            driver_crop=None,
            pose=pose,
            phone_conf=phone_conf,
            phone_box_local=phone_box_local,
            session_id=session_id,
            camera_id=camera_id,
            occupant_track_id=occupant_track_id,
            timestamp=available_at_timestamp,
            sequence=sequence,
        )
        score_timestamp = sequence_timestamps[len(sequence_timestamps) // 2]
        timing = {
            "sequence_timestamps": list(sequence_timestamps),
            "score_timestamp": score_timestamp,
            "available_at_timestamp": float(max(available_at_timestamp, sequence_timestamps[-1])),
            "emission_latency_seconds": float(
                max(available_at_timestamp, sequence_timestamps[-1]) - score_timestamp
            ),
            "state_updated_by_call": False,
        }
        return replace(scores, provenance={**scores.provenance, **timing})
