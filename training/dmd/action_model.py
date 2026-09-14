"""Lightweight Pose Action Classifier and Temporal Window Aggregator for Roadwatch V3.

Implements:
1. PoseActionClassifier: Lightweight classifier mapping normalized pose geometry and
   phone detection signals to driver action probabilities:
   - NORMAL (two hands on wheel, neutral driving)
   - PHONECALL (hand/phone to ear/face, acute elbow bend)
   - TEXTING (phone in hands, wrists together in lap/chest area)
   - OTHER_GESTURE (hair touching, scratching, drinking without phone)

2. TemporalWindowAggregator: Short context window (15, 20, 30 frames) rolling buffer
   that stabilizes frame-level predictions, prevents spurious single-frame triggers,
   and computes action persistence.
"""
from __future__ import annotations

import collections
import math
from dataclasses import asdict, dataclass
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

ACTION_CLASSES = ("NORMAL", "PHONECALL", "TEXTING", "OTHER_GESTURE")


@dataclass(frozen=True)
class ActionProbabilities:
    normal: float
    phonecall: float
    texting: float
    other_gesture: float
    predicted_class: str
    confidence: float

    def to_dict(self) -> Dict[str, float | str]:
        return asdict(self)


class RuleGuidedPoseClassifier:
    """Fast, deterministic, auditable pose classifier serving as the baseline / prior.

    Can be used standalone or fused with learned weights.
    Adheres strictly to the Roadwatch scientific principle:
    - Never infer phone use without physical phone evidence or strong proximal multi-view support.
    - Suppress false alarms from scratching / hair touching via OTHER_GESTURE class.
    """

    def __init__(
        self,
        call_elbow_max_deg: float = 85.0,
        call_ear_dist_max: float = 0.55,
        texting_dist_max: float = 0.60,
    ):
        self.call_elbow_max_deg = call_elbow_max_deg
        self.call_ear_dist_max = call_ear_dist_max
        self.texting_dist_max = texting_dist_max

    def predict_probabilities(
        self,
        pose_features: Optional[np.ndarray],
        phone_detector_conf: float = 0.0,
    ) -> ActionProbabilities:
        if pose_features is None or len(pose_features) < 16:
            # Neutral fallback
            return ActionProbabilities(
                normal=0.8,
                phonecall=0.0,
                texting=0.0,
                other_gesture=0.2,
                predicted_class="NORMAL",
                confidence=0.8,
            )

        (
            lw_le, lw_re, rw_re, rw_le,
            lw_nose, rw_nose,
            lw_ls, rw_rs,
            l_elbow, r_elbow,
            ph_lw, ph_rw, ph_face,
            vis_count, pose_conf, scale
        ) = pose_features[:16]

        # Geometry checks for Calling
        left_call_geom = (lw_le < self.call_ear_dist_max or lw_nose < self.call_ear_dist_max) and l_elbow < self.call_elbow_max_deg
        right_call_geom = (rw_re < self.call_ear_dist_max or rw_nose < self.call_ear_dist_max) and r_elbow < self.call_elbow_max_deg
        call_geom_score = 0.0
        if left_call_geom or right_call_geom:
            min_ear = min(lw_le, rw_re, lw_nose, rw_nose)
            call_geom_score = max(0.0, 1.0 - min_ear / self.call_ear_dist_max)

        # Geometry checks for Texting (wrists near each other or phone near wrist in lap/chest)
        min_wrist_phone = min(ph_lw, ph_rw)
        texting_geom_score = 0.0
        if min_wrist_phone < self.texting_dist_max:
            texting_geom_score = max(0.0, 1.0 - min_wrist_phone / self.texting_dist_max)

        # Probability derivation
        if phone_detector_conf > 0.15:
            # Physical phone is detected: elevate call or text depending on geometry
            p_call = call_geom_score * min(1.0, phone_detector_conf * 1.5)
            p_text = texting_geom_score * min(1.0, phone_detector_conf * 1.5)
            # If phone is detected but neither call nor text geometry matches strongly,
            # assign to texting if below shoulder or calling if near face
            if p_call < 0.2 and p_text < 0.2:
                if ph_face < 1.0:
                    p_call = 0.4
                else:
                    p_text = 0.4
            p_other = 0.05
            p_normal = max(0.0, 1.0 - (p_call + p_text + p_other))
        else:
            # No phone detected: hand near ear is scratching/hair adjust (OTHER_GESTURE)
            if call_geom_score > 0.3:
                p_other = min(0.9, 0.4 + call_geom_score * 0.5)
                p_call = 0.05  # low residual
                p_text = 0.0
                p_normal = max(0.0, 1.0 - (p_other + p_call))
            else:
                p_normal = 0.90
                p_other = 0.08
                p_call = 0.01
                p_text = 0.01

        # Normalize to sum to 1.0
        total = p_normal + p_call + p_text + p_other
        if total > 0:
            p_normal /= total
            p_call /= total
            p_text /= total
            p_other /= total

        probs = {
            "NORMAL": p_normal,
            "PHONECALL": p_call,
            "TEXTING": p_text,
            "OTHER_GESTURE": p_other,
        }
        best_cls = max(probs, key=probs.get)

        return ActionProbabilities(
            normal=float(p_normal),
            phonecall=float(p_call),
            texting=float(p_text),
            other_gesture=float(p_other),
            predicted_class=best_cls,
            confidence=float(probs[best_cls]),
        )


class TemporalWindowAggregator:
    """Short-window temporal aggregator (15, 20, 30 frames) for stabilizing actions."""

    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.buffer: Deque[ActionProbabilities] = collections.deque(maxlen=window_size)
        self.phone_confs: Deque[float] = collections.deque(maxlen=window_size)

    def push(self, probs: ActionProbabilities, phone_conf: float = 0.0) -> None:
        self.buffer.append(probs)
        self.phone_confs.append(phone_conf)

    def reset(self) -> None:
        self.buffer.clear()
        self.phone_confs.clear()

    @property
    def is_full(self) -> bool:
        return len(self.buffer) >= self.window_size

    def aggregate(self) -> Dict[str, float]:
        """Compute window-aggregated action scores."""
        if not self.buffer:
            return {
                "mean_phonecall": 0.0,
                "max_phonecall": 0.0,
                "mean_texting": 0.0,
                "max_texting": 0.0,
                "mean_normal": 1.0,
                "call_persistence": 0.0,
                "text_persistence": 0.0,
                "phone_presence_ratio": 0.0,
            }

        calls = [p.phonecall for p in self.buffer]
        texts = [p.texting for p in self.buffer]
        normals = [p.normal for p in self.buffer]
        phones = [c for c in self.phone_confs]

        # Persistence: fraction of frames where probability exceeded 0.35
        call_persist = sum(1 for c in calls if c > 0.35) / len(calls)
        text_persist = sum(1 for t in texts if t > 0.35) / len(texts)
        phone_pres = sum(1 for p in phones if p > 0.15) / len(phones)

        return {
            "mean_phonecall": float(np.mean(calls)),
            "max_phonecall": float(np.max(calls)),
            "mean_texting": float(np.mean(texts)),
            "max_texting": float(np.max(texts)),
            "mean_normal": float(np.mean(normals)),
            "call_persistence": float(call_persist),
            "text_persistence": float(text_persist),
            "phone_presence_ratio": float(phone_pres),
        }
