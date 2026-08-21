"""Geometric joint-angle calculation from PoseSequence keypoints.

Independent of MMPose and temporal smoothing — consumes Keypoint data only.
Angles are interior angles at the middle joint, in degrees [0, 180].
"""

from __future__ import annotations

import math

from app.schemas.angles import AngleFrame, AngleSequence
from app.schemas.pose import Keypoint, PoseSequence

# (angle_name, proximal, vertex, distal)
_JOINT_TRIPLETS: tuple[tuple[str, str, str, str], ...] = (
    ("right_elbow", "right_shoulder", "right_elbow", "right_wrist"),
    ("right_knee", "right_hip", "right_knee", "right_ankle"),
    ("right_shoulder", "right_hip", "right_shoulder", "right_elbow"),
)


def angle_at_vertex(
    proximal: Keypoint,
    vertex: Keypoint,
    distal: Keypoint,
) -> float:
    """Interior angle at ``vertex`` formed by proximal–vertex–distal, in degrees."""
    v_px = proximal.x - vertex.x
    v_py = proximal.y - vertex.y
    v_dx = distal.x - vertex.x
    v_dy = distal.y - vertex.y

    norm_p = math.hypot(v_px, v_py)
    norm_d = math.hypot(v_dx, v_dy)
    if norm_p == 0.0 or norm_d == 0.0:
        raise ValueError("Degenerate keypoint geometry (zero-length segment)")

    cos_theta = (v_px * v_dx + v_py * v_dy) / (norm_p * norm_d)
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return math.degrees(math.acos(cos_theta))


def compute_joint_angle(
    keypoints: dict[str, Keypoint],
    proximal_name: str,
    vertex_name: str,
    distal_name: str,
    *,
    confidence_threshold: float,
) -> float | None:
    """Return angle in degrees, or None if any required keypoint is unusable."""
    proximal = keypoints.get(proximal_name)
    vertex = keypoints.get(vertex_name)
    distal = keypoints.get(distal_name)
    if proximal is None or vertex is None or distal is None:
        return None
    if (
        proximal.confidence < confidence_threshold
        or vertex.confidence < confidence_threshold
        or distal.confidence < confidence_threshold
    ):
        return None
    try:
        return angle_at_vertex(proximal, vertex, distal)
    except ValueError:
        return None


def compute_angle_sequence(
    sequence: PoseSequence,
    *,
    confidence_threshold: float = 0.5,
) -> AngleSequence:
    """Build per-frame right elbow / knee / shoulder angles from a pose sequence."""
    out = AngleSequence(video=sequence.video)
    for frame in sequence.frames:
        angles: dict[str, float | None] = {}
        for angle_name, proximal, vertex, distal in _JOINT_TRIPLETS:
            angles[angle_name] = compute_joint_angle(
                frame.keypoints,
                proximal,
                vertex,
                distal,
                confidence_threshold=confidence_threshold,
            )
        out.append(
            AngleFrame(
                frame_index=frame.frame_index,
                timestamp=frame.timestamp,
                right_elbow=angles["right_elbow"],
                right_knee=angles["right_knee"],
                right_shoulder=angles["right_shoulder"],
            )
        )
    return out
