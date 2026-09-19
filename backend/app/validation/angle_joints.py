"""Angle names and COCO triplets used by offline angle validation.

Mirrors ``app.processing.angles`` geometry (interior angle at vertex).
Left-side triplets exist only for validation — production AngleSequence
currently stores right-side angles only.
"""

from __future__ import annotations

# (angle_name, proximal, vertex, distal)
ANGLE_TRIPLETS: tuple[tuple[str, str, str, str], ...] = (
    ("right_elbow", "right_shoulder", "right_elbow", "right_wrist"),
    ("left_elbow", "left_shoulder", "left_elbow", "left_wrist"),
    ("right_knee", "right_hip", "right_knee", "right_ankle"),
    ("left_knee", "left_hip", "left_knee", "left_ankle"),
    # Shoulder / arm angle: trunk (hip) – shoulder – elbow
    ("right_shoulder", "right_hip", "right_shoulder", "right_elbow"),
    ("left_shoulder", "left_hip", "left_shoulder", "left_elbow"),
)

VALIDATED_ANGLE_NAMES: tuple[str, ...] = tuple(t[0] for t in ANGLE_TRIPLETS)

# Angles available on production AngleFrame today.
PIPELINE_ANGLE_NAMES: tuple[str, ...] = (
    "right_elbow",
    "right_knee",
    "right_shoulder",
)

TRIPLET_BY_ANGLE: dict[str, tuple[str, str, str]] = {
    name: (proximal, vertex, distal)
    for name, proximal, vertex, distal in ANGLE_TRIPLETS
}
