"""Joint name sets for pose validation (COCO-17 + badminton-critical)."""

from __future__ import annotations

from app.cv.skeleton import COCO_KEYPOINT_NAMES

COCO17_JOINTS: tuple[str, ...] = tuple(COCO_KEYPOINT_NAMES)

# Joints most important for smash / stroke biomechanics.
BADMINTON_CRITICAL_JOINTS: tuple[str, ...] = (
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)
