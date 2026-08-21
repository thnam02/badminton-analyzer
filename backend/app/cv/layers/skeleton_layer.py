"""Skeleton render layer."""

from __future__ import annotations

import numpy as np

from app.cv.draw import draw_skeleton
from app.schemas.pose import PoseFrame


def render_skeleton_layer(
    frame: np.ndarray,
    pose_frame: PoseFrame | None,
) -> np.ndarray:
    if pose_frame is None or not pose_frame.keypoints:
        return frame
    return draw_skeleton(frame, pose_frame.keypoints)
