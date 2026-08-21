"""Draw skeleton keypoints and bones onto video frames."""

from __future__ import annotations

import cv2
import numpy as np

from app.config import settings
from app.cv.skeleton import SKELETON_CONNECTIONS
from app.schemas.pose import Keypoint


def draw_skeleton(
    frame: np.ndarray,
    keypoints: dict[str, Keypoint],
    *,
    threshold: float | None = None,
) -> np.ndarray:
    """Draw joints and connections above the confidence threshold."""
    if not keypoints:
        return frame

    thr = settings.pose_confidence_threshold if threshold is None else threshold
    height, width = frame.shape[:2]
    annotated = frame.copy()

    def to_px(joint: Keypoint) -> tuple[int, int] | None:
        if joint.confidence < thr:
            return None
        x = int(round(joint.x * width))
        y = int(round(joint.y * height))
        return x, y

    # Bones first, then joints on top.
    for start_name, end_name in SKELETON_CONNECTIONS:
        start = keypoints.get(start_name)
        end = keypoints.get(end_name)
        if start is None or end is None:
            continue
        p1 = to_px(start)
        p2 = to_px(end)
        if p1 is None or p2 is None:
            continue
        cv2.line(annotated, p1, p2, (0, 255, 128), 2, lineType=cv2.LINE_AA)

    for joint in keypoints.values():
        pt = to_px(joint)
        if pt is None:
            continue
        cv2.circle(annotated, pt, 4, (0, 200, 255), -1, lineType=cv2.LINE_AA)
        cv2.circle(annotated, pt, 4, (20, 20, 20), 1, lineType=cv2.LINE_AA)

    return annotated
