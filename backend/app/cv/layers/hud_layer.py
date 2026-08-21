"""Global HUD layer: timestamp always; confidence warning only when low."""

from __future__ import annotations

import cv2
import numpy as np

from app.config import settings
from app.cv.layers.helpers import fmt_metric, mean_pose_confidence
from app.schemas.angles import AngleFrame
from app.schemas.motion import MotionFrame
from app.schemas.pose import PoseFrame


def render_hud_layer(
    frame: np.ndarray,
    *,
    pose_frame: PoseFrame | None,
    angle_frame: AngleFrame | None,
    motion_frame: MotionFrame | None,
) -> np.ndarray:
    out = frame
    timestamp = _timestamp(pose_frame, angle_frame, motion_frame)
    _draw_timestamp(out, timestamp)

    confidence = (
        mean_pose_confidence(pose_frame.keypoints) if pose_frame is not None else None
    )
    warn_below = settings.overlay_low_confidence_warn
    if confidence is not None and confidence < warn_below:
        _draw_low_confidence_warning(out, confidence)
    return out


def _timestamp(
    pose_frame: PoseFrame | None,
    angle_frame: AngleFrame | None,
    motion_frame: MotionFrame | None,
) -> float | None:
    if motion_frame is not None:
        return motion_frame.timestamp
    if angle_frame is not None:
        return angle_frame.timestamp
    if pose_frame is not None:
        return pose_frame.timestamp
    return None


def _draw_timestamp(frame: np.ndarray, timestamp: float | None) -> None:
    text = f"Timestamp: {fmt_metric(timestamp, digits=3, suffix='s')}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    pad = 6
    x, y = 10, 10 + th
    cv2.rectangle(
        frame,
        (x - pad, y - th - pad),
        (x + tw + pad, y + baseline + pad),
        (18, 18, 18),
        thickness=-1,
    )
    cv2.putText(
        frame,
        text,
        (x, y),
        font,
        scale,
        (235, 235, 235),
        thickness,
        lineType=cv2.LINE_AA,
    )


def _draw_low_confidence_warning(frame: np.ndarray, confidence: float) -> None:
    text = f"Low pose confidence: {confidence:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    pad = 6
    h, w = frame.shape[:2]
    x = max(10, w - tw - 20)
    y = 10 + th
    cv2.rectangle(
        frame,
        (x - pad, y - th - pad),
        (x + tw + pad, y + baseline + pad),
        (20, 40, 180),
        thickness=-1,
    )
    cv2.putText(
        frame,
        text,
        (x, y),
        font,
        scale,
        (255, 255, 255),
        thickness,
        lineType=cv2.LINE_AA,
    )
