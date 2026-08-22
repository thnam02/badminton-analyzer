"""Global HUD layer: timestamp + smash phase; confidence warning only when low."""

from __future__ import annotations

import cv2
import numpy as np

from app.config import settings
from app.cv.layers.helpers import fmt_metric, mean_pose_confidence
from app.schemas.angles import AngleFrame
from app.schemas.motion import MotionFrame
from app.schemas.phases import SmashPhase
from app.schemas.pose import PoseFrame


def render_hud_layer(
    frame: np.ndarray,
    *,
    pose_frame: PoseFrame | None,
    angle_frame: AngleFrame | None,
    motion_frame: MotionFrame | None,
    phase: SmashPhase | None = None,
) -> np.ndarray:
    out = frame
    timestamp = _timestamp(pose_frame, angle_frame, motion_frame)
    _draw_top_left_hud(out, timestamp, phase)

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


def _phase_label(phase: SmashPhase | None) -> str:
    if phase is None:
        return "Phase: N/A"
    if phase is SmashPhase.ESTIMATED_CONTACT:
        return "Phase: ESTIMATED_CONTACT"
    return f"Phase: {phase.value}"


def _draw_top_left_hud(
    frame: np.ndarray,
    timestamp: float | None,
    phase: SmashPhase | None,
) -> None:
    lines = [
        f"Timestamp: {fmt_metric(timestamp, digits=3, suffix='s')}",
        _phase_label(phase),
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    pad = 6
    gap = 4
    sizes = [cv2.getTextSize(line, font, scale, thickness)[0] for line in lines]
    text_w = max(w for w, _ in sizes)
    text_h = sum(h for _, h in sizes) + gap * (len(lines) - 1)
    x0, y0 = 10, 10
    cv2.rectangle(
        frame,
        (x0 - pad, y0 - pad),
        (x0 + text_w + pad, y0 + text_h + pad),
        (18, 18, 18),
        thickness=-1,
    )
    y = y0 + sizes[0][1]
    for i, line in enumerate(lines):
        color = (235, 235, 235)
        if phase is SmashPhase.ESTIMATED_CONTACT and i == 1:
            color = (80, 220, 255)
        cv2.putText(
            frame,
            line,
            (x0, y),
            font,
            scale,
            color,
            thickness,
            lineType=cv2.LINE_AA,
        )
        if i + 1 < len(lines):
            y += sizes[i + 1][1] + gap


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
