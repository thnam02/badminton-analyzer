"""Text overlay for precomputed motion / angle metrics (no recalculation)."""

from __future__ import annotations

import cv2
import numpy as np

from app.schemas.angles import AngleFrame
from app.schemas.motion import MotionFrame
from app.schemas.pose import Keypoint, PoseFrame


def mean_pose_confidence(keypoints: dict[str, Keypoint]) -> float | None:
    """Average joint confidence for the frame, or None if no joints."""
    if not keypoints:
        return None
    return sum(kp.confidence for kp in keypoints.values()) / len(keypoints)


def format_overlay_lines(
    *,
    timestamp: float | None,
    right_elbow_angle: float | None,
    right_knee_angle: float | None,
    right_wrist_speed: float | None,
    pose_confidence: float | None,
) -> list[str]:
    """Build simple HUD lines; missing values become N/A."""
    return [
        f"Timestamp: {_fmt(timestamp, suffix='s', digits=3)}",
        f"Right elbow angle: {_fmt(right_elbow_angle, suffix=' deg', digits=1)}",
        f"Right knee angle: {_fmt(right_knee_angle, suffix=' deg', digits=1)}",
        f"Right wrist speed: {_fmt(right_wrist_speed, suffix='', digits=3)}",
        f"Pose confidence: {_fmt(pose_confidence, suffix='', digits=2)}",
    ]


def draw_metrics_overlay(
    frame: np.ndarray,
    *,
    pose_frame: PoseFrame | None,
    angle_frame: AngleFrame | None,
    motion_frame: MotionFrame | None,
) -> np.ndarray:
    """Draw a readable metrics panel; does not recompute any values."""
    annotated = frame.copy()
    timestamp = None
    if motion_frame is not None:
        timestamp = motion_frame.timestamp
    elif angle_frame is not None:
        timestamp = angle_frame.timestamp
    elif pose_frame is not None:
        timestamp = pose_frame.timestamp

    confidence = (
        mean_pose_confidence(pose_frame.keypoints) if pose_frame is not None else None
    )
    lines = format_overlay_lines(
        timestamp=timestamp,
        right_elbow_angle=angle_frame.right_elbow if angle_frame else None,
        right_knee_angle=angle_frame.right_knee if angle_frame else None,
        right_wrist_speed=motion_frame.right_wrist_speed if motion_frame else None,
        pose_confidence=confidence,
    )
    return _draw_text_panel(annotated, lines)


def _fmt(value: float | None, *, suffix: str, digits: int) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}{suffix}"


def _draw_text_panel(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    if not lines:
        return frame

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    pad = 10
    line_gap = 6

    sizes = [cv2.getTextSize(line, font, scale, thickness)[0] for line in lines]
    text_w = max(size[0] for size in sizes)
    text_h = sum(size[1] for size in sizes) + line_gap * (len(lines) - 1)
    box_w = text_w + pad * 2
    box_h = text_h + pad * 2

    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (8 + box_w, 8 + box_h), (20, 20, 20), thickness=-1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    y = 8 + pad + sizes[0][1]
    for i, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (8 + pad, y),
            font,
            scale,
            (240, 240, 240),
            thickness,
            lineType=cv2.LINE_AA,
        )
        if i + 1 < len(lines):
            y += sizes[i + 1][1] + line_gap
    return frame
