"""Shared helpers for sports-analysis overlays (no biomechanics calc)."""

from __future__ import annotations

import math

import cv2
import numpy as np

from app.schemas.pose import Keypoint


def mean_pose_confidence(keypoints: dict[str, Keypoint]) -> float | None:
    if not keypoints:
        return None
    return sum(kp.confidence for kp in keypoints.values()) / len(keypoints)


def keypoint_to_px(
    kp: Keypoint,
    width: int,
    height: int,
    *,
    min_confidence: float,
) -> tuple[float, float] | None:
    if kp.confidence < min_confidence:
        return None
    return kp.x * width, kp.y * height


def fmt_metric(value: float | None, *, digits: int, suffix: str = "") -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}{suffix}"


class AnchorSmoother:
    """EMA smoother for label/arc anchor positions (reduces text jitter)."""

    def __init__(self, alpha: float = 0.35) -> None:
        self.alpha = max(0.0, min(1.0, float(alpha)))
        self._state: dict[str, tuple[float, float]] = {}

    def reset(self) -> None:
        self._state.clear()

    def update(self, key: str, xy: tuple[float, float] | None) -> tuple[float, float] | None:
        if xy is None:
            self._state.pop(key, None)
            return None
        if key not in self._state:
            self._state[key] = xy
            return xy
        px, py = self._state[key]
        nx = self.alpha * xy[0] + (1.0 - self.alpha) * px
        ny = self.alpha * xy[1] + (1.0 - self.alpha) * py
        self._state[key] = (nx, ny)
        return nx, ny


def clamp_label_origin(
    anchor_x: float,
    anchor_y: float,
    text_w: int,
    text_h: int,
    frame_w: int,
    frame_h: int,
    *,
    offset_x: int = 18,
    offset_y: int = -14,
    margin: int = 6,
) -> tuple[int, int]:
    """Place label near anchor; flip horizontally/vertically and clamp to edges."""
    x = int(round(anchor_x + offset_x))
    y = int(round(anchor_y + offset_y))

    if x + text_w + margin > frame_w:
        x = int(round(anchor_x - offset_x - text_w))
    if y - text_h < margin:
        y = int(round(anchor_y - offset_y + text_h))

    x = max(margin, min(x, frame_w - text_w - margin))
    y = max(margin + text_h, min(y, frame_h - margin))
    return x, y


def draw_label(
    frame: np.ndarray,
    text: str,
    anchor: tuple[float, float],
    *,
    color: tuple[int, int, int] = (245, 245, 245),
    scale: float = 0.45,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    h, w = frame.shape[:2]
    x, y = clamp_label_origin(anchor[0], anchor[1], tw, th + baseline, w, h)

    pad = 3
    cv2.rectangle(
        frame,
        (x - pad, y - th - pad),
        (x + tw + pad, y + baseline + pad),
        (15, 15, 15),
        thickness=-1,
    )
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, lineType=cv2.LINE_AA)


def draw_angle_arc(
    frame: np.ndarray,
    proximal: tuple[float, float],
    vertex: tuple[float, float],
    distal: tuple[float, float],
    *,
    color: tuple[int, int, int] = (0, 220, 255),
    radius: int = 22,
) -> None:
    """Draw a small arc at the joint from bone directions (rendering only)."""
    ang1 = math.degrees(math.atan2(-(proximal[1] - vertex[1]), proximal[0] - vertex[0]))
    ang2 = math.degrees(math.atan2(-(distal[1] - vertex[1]), distal[0] - vertex[0]))
    start = ang1
    end = ang2
    sweep = (end - start) % 360.0
    if sweep > 180.0:
        start, end = end, start
        sweep = (end - start) % 360.0

    center = (int(round(vertex[0])), int(round(vertex[1])))
    cv2.ellipse(
        frame,
        center,
        (radius, radius),
        0,
        -start,
        -(start + sweep),
        color,
        2,
        lineType=cv2.LINE_AA,
    )
