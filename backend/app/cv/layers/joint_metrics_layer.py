"""Body-anchored joint metric labels and angle arcs (precomputed values only)."""

from __future__ import annotations

import numpy as np

from app.config import settings
from app.cv.layers import helpers as H
from app.schemas.angles import AngleFrame
from app.schemas.motion import MotionFrame
from app.schemas.pose import Keypoint, PoseFrame

# Metric display uses precomputed numbers; triples are only for arc geometry.
_ELBOW_TRIPLE = ("right_shoulder", "right_elbow", "right_wrist")
_KNEE_TRIPLE = ("right_hip", "right_knee", "right_ankle")


def render_joint_metrics_layer(
    frame: np.ndarray,
    *,
    pose_frame: PoseFrame | None,
    angle_frame: AngleFrame | None,
    motion_frame: MotionFrame | None,
    smoother: H.AnchorSmoother,
) -> np.ndarray:
    if pose_frame is None:
        return frame

    out = frame
    h, w = out.shape[:2]
    thr = settings.pose_confidence_threshold
    kps = pose_frame.keypoints

    # Right elbow angle + arc
    elbow_angle = angle_frame.right_elbow if angle_frame is not None else None
    elbow_pt = _joint_px(kps, "right_elbow", w, h, thr)
    elbow_pt = smoother.update("right_elbow", elbow_pt)
    if elbow_angle is not None and elbow_pt is not None:
        triple = _triple_px(kps, _ELBOW_TRIPLE, w, h, thr)
        if triple is not None:
            H.draw_angle_arc(out, triple[0], triple[1], triple[2])
        H.draw_label(out, f"{elbow_angle:.1f}°", elbow_pt, color=(0, 220, 255))

    # Right knee angle + arc
    knee_angle = angle_frame.right_knee if angle_frame is not None else None
    knee_pt = _joint_px(kps, "right_knee", w, h, thr)
    knee_pt = smoother.update("right_knee", knee_pt)
    if knee_angle is not None and knee_pt is not None:
        triple = _triple_px(kps, _KNEE_TRIPLE, w, h, thr)
        if triple is not None:
            H.draw_angle_arc(out, triple[0], triple[1], triple[2], color=(80, 255, 160))
        H.draw_label(out, f"{knee_angle:.1f}°", knee_pt, color=(80, 255, 160))

    # Right wrist speed
    wrist_speed = motion_frame.right_wrist_speed if motion_frame is not None else None
    wrist_pt = _joint_px(kps, "right_wrist", w, h, thr)
    wrist_pt = smoother.update("right_wrist", wrist_pt)
    if wrist_speed is not None and wrist_pt is not None:
        H.draw_label(
            out,
            f"{wrist_speed:.3f}",
            wrist_pt,
            color=(255, 200, 80),
        )

    return out


def _joint_px(
    keypoints: dict[str, Keypoint],
    name: str,
    width: int,
    height: int,
    thr: float,
) -> tuple[float, float] | None:
    kp = keypoints.get(name)
    if kp is None:
        return None
    return H.keypoint_to_px(kp, width, height, min_confidence=thr)


def _triple_px(
    keypoints: dict[str, Keypoint],
    names: tuple[str, str, str],
    width: int,
    height: int,
    thr: float,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None:
    pts = []
    for name in names:
        pt = _joint_px(keypoints, name, width, height, thr)
        if pt is None:
            return None
        pts.append(pt)
    return pts[0], pts[1], pts[2]
