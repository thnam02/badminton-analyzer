"""Compose skeleton, joint-metric, and HUD layers. No biomechanics recalculation."""

from __future__ import annotations

import numpy as np

from app.config import settings
from app.cv.layers.helpers import AnchorSmoother
from app.cv.layers.hud_layer import render_hud_layer
from app.cv.layers.joint_metrics_layer import render_joint_metrics_layer
from app.cv.layers.skeleton_layer import render_skeleton_layer
from app.schemas.angles import AngleFrame
from app.schemas.motion import MotionFrame
from app.schemas.phases import SmashPhase
from app.schemas.pose import PoseFrame


class AnnotationRenderer:
    """Stateful frame compositor (anchor EMA lives across frames)."""

    def __init__(self, *, anchor_smoothing: float | None = None) -> None:
        alpha = (
            settings.overlay_anchor_smoothing
            if anchor_smoothing is None
            else anchor_smoothing
        )
        self._smoother = AnchorSmoother(alpha=alpha)

    def reset(self) -> None:
        self._smoother.reset()

    def render(
        self,
        frame: np.ndarray,
        *,
        pose_frame: PoseFrame | None,
        angle_frame: AngleFrame | None,
        motion_frame: MotionFrame | None,
        phase: SmashPhase | None = None,
    ) -> np.ndarray:
        out = frame.copy()
        out = render_skeleton_layer(out, pose_frame)
        out = render_joint_metrics_layer(
            out,
            pose_frame=pose_frame,
            angle_frame=angle_frame,
            motion_frame=motion_frame,
            smoother=self._smoother,
        )
        out = render_hud_layer(
            out,
            pose_frame=pose_frame,
            angle_frame=angle_frame,
            motion_frame=motion_frame,
            phase=phase,
        )
        return out


def draw_metrics_overlay(
    frame: np.ndarray,
    *,
    pose_frame: PoseFrame | None,
    angle_frame: AngleFrame | None,
    motion_frame: MotionFrame | None,
    phase: SmashPhase | None = None,
    renderer: AnnotationRenderer | None = None,
) -> np.ndarray:
    active = renderer or AnnotationRenderer()
    return active.render(
        frame,
        pose_frame=pose_frame,
        angle_frame=angle_frame,
        motion_frame=motion_frame,
        phase=phase,
    )
