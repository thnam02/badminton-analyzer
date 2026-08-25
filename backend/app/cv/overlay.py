"""Compose DensePose muscle, skeleton, joint-metric, and HUD layers."""

from __future__ import annotations

import numpy as np

from app.config import settings
from app.cv.densepose.debug import DensePoseDebugSession
from app.cv.layers.helpers import AnchorSmoother
from app.cv.layers.hud_layer import render_hud_layer
from app.cv.layers.joint_metrics_layer import render_joint_metrics_layer
from app.cv.layers.skeleton_layer import render_skeleton_layer
from app.cv.muscles.renderer import MuscleOverlayRenderer
from app.schemas.angles import AngleFrame
from app.schemas.motion import MotionFrame
from app.schemas.phases import SmashPhase
from app.schemas.pose import PoseFrame


class AnnotationRenderer:
    """Stateful frame compositor (anchor + muscle involvement EMA across frames)."""

    def __init__(
        self,
        *,
        anchor_smoothing: float | None = None,
        muscle_overlay: bool | None = None,
        muscle_renderer: MuscleOverlayRenderer | None = None,
        debug_session: DensePoseDebugSession | None = None,
    ) -> None:
        alpha = (
            settings.overlay_anchor_smoothing
            if anchor_smoothing is None
            else anchor_smoothing
        )
        self._smoother = AnchorSmoother(alpha=alpha)
        self._muscle_overlay = (
            settings.overlay_muscle_enabled
            if muscle_overlay is None
            else muscle_overlay
        )
        self._muscle_renderer = muscle_renderer
        self._debug_session = debug_session

    def reset(self) -> None:
        self._smoother.reset()
        if self._muscle_renderer is not None:
            self._muscle_renderer.reset()

    @property
    def muscle_overlay_enabled(self) -> bool:
        return self._muscle_overlay

    def set_debug_session(self, session: DensePoseDebugSession | None) -> None:
        self._debug_session = session
        if self._muscle_renderer is not None:
            self._muscle_renderer.set_debug_session(session)

    def ensure_muscle_overlay_ready(self) -> None:
        if self._muscle_overlay:
            self._muscle_layer().ensure_ready()

    def _muscle_layer(self) -> MuscleOverlayRenderer:
        if self._muscle_renderer is None:
            from app.cv.densepose.inferencer import DensePoseInferencer

            self._muscle_renderer = MuscleOverlayRenderer(
                inferencer=DensePoseInferencer(),
                debug_session=self._debug_session,
            )
        return self._muscle_renderer

    def render(
        self,
        frame: np.ndarray,
        *,
        pose_frame: PoseFrame | None,
        angle_frame: AngleFrame | None,
        motion_frame: MotionFrame | None,
        phase: SmashPhase | None = None,
        frame_index: int = 0,
        muscle_overlay: bool | None = None,
    ) -> np.ndarray:
        show_muscles = (
            self._muscle_overlay if muscle_overlay is None else muscle_overlay
        )
        out = frame.copy()
        if show_muscles:
            out = self._muscle_layer().render(
                out,
                pose_frame=pose_frame,
                phase=phase,
                frame_index=frame_index,
                enabled=True,
            )
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
    frame_index: int = 0,
    muscle_overlay: bool | None = None,
    renderer: AnnotationRenderer | None = None,
) -> np.ndarray:
    active = renderer or AnnotationRenderer(muscle_overlay=muscle_overlay)
    return active.render(
        frame,
        pose_frame=pose_frame,
        angle_frame=angle_frame,
        motion_frame=motion_frame,
        phase=phase,
        frame_index=frame_index,
        muscle_overlay=muscle_overlay,
    )
