"""DensePose-constrained muscle heat overlay renderer."""

from __future__ import annotations

import json
import logging
from typing import Protocol

import numpy as np

from app.config import settings
from app.cv.densepose.debug import (
    DensePoseDebugSession,
    log_frame_diagnostics,
    render_densepose_parts_overlay,
)
from app.cv.densepose.inferencer import DensePoseError
from app.cv.densepose.mapping import (
    DensePoseFrameResult,
    DensePoseInferenceDiagnostics,
)
from app.cv.muscles.activation import (
    InvolvementSmoother,
    MuscleActivationMapper,
    MuscleGroup,
)
from app.cv.muscles.atlas import MUSCLE_COLORS, build_muscle_masks
from app.schemas.phases import SmashPhase
from app.schemas.pose import PoseFrame

logger = logging.getLogger(__name__)


class MuscleOverlayError(DensePoseError):
    """Muscle overlay could not be produced for a frame."""


class DensePosePredictor(Protocol):
    def predict(
        self,
        frame_bgr: np.ndarray,
        *,
        frame_index: int = 0,
        pose_frame: PoseFrame | None = None,
    ) -> tuple[DensePoseFrameResult | None, DensePoseInferenceDiagnostics]: ...


class MuscleOverlayRenderer:
    """Render phase-driven muscle heatmaps clipped to DensePose body regions."""

    def __init__(
        self,
        *,
        inferencer: DensePosePredictor | None = None,
        mapper: MuscleActivationMapper | None = None,
        smoother: InvolvementSmoother | None = None,
        base_alpha: float | None = None,
        fail_loud: bool | None = None,
        debug_session: DensePoseDebugSession | None = None,
    ) -> None:
        self._inferencer = inferencer
        self._mapper = mapper or MuscleActivationMapper()
        self._smoother = smoother or InvolvementSmoother(
            alpha=settings.overlay_muscle_smoothing
        )
        self._base_alpha = (
            settings.overlay_muscle_base_alpha
            if base_alpha is None
            else base_alpha
        )
        self._fail_loud = (
            settings.densepose_fail_loud if fail_loud is None else fail_loud
        )
        self._debug_session = debug_session

    @property
    def inferencer(self) -> DensePosePredictor | None:
        return self._inferencer

    @property
    def debug_session(self) -> DensePoseDebugSession | None:
        return self._debug_session

    def set_debug_session(self, session: DensePoseDebugSession | None) -> None:
        self._debug_session = session

    def reset(self) -> None:
        self._smoother.reset()

    def ensure_ready(self) -> None:
        """Eager-load DensePose; raise if unavailable."""
        if self._inferencer is None:
            raise MuscleOverlayError(
                "Muscle overlay enabled but no DensePose inferencer is configured"
            )
        if hasattr(self._inferencer, "_ensure_predictor"):
            self._inferencer._ensure_predictor()  # type: ignore[attr-defined]

    def render(
        self,
        frame_bgr: np.ndarray,
        *,
        pose_frame: PoseFrame | None,
        phase: SmashPhase | None,
        frame_index: int = 0,
        enabled: bool = True,
        densepose: DensePoseFrameResult | None = None,
        diagnostics: DensePoseInferenceDiagnostics | None = None,
    ) -> np.ndarray:
        if not enabled:
            return frame_bgr

        if self._inferencer is None and densepose is None:
            self._raise_or_return(
                frame_bgr,
                MuscleOverlayError(
                    "Muscle overlay enabled but DensePose inferencer is missing"
                ),
            )

        diag = diagnostics
        dp = densepose
        if dp is None:
            assert self._inferencer is not None
            dp, diag = self._inferencer.predict(
                frame_bgr,
                frame_index=frame_index,
                pose_frame=pose_frame,
            )
        elif diag is None:
            diag = DensePoseInferenceDiagnostics(frame_index=frame_index)
            diag.update_from_result(dp)

        involvements = self._smoother.smooth(
            self._mapper.involvements_for_phase(phase)
        )
        h, w = frame_bgr.shape[:2]
        masks = build_muscle_masks(
            dp,
            pose_frame,
            w,
            h,
            min_confidence=settings.pose_confidence_threshold,
        )
        muscle_counts = {
            muscle.value: int(mask.sum()) if mask is not None else 0
            for muscle, mask in masks.items()
        }

        if settings.densepose_debug or self._debug_session is not None:
            log_frame_diagnostics(
                diag,
                involvements={
                    muscle.value: round(v, 4)
                    for muscle, v in involvements.items()
                },
                muscle_mask_counts=muscle_counts,
            )

        if self._debug_session is not None and self._debug_session.should_save(
            frame_index
        ):
            self._debug_session.record(
                frame_index,
                frame_bgr,
                densepose=dp,
                diagnostics=diag,
                involvements={
                    muscle.value: round(v, 4)
                    for muscle, v in involvements.items()
                },
                muscle_masks={
                    muscle.value: mask for muscle, mask in masks.items()
                },
            )

        if dp is None or diag.error or dp.nonzero_pixel_count() == 0:
            self._raise_or_return(
                frame_bgr,
                MuscleOverlayError(
                    f"DensePose failed on frame {frame_index}: "
                    f"{json.dumps(diag.to_dict())}"
                ),
            )

        if not any(count > 0 for count in muscle_counts.values()):
            activation_summary = {
                muscle.value: round(v, 3) for muscle, v in involvements.items()
            }
            self._raise_or_return(
                frame_bgr,
                MuscleOverlayError(
                    f"All muscle masks empty on frame {frame_index}: "
                    f"densepose_parts={diag.detected_part_ids}, "
                    f"muscle_counts={muscle_counts}, "
                    f"activation={activation_summary}"
                ),
            )

        if settings.densepose_debug_show_parts:
            return render_densepose_parts_overlay(frame_bgr, dp)

        out = composite_muscle_heatmap(frame_bgr, masks, involvements, self._base_alpha)
        if np.array_equal(out, frame_bgr):
            self._raise_or_return(
                frame_bgr,
                MuscleOverlayError(
                    f"Muscle composite produced no visible change on frame {frame_index}; "
                    f"muscle_counts={muscle_counts}"
                ),
            )
        return out

    def _raise_or_return(
        self, frame_bgr: np.ndarray, exc: MuscleOverlayError
    ) -> np.ndarray:
        if self._fail_loud:
            raise exc
        logger.error(str(exc))
        return frame_bgr


def composite_muscle_heatmap(
    frame_bgr: np.ndarray,
    masks: dict[MuscleGroup, np.ndarray | None],
    involvements: dict[MuscleGroup, float],
    base_alpha: float,
) -> np.ndarray:
    """Alpha-blend sports-science style heat onto ``frame_bgr``."""
    h, w = frame_bgr.shape[:2]
    heat = np.zeros((h, w, 3), dtype=np.float32)
    weight = np.zeros((h, w), dtype=np.float32)
    alpha_map = np.zeros((h, w), dtype=np.float32)

    for muscle, mask in masks.items():
        if mask is None:
            continue
        inv = involvements.get(muscle, 0.0)
        if inv <= 0.0:
            continue
        color = np.array(MUSCLE_COLORS[muscle], dtype=np.float32)
        heat[mask] += color * inv
        weight[mask] += inv
        alpha_map[mask] = np.maximum(alpha_map[mask], inv * base_alpha)

    if not np.any(weight > 0):
        return frame_bgr

    valid = weight > 0
    heat[valid] /= weight[valid, None]

    frame_f = frame_bgr.astype(np.float32)
    alpha3 = alpha_map[..., None]
    blended = frame_f * (1.0 - alpha3) + heat * alpha3
    return np.clip(blended, 0, 255).astype(np.uint8)
