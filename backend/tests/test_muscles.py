"""Tests for DensePose-based muscle involvement overlay."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.cv.densepose.bbox import pose_player_bbox
from app.cv.densepose.debug import DensePoseDebugSession
from app.cv.densepose.mapping import DensePoseFrameResult, DensePoseInferenceDiagnostics, DensePosePart
from app.cv.muscles.activation import (
    InvolvementSmoother,
    MuscleActivationMapper,
    MuscleGroup,
)
from app.cv.muscles.atlas import build_muscle_masks
from app.cv.muscles.renderer import MuscleOverlayError, MuscleOverlayRenderer
from app.cv.overlay import AnnotationRenderer
from app.schemas.phases import SmashPhase
from app.schemas.pose import Keypoint, PoseFrame


def _full_pose() -> PoseFrame:
    return PoseFrame(
        frame_index=0,
        timestamp=0.0,
        keypoints={
            "left_shoulder": Keypoint(0.35, 0.30, 0.95),
            "right_shoulder": Keypoint(0.45, 0.30, 0.95),
            "left_hip": Keypoint(0.36, 0.55, 0.95),
            "right_hip": Keypoint(0.44, 0.55, 0.95),
            "right_elbow": Keypoint(0.58, 0.32, 0.95),
            "right_wrist": Keypoint(0.70, 0.36, 0.95),
            "right_knee": Keypoint(0.44, 0.72, 0.95),
            "right_ankle": Keypoint(0.46, 0.88, 0.95),
        },
    )


def _synthetic_densepose(width: int = 320, height: int = 240) -> DensePoseFrameResult:
    part_masks: dict[int, np.ndarray] = {}

    def box(part: DensePosePart, x1: int, y1: int, x2: int, y2: int) -> None:
        m = np.zeros((height, width), dtype=bool)
        m[y1:y2, x1:x2] = True
        part_masks[int(part)] = m

    box(DensePosePart.TORSO, 110, 60, 210, 140)
    box(DensePosePart.RIGHT_UPPER_ARM, 200, 55, 240, 95)
    box(DensePosePart.RIGHT_LOWER_ARM, 235, 90, 275, 130)
    box(DensePosePart.RIGHT_UPPER_LEG, 150, 140, 190, 190)
    box(DensePosePart.RIGHT_LOWER_LEG, 155, 185, 190, 230)

    person_mask = np.zeros((height, width), dtype=bool)
    for mask in part_masks.values():
        person_mask |= mask

    return DensePoseFrameResult(
        person_mask=person_mask,
        part_masks=part_masks,
        confidence=0.92,
        bbox=(110, 55, 275, 230),
    )


class _FakeInferencer:
    def predict(
        self,
        frame_bgr: np.ndarray,
        *,
        frame_index: int = 0,
        pose_frame: PoseFrame | None = None,
    ) -> tuple[DensePoseFrameResult, DensePoseInferenceDiagnostics]:
        del pose_frame
        h, w = frame_bgr.shape[:2]
        result = _synthetic_densepose(w, h)
        diag = DensePoseInferenceDiagnostics(frame_index=frame_index)
        diag.update_from_result(result)
        return result, diag


def test_coarse_segm_handles_batched_layout() -> None:
    import torch

    from app.cv.densepose.inferencer import _coarse_segm_to_labels

    class _Out:
        def __init__(self, tensor: torch.Tensor) -> None:
            self.coarse_segm = tensor

    labels = _coarse_segm_to_labels(_Out(torch.randn(1, 15, 32, 32)))
    assert labels.shape == (32, 32)
    assert labels.dtype == np.uint8


def test_mapper_returns_bounded_involvement() -> None:
    mapper = MuscleActivationMapper()
    for phase in SmashPhase:
        for muscle in MuscleGroup:
            value = mapper.involvement(phase, muscle)
            assert 0.0 <= value <= 1.0


def test_pose_player_bbox_from_keypoints() -> None:
    bbox = pose_player_bbox(_full_pose(), 320, 240, min_confidence=0.5)
    assert bbox is not None
    x1, y1, x2, y2 = bbox
    assert x2 > x1 and y2 > y1


def test_renderer_paints_with_injected_densepose() -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    dp = _synthetic_densepose()
    diag = DensePoseInferenceDiagnostics(frame_index=0)
    diag.update_from_result(dp)
    renderer = MuscleOverlayRenderer(inferencer=_FakeInferencer(), fail_loud=True)
    out = renderer.render(
        frame,
        pose_frame=_full_pose(),
        phase=SmashPhase.ACCELERATION,
        densepose=dp,
        diagnostics=diag,
    )
    assert not np.array_equal(out, frame)


def test_renderer_fail_loud_on_missing_densepose() -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    renderer = MuscleOverlayRenderer(inferencer=None, fail_loud=True)
    with pytest.raises(MuscleOverlayError):
        renderer.render(frame, pose_frame=_full_pose(), phase=SmashPhase.ACCELERATION)


def test_renderer_fail_loud_on_empty_masks() -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    empty = DensePoseFrameResult(
        person_mask=np.zeros((240, 320), dtype=bool),
        part_masks={},
        confidence=0.9,
    )
    diag = DensePoseInferenceDiagnostics(frame_index=0)
    diag.update_from_result(empty)
    renderer = MuscleOverlayRenderer(fail_loud=True)
    with pytest.raises(MuscleOverlayError):
        renderer.render(
            frame,
            pose_frame=_full_pose(),
            phase=SmashPhase.ACCELERATION,
            densepose=empty,
            diagnostics=diag,
        )


def test_debug_session_writes_artifacts(tmp_path: Path) -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    dp = _synthetic_densepose()
    diag = DensePoseInferenceDiagnostics(frame_index=3)
    diag.update_from_result(dp)
    masks = build_muscle_masks(dp, _full_pose(), 320, 240, min_confidence=0.5)
    session = DensePoseDebugSession(output_dir=tmp_path, max_frames=2)
    session.record(
        3,
        frame,
        densepose=dp,
        diagnostics=diag,
        involvements={m.value: 0.8 for m in MuscleGroup},
        muscle_masks={m.value: masks[m] for m in MuscleGroup},
    )
    assert (tmp_path / "frame_00003_input.jpg").exists()
    assert (tmp_path / "frame_00003_densepose_parts.jpg").exists()
    assert (tmp_path / "frame_00003_muscle_masks.jpg").exists()
    assert (tmp_path / "frame_00003_diagnostics.json").exists()


def test_overlay_muscle_layer_before_skeleton() -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    pose = _full_pose()
    muscle_renderer = MuscleOverlayRenderer(inferencer=_FakeInferencer(), fail_loud=True)
    with_muscles = AnnotationRenderer(
        anchor_smoothing=1.0,
        muscle_overlay=True,
        muscle_renderer=muscle_renderer,
    ).render(
        frame,
        pose_frame=pose,
        angle_frame=None,
        motion_frame=None,
        phase=SmashPhase.ACCELERATION,
    )
    without_muscles = AnnotationRenderer(
        anchor_smoothing=1.0,
        muscle_overlay=False,
    ).render(
        frame,
        pose_frame=pose,
        angle_frame=None,
        motion_frame=None,
        phase=SmashPhase.ACCELERATION,
    )
    assert not np.array_equal(with_muscles, without_muscles)
