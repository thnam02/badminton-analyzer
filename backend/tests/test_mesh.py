"""Tests for WHAM mesh overlay path (parser + render + alignment)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.cv.mesh.align import frame_alignment_errors, log_sequence_alignment
from app.cv.mesh.backends import get_mesh_backend
from app.cv.mesh.backends.base import MeshRecoveryError
from app.cv.mesh.backends.wham import WhamBackend, _parse_wham_results
from app.cv.mesh.render import render_mesh_overlay
from app.cv.mesh.types import CameraParams, MeshFrame, MeshSequence
from app.cv.overlay import AnnotationRenderer
from app.schemas.pose import Keypoint, PoseFrame, PoseSequence
from app.services.video_service import mesh_video_path_for


def _pose_seq() -> PoseSequence:
    frames = []
    for i in range(5):
        frames.append(
            PoseFrame(
                frame_index=i,
                timestamp=i * 0.04,
                keypoints={
                    "left_shoulder": Keypoint(0.40, 0.30, 0.95),
                    "right_shoulder": Keypoint(0.55, 0.30, 0.95),
                    "left_elbow": Keypoint(0.35, 0.40, 0.95),
                    "right_elbow": Keypoint(0.65, 0.38, 0.95),
                    "left_wrist": Keypoint(0.32, 0.50, 0.95),
                    "right_wrist": Keypoint(0.72, 0.45, 0.95),
                    "left_hip": Keypoint(0.42, 0.55, 0.95),
                    "right_hip": Keypoint(0.53, 0.55, 0.95),
                    "left_knee": Keypoint(0.42, 0.72, 0.95),
                    "right_knee": Keypoint(0.53, 0.72, 0.95),
                    "left_ankle": Keypoint(0.42, 0.88, 0.95),
                    "right_ankle": Keypoint(0.53, 0.88, 0.95),
                },
            )
        )
    return PoseSequence(video="clip.mp4", frames=frames)


def test_mesh_video_path_naming() -> None:
    path = mesh_video_path_for(Path("/tmp/abc_pose.mp4"))
    assert path.name == "abc_mesh.mp4"


def test_get_mesh_backend_defaults_to_wham() -> None:
    backend = get_mesh_backend("wham")
    assert isinstance(backend, WhamBackend)
    assert backend.name == "wham"


def test_get_mesh_backend_rejects_proxy() -> None:
    with pytest.raises(MeshRecoveryError, match="not supported"):
        get_mesh_backend("keypoint_proxy")


def test_parse_wham_verts_cam() -> None:
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    # Minimal 4-vert "mesh"; J_regressor maps to 3 joints for test.
    verts = np.zeros((3, 4, 3), dtype=np.float64)
    for t in range(3):
        verts[t] = np.array(
            [
                [-0.2, -0.2, 2.0 + 0.01 * t],
                [0.2, -0.2, 2.0 + 0.01 * t],
                [0.0, 0.2, 2.0 + 0.01 * t],
                [0.0, 0.0, 1.9 + 0.01 * t],
            ]
        )
    j_reg = np.zeros((24, 4), dtype=np.float64)
    j_reg[16, 0] = 1.0  # left_shoulder-ish
    j_reg[17, 1] = 1.0
    j_reg[1, 2] = 1.0
    j_reg[2, 3] = 1.0

    results = {
        0: {
            "verts_cam": verts,
            "frame_id": np.array([0, 1, 2]),
        }
    }
    seq = _parse_wham_results(
        results,
        video_name="clip.mp4",
        image_width=320,
        image_height=240,
        faces=faces,
        j_regressor=j_reg,
    )
    assert seq.backend == "wham"
    assert seq.frame_count == 3
    assert seq.frames[0].joints_3d is not None
    assert seq.frames[0].camera.fx == pytest.approx(np.sqrt(320**2 + 240**2))
    # Identity extrinsics for camera-space verts
    assert np.allclose(seq.frames[0].camera.R, np.eye(3))
    assert np.allclose(seq.frames[0].camera.t, 0)


def test_parse_wham_picks_longest_track() -> None:
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    short = np.zeros((2, 3, 3), dtype=np.float64)
    long = np.zeros((5, 3, 3), dtype=np.float64)
    long[..., 2] = 2.0
    results = {
        "a": {"verts": short},
        "b": {"verts_cam": long, "frame_ids": np.arange(5)},
    }
    seq = _parse_wham_results(
        results,
        video_name="clip.mp4",
        image_width=100,
        image_height=100,
        faces=faces,
    )
    assert seq.frame_count == 5


def test_alignment_errors_for_focus_joints() -> None:
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    verts = np.array([[0, 0, 2], [0.1, 0, 2], [0, 0.1, 2]], dtype=np.float64)
    joints = np.zeros((24, 3), dtype=np.float64)
    # Place right shoulder near image center projection
    joints[17] = [0.0, -0.2, 2.0]
    mesh = MeshFrame(
        frame_index=0,
        vertices=verts,
        faces=faces,
        camera=CameraParams.from_image_size(320, 240, focal_scale=1.0),
        joints_3d=joints,
    )
    pose = _pose_seq().frames[0]
    errs = frame_alignment_errors(
        mesh, pose, image_width=320, image_height=240, min_confidence=0.5
    )
    assert "right_shoulder" in errs
    assert errs["right_shoulder"] >= 0.0


def test_log_sequence_alignment_summary() -> None:
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    frames = []
    for i in range(5):
        joints = np.zeros((24, 3), dtype=np.float64)
        joints[16] = [-0.1, -0.2, 2.0]
        joints[17] = [0.1, -0.2, 2.0]
        joints[1] = [-0.05, 0.2, 2.0]
        joints[2] = [0.05, 0.2, 2.0]
        frames.append(
            MeshFrame(
                frame_index=i,
                vertices=np.zeros((3, 3)),
                faces=faces,
                camera=CameraParams.from_image_size(320, 240),
                joints_3d=joints,
            )
        )
    mesh_seq = MeshSequence(video="clip.mp4", frames=frames, backend="wham")
    summary = log_sequence_alignment(
        mesh_seq,
        _pose_seq(),
        image_width=320,
        image_height=240,
        min_confidence=0.5,
        sample_every=1,
    )
    assert summary["available"] is True
    assert summary["sampled_frames"] >= 1
    assert "overall_mean_px" in summary


def test_render_mesh_overlay_paints_pixels() -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    verts = np.array(
        [
            [-0.2, -0.2, 2.0],
            [0.2, -0.2, 2.0],
            [0.0, 0.2, 2.0],
            [0.0, 0.0, 1.8],
        ],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2], [0, 1, 3], [1, 2, 3], [0, 2, 3]], dtype=np.int32)
    mesh = MeshFrame(
        frame_index=0,
        vertices=verts,
        faces=faces,
        camera=CameraParams.from_image_size(320, 240),
    )
    out = render_mesh_overlay(frame, mesh, alpha=0.7)
    assert not np.array_equal(out, frame)


def test_annotation_renderer_skips_muscle_layer() -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    pose = _pose_seq().frames[0]
    out = AnnotationRenderer(muscle_overlay=True).render(
        frame,
        pose_frame=pose,
        angle_frame=None,
        motion_frame=None,
    )
    assert out.shape == frame.shape
    assert not np.array_equal(out, frame)
