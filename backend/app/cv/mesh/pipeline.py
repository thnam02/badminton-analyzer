"""End-to-end WHAM mesh recovery → debug overlay video."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from app.config import settings
from app.cv.mesh.align import log_sequence_alignment
from app.cv.mesh.backends import get_mesh_backend
from app.cv.mesh.backends.base import MeshRecoveryBackend, MeshRecoveryError
from app.cv.mesh.render import draw_reprojection_residuals, render_mesh_overlay
from app.cv.mesh.types import MeshSequence
from app.schemas.pose import PoseSequence
from app.services.video_service import process_video_frames

logger = logging.getLogger(__name__)


def recover_mesh_sequence(
    video_path: Path,
    *,
    pose_sequence: PoseSequence | None,
    image_width: int,
    image_height: int,
    backend: MeshRecoveryBackend | None = None,
) -> MeshSequence:
    active = backend or get_mesh_backend()
    if not active.is_available():
        raise MeshRecoveryError(
            f"Mesh backend '{active.name}' is not available. "
            "Install WHAM via backend/scripts/bootstrap_wham.sh and set "
            "MESH_BACKEND=wham + MESH_WHAM_ROOT."
        )
    sequence = active.recover(
        video_path,
        pose_sequence=pose_sequence,
        image_width=image_width,
        image_height=image_height,
    )
    if sequence.frame_count == 0:
        raise MeshRecoveryError("Mesh recovery produced zero frames")

    alignment = log_sequence_alignment(
        sequence,
        pose_sequence,
        image_width=image_width,
        image_height=image_height,
        min_confidence=settings.pose_confidence_threshold,
    )
    sequence.alignment = alignment
    if alignment.get("available") and "overall_mean_px" in alignment:
        sequence.notes = (
            f"{sequence.notes} "
            f"alignment_overall_mean_px={alignment['overall_mean_px']:.1f}"
        )
    return sequence


def render_mesh_debug_video(
    video_path: Path,
    mesh_output_path: Path,
    mesh_sequence: MeshSequence,
    *,
    pose_sequence: PoseSequence | None = None,
) -> Path:
    """Write `{uuid}_mesh.mp4` with semi-transparent SMPL mesh over the player."""
    pose_by = (
        {f.frame_index: f for f in pose_sequence.frames} if pose_sequence else {}
    )
    mesh_by = {f.frame_index: f for f in mesh_sequence.frames}
    alpha = settings.mesh_overlay_alpha
    show_reproj = settings.mesh_show_reprojection

    def annotate(frame: np.ndarray, frame_index: int, fps: float) -> np.ndarray:
        del fps
        out = frame.copy()
        mesh_frame = mesh_by.get(frame_index)
        if mesh_frame is None:
            cv2.putText(
                out,
                "mesh: missing frame",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                lineType=cv2.LINE_AA,
            )
            return out

        out = render_mesh_overlay(out, mesh_frame, alpha=alpha)
        if show_reproj:
            out, _ = draw_reprojection_residuals(
                out,
                mesh_frame,
                pose_by.get(frame_index),
                min_confidence=settings.pose_confidence_threshold,
            )
        cv2.putText(
            out,
            f"mesh:{mesh_sequence.backend} f={frame_index}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (240, 240, 240),
            2,
            lineType=cv2.LINE_AA,
        )
        return out

    process_video_frames(video_path, mesh_output_path, annotate)
    logger.info("Wrote mesh debug video %s (%d frames)", mesh_output_path, mesh_sequence.frame_count)
    return mesh_output_path


def probe_video_size(video_path: Path) -> tuple[int, int]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise MeshRecoveryError(f"Could not open video for mesh recovery: {video_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if width <= 0 or height <= 0:
        raise MeshRecoveryError("Invalid video dimensions for mesh recovery")
    return width, height
