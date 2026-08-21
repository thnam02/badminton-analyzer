"""Orchestrate frame-by-frame pose estimation, skeleton overlay, and JSON export."""

from __future__ import annotations

from pathlib import Path

from app.config import settings
from app.cv.mmpose_estimator import MMPoseEstimator
from app.cv.overlay import AnnotationRenderer
from app.processing.angles import compute_angle_sequence
from app.processing.motion import compute_motion_derivatives
from app.processing.temporal import preprocess_pose_sequence
from app.schemas.angles import AngleSequence
from app.schemas.motion import MotionSequence
from app.schemas.pose import PoseFrame, PoseSequence
from app.services.video_service import (
    angles_json_path_for,
    iter_video_frames,
    motion_json_path_for,
    pose_json_path_for,
    process_video_frames,
    smoothed_pose_json_path_for,
)


class PoseService:
    def __init__(self) -> None:
        self._estimator: MMPoseEstimator | None = None

    @property
    def estimator(self) -> MMPoseEstimator:
        if self._estimator is None:
            self._estimator = MMPoseEstimator()
        return self._estimator

    def analyze_video(
        self, input_path: Path, output_path: Path
    ) -> tuple[
        Path,
        Path,
        Path,
        Path,
        Path,
        PoseSequence,
        PoseSequence,
        AngleSequence,
        MotionSequence,
    ]:
        """Process video; return artifact paths plus pose/angle/motion sequences."""
        estimator = self.estimator
        raw_sequence = PoseSequence(video=output_path.name)

        def collect_frame(frame, frame_index: int, fps: float) -> None:
            keypoints = estimator.predict(frame)
            raw_sequence.append(
                PoseFrame(
                    frame_index=frame_index,
                    timestamp=frame_index / fps if fps > 0 else 0.0,
                    keypoints=keypoints,
                )
            )

        iter_video_frames(input_path, collect_frame)

        smoothed_sequence = preprocess_pose_sequence(
            raw_sequence,
            confidence_threshold=settings.pose_confidence_threshold,
            max_gap=settings.pose_interp_max_gap,
            savgol_window=settings.pose_savgol_window,
            savgol_polyorder=settings.pose_savgol_polyorder,
        )
        angle_sequence = compute_angle_sequence(
            smoothed_sequence,
            confidence_threshold=settings.pose_confidence_threshold,
        )
        motion_sequence = compute_motion_derivatives(
            smoothed_sequence,
            angle_sequence,
            confidence_threshold=settings.pose_confidence_threshold,
        )

        pose_by_index = {f.frame_index: f for f in smoothed_sequence.frames}
        angle_by_index = {f.frame_index: f for f in angle_sequence.frames}
        motion_by_index = {f.frame_index: f for f in motion_sequence.frames}
        renderer = AnnotationRenderer()

        def render_frame(frame, frame_index: int, fps: float):
            del fps  # timestamps come from precomputed sequences
            return renderer.render(
                frame,
                pose_frame=pose_by_index.get(frame_index),
                angle_frame=angle_by_index.get(frame_index),
                motion_frame=motion_by_index.get(frame_index),
            )

        process_video_frames(input_path, output_path, render_frame)

        raw_json_path = pose_json_path_for(output_path)
        smoothed_json_path = smoothed_pose_json_path_for(output_path)
        angles_json_path = angles_json_path_for(output_path)
        motion_json_path = motion_json_path_for(output_path)
        raw_sequence.save_json(raw_json_path)
        smoothed_sequence.save_json(smoothed_json_path)
        angle_sequence.save_json(angles_json_path)
        motion_sequence.save_json(motion_json_path)
        return (
            output_path,
            raw_json_path,
            smoothed_json_path,
            angles_json_path,
            motion_json_path,
            raw_sequence,
            smoothed_sequence,
            angle_sequence,
            motion_sequence,
        )


pose_service = PoseService()
