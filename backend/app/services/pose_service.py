"""Orchestrate frame-by-frame pose estimation, skeleton overlay, and JSON export."""

from __future__ import annotations

from pathlib import Path

from app.config import settings
from app.cv.draw import draw_skeleton
from app.cv.mmpose_estimator import MMPoseEstimator
from app.processing.temporal import preprocess_pose_sequence
from app.schemas.pose import PoseFrame, PoseSequence
from app.services.video_service import (
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
    ) -> tuple[Path, Path, Path, PoseSequence, PoseSequence]:
        """Process video; return paths plus raw and smoothed PoseSequence."""
        estimator = self.estimator
        raw_sequence = PoseSequence(video=output_path.name)

        def annotate(frame, frame_index: int, fps: float):
            keypoints = estimator.predict(frame)
            raw_sequence.append(
                PoseFrame(
                    frame_index=frame_index,
                    timestamp=frame_index / fps if fps > 0 else 0.0,
                    keypoints=keypoints,
                )
            )
            return draw_skeleton(frame, keypoints)

        process_video_frames(input_path, output_path, annotate)

        smoothed_sequence = preprocess_pose_sequence(
            raw_sequence,
            confidence_threshold=settings.pose_confidence_threshold,
            max_gap=settings.pose_interp_max_gap,
            savgol_window=settings.pose_savgol_window,
            savgol_polyorder=settings.pose_savgol_polyorder,
        )

        raw_json_path = pose_json_path_for(output_path)
        smoothed_json_path = smoothed_pose_json_path_for(output_path)
        raw_sequence.save_json(raw_json_path)
        smoothed_sequence.save_json(smoothed_json_path)
        return (
            output_path,
            raw_json_path,
            smoothed_json_path,
            raw_sequence,
            smoothed_sequence,
        )


pose_service = PoseService()
