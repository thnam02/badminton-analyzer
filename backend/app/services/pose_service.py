"""Orchestrate frame-by-frame pose estimation and skeleton overlay."""

from __future__ import annotations

from pathlib import Path

from app.cv.draw import draw_skeleton
from app.cv.mmpose_estimator import MMPoseEstimator
from app.services.video_service import process_video_frames


class PoseService:
    def __init__(self) -> None:
        self._estimator: MMPoseEstimator | None = None

    @property
    def estimator(self) -> MMPoseEstimator:
        if self._estimator is None:
            self._estimator = MMPoseEstimator()
        return self._estimator

    def analyze_video(self, input_path: Path, output_path: Path) -> Path:
        estimator = self.estimator

        def annotate(frame):
            keypoints = estimator.predict(frame)
            return draw_skeleton(frame, keypoints)

        return process_video_frames(input_path, output_path, annotate)


pose_service = PoseService()
