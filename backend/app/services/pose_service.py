"""Orchestrate frame-by-frame pose estimation, skeleton overlay, and JSON export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.cv.draw import draw_skeleton
from app.cv.mmpose_estimator import MMPoseEstimator
from app.services.video_service import pose_json_path_for, process_video_frames


class PoseService:
    def __init__(self) -> None:
        self._estimator: MMPoseEstimator | None = None

    @property
    def estimator(self) -> MMPoseEstimator:
        if self._estimator is None:
            self._estimator = MMPoseEstimator()
        return self._estimator

    def analyze_video(self, input_path: Path, output_path: Path) -> tuple[Path, Path]:
        """Process video and write matching pose JSON next to the output MP4."""
        estimator = self.estimator
        frame_records: list[dict[str, Any]] = []

        def annotate(frame, frame_index: int, fps: float):
            keypoints = estimator.predict(frame)
            frame_records.append(
                {
                    "frame_index": frame_index,
                    "timestamp": frame_index / fps if fps > 0 else 0.0,
                    "keypoints": keypoints,
                }
            )
            return draw_skeleton(frame, keypoints)

        process_video_frames(input_path, output_path, annotate)

        json_path = pose_json_path_for(output_path)
        payload = {
            "video": output_path.name,
            "frame_count": len(frame_records),
            "frames": frame_records,
        }
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return output_path, json_path


pose_service = PoseService()
