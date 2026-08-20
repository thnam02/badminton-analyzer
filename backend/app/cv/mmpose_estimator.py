"""MMPose / RTMPose wrapper returning normalized keypoints."""

from __future__ import annotations

from typing import Any

import numpy as np

from app.config import settings
from app.cv.skeleton import COCO_KEYPOINT_NAMES


class MMPoseEstimator:
    """Thin adapter around MMPoseInferencer (RTMPose)."""

    def __init__(self) -> None:
        from mmpose.apis import MMPoseInferencer

        init_kwargs: dict[str, Any] = {"device": settings.device}

        if settings.mmpose_config and settings.mmpose_checkpoint:
            init_kwargs["pose2d"] = settings.mmpose_config
            init_kwargs["pose2d_weights"] = settings.mmpose_checkpoint
        else:
            # Alias resolves to RTMPose-m (see mmpose model aliases).
            init_kwargs["pose2d"] = "human"

        self._inferencer = MMPoseInferencer(**init_kwargs)
        self._threshold = settings.pose_confidence_threshold

    def predict(self, frame: np.ndarray) -> dict[str, dict[str, float]]:
        """Run pose estimation on a BGR frame.

        Returns a mapping of joint name -> {x, y, confidence} with
        x/y normalized to [0, 1] relative to frame width/height.
        Uses the highest-confidence instance when multiple people appear.
        """
        height, width = frame.shape[:2]
        if height == 0 or width == 0:
            return {}

        result_generator = self._inferencer(
            frame,
            show=False,
            return_vis=False,
            return_datasample=False,
        )
        result = next(result_generator)
        predictions = result.get("predictions") or []
        if not predictions:
            return {}

        # predictions is List[List[dict]] — one list of instances per image.
        instances = predictions[0] if predictions else []
        if not instances:
            return {}

        best = max(instances, key=lambda inst: float(np.mean(inst.get("keypoint_scores", [0.0]))))
        keypoints = best.get("keypoints")
        scores = best.get("keypoint_scores")
        if keypoints is None or scores is None:
            return {}

        keypoints = np.asarray(keypoints, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32)

        named: dict[str, dict[str, float]] = {}
        for idx, name in enumerate(COCO_KEYPOINT_NAMES):
            if idx >= len(keypoints) or idx >= len(scores):
                break
            x_px, y_px = float(keypoints[idx][0]), float(keypoints[idx][1])
            conf = float(scores[idx])
            named[name] = {
                "x": x_px / width,
                "y": y_px / height,
                "confidence": conf,
            }
        return named
