"""Convert tracker-native detections into internal ShuttleTrajectory."""

from __future__ import annotations

from pathlib import Path

from app.cv.shuttle.base import RawShuttleDetection
from app.schemas.shuttle import ShuttlePoint, ShuttleTrajectory


def detections_to_trajectory(
    detections: list[RawShuttleDetection],
    *,
    video_path: Path,
    backend: str,
    fps: float,
    width: int,
    height: int,
) -> ShuttleTrajectory:
    """Map pixel detections → normalized ShuttlePoint immediately after inference."""
    if width <= 0 or height <= 0:
        raise ValueError("Video width/height must be positive")
    if fps <= 0:
        fps = 30.0

    frames: list[ShuttlePoint] = []
    missing: list[int] = []
    for det in sorted(detections, key=lambda d: d.frame_index):
        visible = (
            det.visibility > 0
            and det.x_px is not None
            and det.y_px is not None
        )
        if visible:
            x = float(det.x_px) / float(width)
            y = float(det.y_px) / float(height)
            # Clamp lightly; TrackNet can occasionally emit near-border outliers.
            x = min(1.0, max(0.0, x))
            y = min(1.0, max(0.0, y))
            frames.append(
                ShuttlePoint(
                    frame_index=det.frame_index,
                    timestamp=det.frame_index / fps,
                    x=x,
                    y=y,
                    visible=True,
                    confidence=float(max(0.0, min(1.0, det.confidence))),
                    interpolated=False,
                )
            )
        else:
            missing.append(det.frame_index)
            frames.append(
                ShuttlePoint(
                    frame_index=det.frame_index,
                    timestamp=det.frame_index / fps,
                    x=None,
                    y=None,
                    visible=False,
                    confidence=0.0,
                    interpolated=False,
                )
            )

    return ShuttleTrajectory(
        video=str(video_path.name),
        backend=backend,
        fps=fps,
        width=width,
        height=height,
        frames=frames,
        missing_frame_indices=missing,
    )
