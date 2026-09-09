"""Render a debug video that shows only the shuttle trajectory overlay."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.schemas.shuttle import ShuttleTrajectory


def render_shuttle_debug_video(
    video_path: Path,
    trajectory: ShuttleTrajectory,
    output_path: Path,
    *,
    trail_length: int = 16,
) -> Path:
    """Write MP4 with shuttle points / short trail; no pose overlay."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or trajectory.fps or 30.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or trajectory.width)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or trajectory.height)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open writer: {output_path}")

    by_frame = {p.frame_index: p for p in trajectory.frames}
    trail: list[tuple[int, int]] = []
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            # Dim source so the trajectory is the visual focus.
            canvas = (frame.astype(np.float32) * 0.35).astype(np.uint8)
            point = by_frame.get(frame_index)
            if (
                point is not None
                and point.visible
                and point.x is not None
                and point.y is not None
            ):
                px = int(round(point.x * width))
                py = int(round(point.y * height))
                trail.append((px, py))
                if len(trail) > trail_length:
                    trail = trail[-trail_length:]
                color = (0, 220, 255) if not point.interpolated else (180, 180, 80)
                for i in range(1, len(trail)):
                    cv2.line(canvas, trail[i - 1], trail[i], color, 2, cv2.LINE_AA)
                cv2.circle(canvas, (px, py), 6, color, -1, cv2.LINE_AA)
                label = "interp" if point.interpolated else f"{point.confidence:.2f}"
                cv2.putText(
                    canvas,
                    f"shuttle {label}",
                    (px + 10, max(20, py - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )
            else:
                trail = []
                cv2.putText(
                    canvas,
                    "shuttle missing",
                    (16, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (80, 80, 200),
                    1,
                    cv2.LINE_AA,
                )
            writer.write(canvas)
            frame_index += 1
    finally:
        capture.release()
        writer.release()
    return output_path
