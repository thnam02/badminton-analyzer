"""Render a debug video showing only racket bbox / tip overlay."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.schemas.racket import RacketTrajectory


def render_racket_debug_video(
    video_path: Path,
    trajectory: RacketTrajectory,
    output_path: Path,
    *,
    trail_length: int = 12,
) -> Path:
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
    hand = trajectory.hitting_hand or "?"
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            canvas = (frame.astype(np.float32) * 0.4).astype(np.uint8)
            point = by_frame.get(frame_index)
            if (
                point is not None
                and point.visible
                and point.x is not None
                and point.y is not None
            ):
                px = int(round(point.x * width))
                py = int(round(point.y * height))
                color = (
                    (80, 200, 80)
                    if point.interpolated or point.tracked
                    else (40, 180, 255)
                )
                if point.bbox is not None:
                    p1 = (
                        int(round(point.bbox.x1 * width)),
                        int(round(point.bbox.y1 * height)),
                    )
                    p2 = (
                        int(round(point.bbox.x2 * width)),
                        int(round(point.bbox.y2 * height)),
                    )
                    cv2.rectangle(canvas, p1, p2, color, 2)
                trail.append((px, py))
                if len(trail) > trail_length:
                    trail = trail[-trail_length:]
                for i in range(1, len(trail)):
                    cv2.line(canvas, trail[i - 1], trail[i], color, 2, cv2.LINE_AA)
                cv2.circle(canvas, (px, py), 5, color, -1, cv2.LINE_AA)
                label = (
                    f"racket {hand} "
                    f"{'track' if point.tracked else f'{point.confidence:.2f}'}"
                )
                cv2.putText(
                    canvas,
                    label,
                    (px + 8, max(20, py - 8)),
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
                    "racket missing",
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
