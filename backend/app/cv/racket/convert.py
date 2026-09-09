"""Convert raw racket detections into internal RacketTrajectory."""

from __future__ import annotations

from pathlib import Path

from app.cv.racket.base import RawRacketDetection
from app.schemas.racket import RacketBBox, RacketPoint, RacketTrajectory


def detections_to_trajectory(
    detections: list[RawRacketDetection],
    *,
    video_path: Path,
    backend: str,
    fps: float,
    width: int,
    height: int,
    hitting_hand: str | None = None,
) -> RacketTrajectory:
    if width <= 0 or height <= 0:
        raise ValueError("Video width/height must be positive")
    if fps <= 0:
        fps = 30.0

    frames: list[RacketPoint] = []
    missing: list[int] = []
    hand_votes: dict[str, int] = {}

    for det in sorted(detections, key=lambda d: d.frame_index):
        visible = (
            det.confidence > 0.0
            and det.x_px is not None
            and det.y_px is not None
        )
        hand = det.hand
        if hand in ("LEFT", "RIGHT"):
            hand_votes[hand] = hand_votes.get(hand, 0) + 1

        if visible:
            x = min(1.0, max(0.0, float(det.x_px) / float(width)))
            y = min(1.0, max(0.0, float(det.y_px) / float(height)))
            bbox = None
            if (
                det.x1_px is not None
                and det.y1_px is not None
                and det.x2_px is not None
                and det.y2_px is not None
            ):
                bbox = RacketBBox(
                    x1=min(1.0, max(0.0, float(det.x1_px) / float(width))),
                    y1=min(1.0, max(0.0, float(det.y1_px) / float(height))),
                    x2=min(1.0, max(0.0, float(det.x2_px) / float(width))),
                    y2=min(1.0, max(0.0, float(det.y2_px) / float(height))),
                )
            frames.append(
                RacketPoint(
                    frame_index=det.frame_index,
                    timestamp=det.frame_index / fps,
                    x=x,
                    y=y,
                    bbox=bbox,
                    visible=True,
                    confidence=float(max(0.0, min(1.0, det.confidence))),
                    hand=hand,
                    tracked=False,
                    interpolated=False,
                )
            )
        else:
            missing.append(det.frame_index)
            frames.append(
                RacketPoint(
                    frame_index=det.frame_index,
                    timestamp=det.frame_index / fps,
                    x=None,
                    y=None,
                    bbox=None,
                    visible=False,
                    confidence=0.0,
                    hand=hand,
                    tracked=False,
                    interpolated=False,
                )
            )

    if hitting_hand not in ("LEFT", "RIGHT"):
        if hand_votes:
            hitting_hand = max(hand_votes, key=hand_votes.get)
        else:
            hitting_hand = None

    return RacketTrajectory(
        video=str(video_path.name),
        backend=backend,
        fps=fps,
        width=width,
        height=height,
        hitting_hand=hitting_hand,
        frames=frames,
        missing_frame_indices=missing,
    )
