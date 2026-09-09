"""Lightweight temporal tracking: smooth centers and fill short gaps."""

from __future__ import annotations

from app.schemas.racket import RacketBBox, RacketPoint, RacketTrajectory


def apply_lightweight_tracking(
    trajectory: RacketTrajectory,
    *,
    max_gap: int = 3,
    max_jump: float = 0.25,
    tracked_confidence: float = 0.4,
) -> RacketTrajectory:
    """Reject large jumps, then linearly fill short missing gaps.

    ``max_jump`` is normalized image diagonal fraction for consecutive visible
    detections; outliers are treated as missing before gap fill.
    """
    if not trajectory.frames:
        return trajectory

    frames = [
        RacketPoint(
            frame_index=p.frame_index,
            timestamp=p.timestamp,
            x=p.x,
            y=p.y,
            bbox=_clone_bbox(p.bbox),
            visible=p.visible,
            confidence=p.confidence,
            hand=p.hand,
            tracked=False,
            interpolated=False,
        )
        for p in trajectory.frames
    ]

    # Drop implausible frame-to-frame jumps.
    last_xy: tuple[float, float] | None = None
    for i, point in enumerate(frames):
        if not point.visible or point.x is None or point.y is None:
            last_xy = None
            continue
        if last_xy is not None:
            dx = point.x - last_xy[0]
            dy = point.y - last_xy[1]
            if (dx * dx + dy * dy) ** 0.5 > max_jump:
                frames[i] = RacketPoint(
                    frame_index=point.frame_index,
                    timestamp=point.timestamp,
                    x=None,
                    y=None,
                    bbox=None,
                    visible=False,
                    confidence=0.0,
                    hand=point.hand,
                    tracked=False,
                    interpolated=False,
                )
                last_xy = None
                continue
        last_xy = (point.x, point.y)

    if max_gap > 0:
        frames = _fill_gaps(frames, max_gap=max_gap, tracked_confidence=tracked_confidence)

    missing = [
        f.frame_index
        for f in frames
        if not f.visible or f.x is None or f.y is None
    ]
    return RacketTrajectory(
        video=trajectory.video,
        backend=trajectory.backend,
        fps=trajectory.fps,
        width=trajectory.width,
        height=trajectory.height,
        hitting_hand=trajectory.hitting_hand,
        frames=frames,
        missing_frame_indices=missing,
        notes=trajectory.notes,
    )


def _clone_bbox(bbox: RacketBBox | None) -> RacketBBox | None:
    if bbox is None:
        return None
    return RacketBBox(x1=bbox.x1, y1=bbox.y1, x2=bbox.x2, y2=bbox.y2)


def _lerp_bbox(a: RacketBBox | None, b: RacketBBox | None, t: float) -> RacketBBox | None:
    if a is None or b is None:
        return None
    return RacketBBox(
        x1=a.x1 + (b.x1 - a.x1) * t,
        y1=a.y1 + (b.y1 - a.y1) * t,
        x2=a.x2 + (b.x2 - a.x2) * t,
        y2=a.y2 + (b.y2 - a.y2) * t,
    )


def _fill_gaps(
    frames: list[RacketPoint],
    *,
    max_gap: int,
    tracked_confidence: float,
) -> list[RacketPoint]:
    n = len(frames)
    i = 0
    while i < n:
        if frames[i].visible and frames[i].x is not None and frames[i].y is not None:
            i += 1
            continue
        start = i
        while i < n and not (
            frames[i].visible and frames[i].x is not None and frames[i].y is not None
        ):
            i += 1
        end = i
        gap_len = end - start
        left, right = start - 1, end
        if (
            gap_len <= max_gap
            and left >= 0
            and right < n
            and frames[left].visible
            and frames[right].visible
            and frames[left].x is not None
            and frames[left].y is not None
            and frames[right].x is not None
            and frames[right].y is not None
            and not frames[left].interpolated
            and not frames[right].interpolated
        ):
            x0, y0 = float(frames[left].x), float(frames[left].y)
            x1, y1 = float(frames[right].x), float(frames[right].y)
            span = right - left
            for k in range(start, end):
                t = (k - left) / span
                frames[k] = RacketPoint(
                    frame_index=frames[k].frame_index,
                    timestamp=frames[k].timestamp,
                    x=x0 + (x1 - x0) * t,
                    y=y0 + (y1 - y0) * t,
                    bbox=_lerp_bbox(frames[left].bbox, frames[right].bbox, t),
                    visible=True,
                    confidence=tracked_confidence,
                    hand=frames[left].hand or frames[right].hand,
                    tracked=True,
                    interpolated=True,
                )
    return frames
