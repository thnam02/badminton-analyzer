"""Optional short-gap interpolation for shuttle trajectories.

Only fills brief missing stretches; long gaps stay missing so we do not invent
flight paths. Pose / contact logic is untouched.
"""

from __future__ import annotations

from app.schemas.shuttle import ShuttlePoint, ShuttleTrajectory


def interpolate_short_gaps(
    trajectory: ShuttleTrajectory,
    *,
    max_gap: int = 3,
    interpolated_confidence: float = 0.35,
) -> ShuttleTrajectory:
    """Linearly interpolate x/y across gaps of length ``1..max_gap``.

    A gap is a contiguous run of non-visible frames bounded by two visible,
    non-interpolated detections. Gaps longer than ``max_gap`` are left as-is.
    """
    if max_gap <= 0 or not trajectory.frames:
        return trajectory

    frames = [ShuttlePoint(**p.to_dict()) for p in trajectory.frames]
    n = len(frames)
    i = 0
    while i < n:
        if frames[i].visible and frames[i].x is not None and frames[i].y is not None:
            i += 1
            continue
        # Start of a missing run.
        start = i
        while i < n and not (
            frames[i].visible and frames[i].x is not None and frames[i].y is not None
        ):
            i += 1
        end = i  # first visible after gap, or n
        gap_len = end - start
        left = start - 1
        right = end
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
                frames[k] = ShuttlePoint(
                    frame_index=frames[k].frame_index,
                    timestamp=frames[k].timestamp,
                    x=x0 + (x1 - x0) * t,
                    y=y0 + (y1 - y0) * t,
                    visible=True,
                    confidence=interpolated_confidence,
                    interpolated=True,
                )

    missing = [
        f.frame_index
        for f in frames
        if not f.visible or f.x is None or f.y is None
    ]
    return ShuttleTrajectory(
        video=trajectory.video,
        backend=trajectory.backend,
        fps=trajectory.fps,
        width=trajectory.width,
        height=trajectory.height,
        frames=frames,
        missing_frame_indices=missing,
        notes=trajectory.notes,
    )
