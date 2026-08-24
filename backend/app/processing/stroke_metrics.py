"""Phase-specific smash feature extraction from angles + motion.

Independent of MMPose, overlay rendering, scoring, and shuttle tracking.
Consumes PhaseSequence windows and per-frame angle / motion metrics only.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Protocol, TypeVar

from app.schemas.angles import AngleSequence
from app.schemas.motion import MotionSequence, PeakStats
from app.schemas.phases import PhaseSegment, PhaseSequence, SmashPhase
from app.schemas.stroke import (
    AccelerationMetrics,
    BackswingMetrics,
    EstimatedContactMetrics,
    FollowThroughMetrics,
    PhaseWindow,
    PreparationMetrics,
    StrokeMetrics,
)

class _Indexed(Protocol):
    frame_index: int
    timestamp: float


T = TypeVar("T", bound=_Indexed)
Getter = Callable[[T], float | None]


def extract_stroke_metrics(
    phases: PhaseSequence,
    angles: AngleSequence,
    motion: MotionSequence,
) -> StrokeMetrics:
    """Pull per-phase extrema / means; values are None when a window or sample is missing."""
    angle_by = {frame.frame_index: frame for frame in angles.frames}
    motion_by = {frame.frame_index: frame for frame in motion.frames}
    video = phases.video or angles.video or motion.video

    prep_seg = _first_segment(phases, SmashPhase.PREPARATION)
    back_seg = _first_segment(phases, SmashPhase.BACKSWING)
    accel_seg = _first_segment(phases, SmashPhase.ACCELERATION)
    follow_seg = _first_segment(phases, SmashPhase.FOLLOW_THROUGH)

    contact_idx = phases.estimated_contact_frame_index
    contact_angle = angle_by.get(contact_idx) if contact_idx is not None else None
    contact_motion = motion_by.get(contact_idx) if contact_idx is not None else None

    accel_motion = _frames_in(motion.frames, accel_seg)
    back_angles = _frames_in(angles.frames, back_seg)
    back_motion = _frames_in(motion.frames, back_seg)
    prep_angles = _frames_in(angles.frames, prep_seg)
    prep_motion = _frames_in(motion.frames, prep_seg)
    follow_angles = _frames_in(angles.frames, follow_seg)
    follow_motion = _frames_in(motion.frames, follow_seg)

    return StrokeMetrics(
        video=video,
        preparation=PreparationMetrics(
            window=_window(prep_seg),
            mean_wrist_speed=_mean(prep_motion, lambda f: f.right_wrist_speed),
            mean_elbow_angle=_mean(prep_angles, lambda f: f.right_elbow),
            mean_knee_angle=_mean(prep_angles, lambda f: f.right_knee),
        ),
        backswing=BackswingMetrics(
            window=_window(back_seg),
            min_elbow_angle=_extremum(
                back_angles, lambda f: f.right_elbow, minimize=True
            ),
            peak_wrist_speed=_extremum(
                back_motion, lambda f: f.right_wrist_speed, minimize=False
            ),
        ),
        acceleration=AccelerationMetrics(
            window=_window(accel_seg),
            peak_wrist_speed=_extremum(
                accel_motion, lambda f: f.right_wrist_speed, minimize=False
            ),
            peak_elbow_angular_velocity=_extremum(
                accel_motion,
                lambda f: f.right_elbow_angular_velocity,
                minimize=False,
                use_abs=True,
            ),
        ),
        estimated_contact=EstimatedContactMetrics(
            frame_index=contact_idx,
            timestamp=(
                phases.estimated_contact_timestamp
                if phases.estimated_contact_timestamp is not None
                else contact_motion.timestamp
                if contact_motion is not None
                else contact_angle.timestamp
                if contact_angle is not None
                else None
            ),
            right_elbow_angle=_finite(
                contact_angle.right_elbow if contact_angle is not None else None
            ),
            right_knee_angle=_finite(
                contact_angle.right_knee if contact_angle is not None else None
            ),
            right_wrist_speed=_finite(
                contact_motion.right_wrist_speed if contact_motion is not None else None
            ),
        ),
        follow_through=FollowThroughMetrics(
            window=_window(follow_seg),
            mean_wrist_speed=_mean(follow_motion, lambda f: f.right_wrist_speed),
            wrist_speed_at_end=_value_at_end(
                follow_motion, follow_seg, lambda f: f.right_wrist_speed
            ),
            elbow_angle_at_end=_value_at_end(
                follow_angles, follow_seg, lambda f: f.right_elbow
            ),
        ),
    )


def _first_segment(phases: PhaseSequence, phase: SmashPhase) -> PhaseSegment | None:
    for segment in phases.segments:
        if segment.phase is phase:
            return segment
    return None


def _window(segment: PhaseSegment | None) -> PhaseWindow:
    if segment is None:
        return PhaseWindow()
    duration = segment.end_timestamp - segment.start_timestamp
    if not math.isfinite(duration) or duration < 0.0:
        duration_value: float | None = None
    else:
        duration_value = duration
    return PhaseWindow(
        start_frame_index=segment.start_frame_index,
        end_frame_index=segment.end_frame_index,
        start_timestamp=segment.start_timestamp,
        end_timestamp=segment.end_timestamp,
        duration=duration_value,
    )


def _frames_in(frames: Sequence[T], segment: PhaseSegment | None) -> list[T]:
    if segment is None:
        return []
    start = segment.start_frame_index
    end = segment.end_frame_index
    return [frame for frame in frames if start <= frame.frame_index <= end]


def _finite(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return value


def _mean(frames: Sequence[T], getter: Getter[T]) -> float | None:
    values = [v for v in (_finite(getter(frame)) for frame in frames) if v is not None]
    if not values:
        return None
    return float(sum(values) / len(values))


def _extremum(
    frames: Sequence[T],
    getter: Getter[T],
    *,
    minimize: bool,
    use_abs: bool = False,
) -> PeakStats:
    best_frame: T | None = None
    best_score = float("inf") if minimize else float("-inf")
    best_value: float | None = None

    for frame in frames:
        value = _finite(getter(frame))
        if value is None:
            continue
        score = abs(value) if use_abs else value
        better = score < best_score if minimize else score > best_score
        if better:
            best_score = score
            best_frame = frame
            best_value = value

    if best_frame is None or best_value is None:
        return PeakStats(value=None, frame_index=None, timestamp=None)

    peak_value = abs(best_value) if use_abs else best_value
    return PeakStats(
        value=peak_value,
        frame_index=best_frame.frame_index,
        timestamp=best_frame.timestamp,
    )


def _value_at_end(
    frames: Sequence[T],
    segment: PhaseSegment | None,
    getter: Getter[T],
) -> float | None:
    if segment is None:
        return None
    by_index = {frame.frame_index: frame for frame in frames}
    at_end = by_index.get(segment.end_frame_index)
    if at_end is not None:
        value = _finite(getter(at_end))
        if value is not None:
            return value
    for frame in reversed(list(frames)):
        value = _finite(getter(frame))
        if value is not None:
            return value
    return None
