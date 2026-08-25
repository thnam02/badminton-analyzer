"""Stroke metrics extraction helpers."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from statistics import mean
from typing import Protocol, TypeVar

from app.schemas.angles import AngleSequence
from app.schemas.motion import MotionSequence, PeakStats
from app.schemas.phases import PhaseSegment, PhaseSequence, SmashPhase
from app.schemas.pose import PoseSequence
from app.schemas.stroke import (
    AccelerationMetrics,
    BackswingMetrics,
    EstimatedContactMetrics,
    FollowThroughMetrics,
    PhaseWindow,
    PreparationMetrics,
    StrokeMetrics as PhaseStrokeMetrics,
)
from app.schemas.stroke_metrics import StrokeMetrics as LegacyStrokeMetrics


class _Indexed(Protocol):
    frame_index: int
    timestamp: float


T = TypeVar("T", bound=_Indexed)
Getter = Callable[[T], float | None]


def compute_stroke_metrics(
    pose: PoseSequence,
    angles: AngleSequence,
    motion: MotionSequence,
    phases: PhaseSequence,
) -> LegacyStrokeMetrics:
    """Aggregate phase-aware metrics for rule-based technique evaluation."""
    video = pose.video or angles.video or motion.video or phases.video
    metrics = LegacyStrokeMetrics(
        video=video,
        estimated_contact_frame_index=phases.estimated_contact_frame_index,
        estimated_contact_timestamp=phases.estimated_contact_timestamp,
        phase_confidence=phases.confidence,
    )

    contact_idx = phases.estimated_contact_frame_index
    if contact_idx is None:
        return metrics

    angle_by = {f.frame_index: f for f in angles.frames}
    motion_by = {f.frame_index: f for f in motion.frames}
    pose_by = {f.frame_index: f for f in pose.frames}

    contact_angle = angle_by.get(contact_idx)
    contact_motion = motion_by.get(contact_idx)
    contact_pose = pose_by.get(contact_idx)

    if contact_angle is not None:
        metrics.contact_elbow_angle_deg = contact_angle.right_elbow
        metrics.contact_knee_angle_deg = contact_angle.right_knee
        metrics.contact_shoulder_angle_deg = contact_angle.right_shoulder

    if contact_pose is not None:
        wrist = contact_pose.keypoints.get("right_wrist")
        if wrist is not None:
            metrics.contact_wrist_y_normalized = wrist.y

    peak = motion.peaks.get("right_wrist_speed")
    if peak is not None and peak.value is not None:
        metrics.peak_wrist_speed = peak.value
    elif contact_motion is not None:
        metrics.peak_wrist_speed = contact_motion.right_wrist_speed

    prep_knee = _phase_mean_angle(phases, angle_by, SmashPhase.PREPARATION, "knee")
    metrics.preparation_knee_angle_deg = prep_knee
    if prep_knee is not None and metrics.contact_knee_angle_deg is not None:
        metrics.knee_contribution_deg = metrics.contact_knee_angle_deg - prep_knee

    metrics.peak_elbow_omega_offset_frames = _peak_elbow_omega_offset(motion_by, contact_idx)
    metrics.acceleration_phase_fraction = _acceleration_fraction(phases, contact_idx)

    follow_ratio, follow_frames = _follow_through_stats(motion_by, phases)
    metrics.follow_through_speed_ratio = follow_ratio
    metrics.follow_through_frame_count = follow_frames

    return metrics


def extract_stroke_metrics(
    phases: PhaseSequence,
    angles: AngleSequence,
    motion: MotionSequence,
) -> PhaseStrokeMetrics:
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

    return PhaseStrokeMetrics(
        video=video,
        preparation=PreparationMetrics(
            window=_window(prep_seg),
            mean_wrist_speed=_mean(prep_motion, lambda f: f.right_wrist_speed),
            mean_elbow_angle=_mean(prep_angles, lambda f: f.right_elbow),
            mean_knee_angle=_mean(prep_angles, lambda f: f.right_knee),
        ),
        backswing=BackswingMetrics(
            window=_window(back_seg),
            min_elbow_angle=_extremum(back_angles, lambda f: f.right_elbow, minimize=True),
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


def _phase_mean_angle(
    phases: PhaseSequence,
    angle_by: dict,
    phase: SmashPhase,
    joint: str,
) -> float | None:
    values: list[float] = []
    for seg in phases.segments:
        if seg.phase is not phase:
            continue
        for idx in range(seg.start_frame_index, seg.end_frame_index + 1):
            af = angle_by.get(idx)
            if af is None:
                continue
            val = getattr(af, f"right_{joint}", None)
            if val is not None and math.isfinite(val):
                values.append(val)
    return mean(values) if values else None


def _peak_elbow_omega_offset(
    motion_by: dict,
    contact_idx: int,
) -> int | None:
    best_idx: int | None = None
    best_mag = float("-inf")
    for idx, mf in motion_by.items():
        omega = mf.right_elbow_angular_velocity
        if omega is None or not math.isfinite(omega):
            continue
        mag = abs(omega)
        if mag > best_mag:
            best_mag = mag
            best_idx = idx
    if best_idx is None:
        return None
    return best_idx - contact_idx


def _acceleration_fraction(phases: PhaseSequence, contact_idx: int) -> float | None:
    accel_seg = _first_segment(phases, SmashPhase.ACCELERATION)
    prep_seg = _first_segment(phases, SmashPhase.PREPARATION)
    if accel_seg is None:
        return None
    accel_frames = accel_seg.end_frame_index - accel_seg.start_frame_index + 1
    start = prep_seg.start_frame_index if prep_seg else accel_seg.start_frame_index
    total = max(1, contact_idx - start + 1)
    return accel_frames / total


def _follow_through_stats(
    motion_by: dict,
    phases: PhaseSequence,
) -> tuple[float | None, int | None]:
    follow_seg = _first_segment(phases, SmashPhase.FOLLOW_THROUGH)
    if follow_seg is None:
        return None, None

    frame_count = follow_seg.end_frame_index - follow_seg.start_frame_index + 1

    speeds: list[float] = []
    for idx in range(follow_seg.start_frame_index, follow_seg.end_frame_index + 1):
        mf = motion_by.get(idx)
        if mf is None or mf.right_wrist_speed is None:
            continue
        speeds.append(mf.right_wrist_speed)

    if not speeds:
        return 0.0, frame_count

    peak_candidates = [
        mf.right_wrist_speed
        for mf in motion_by.values()
        if mf.right_wrist_speed is not None and mf.right_wrist_speed > 0
    ]
    peak = max(peak_candidates) if peak_candidates else None
    if peak is None:
        return None, frame_count

    ratio = max(speeds) / peak
    return ratio, frame_count


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
