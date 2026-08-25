"""Build StrokeMetrics from existing pose / angle / motion / phase sequences."""

from __future__ import annotations

import math
from statistics import mean

from app.schemas.angles import AngleSequence
from app.schemas.motion import MotionSequence
from app.schemas.phases import PhaseSequence, SmashPhase
from app.schemas.pose import PoseSequence
from app.schemas.stroke_metrics import StrokeMetrics


def compute_stroke_metrics(
    pose: PoseSequence,
    angles: AngleSequence,
    motion: MotionSequence,
    phases: PhaseSequence,
) -> StrokeMetrics:
    """Aggregate phase-aware metrics; does not evaluate technique."""
    video = pose.video or angles.video or motion.video or phases.video
    metrics = StrokeMetrics(
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

    metrics.peak_elbow_omega_offset_frames = _peak_elbow_omega_offset(
        motion_by, contact_idx
    )
    metrics.acceleration_phase_fraction = _acceleration_fraction(phases, contact_idx)

    follow_ratio, follow_frames = _follow_through_stats(motion_by, phases, contact_idx)
    metrics.follow_through_speed_ratio = follow_ratio
    metrics.follow_through_frame_count = follow_frames

    return metrics


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
    accel_seg = next(
        (s for s in phases.segments if s.phase is SmashPhase.ACCELERATION),
        None,
    )
    prep_seg = next(
        (s for s in phases.segments if s.phase is SmashPhase.PREPARATION),
        None,
    )
    if accel_seg is None:
        return None
    accel_frames = accel_seg.end_frame_index - accel_seg.start_frame_index + 1
    start = prep_seg.start_frame_index if prep_seg else accel_seg.start_frame_index
    total = max(1, contact_idx - start + 1)
    return accel_frames / total


def _follow_through_stats(
    motion_by: dict,
    phases: PhaseSequence,
    contact_idx: int,
) -> tuple[float | None, int | None]:
    follow_seg = next(
        (s for s in phases.segments if s.phase is SmashPhase.FOLLOW_THROUGH),
        None,
    )
    if follow_seg is None:
        return None, None

    peak_speed = motion_by.get(contact_idx)
    peak = (
        peak_speed.right_wrist_speed
        if peak_speed is not None
        else None
    )
    if peak is None or peak <= 0:
        peak_entry = motion_by.get(contact_idx)
        peak = peak_entry.right_wrist_speed if peak_entry else None
    if peak is None or peak <= 0:
        return None, follow_seg.end_frame_index - follow_seg.start_frame_index + 1

    speeds: list[float] = []
    for idx in range(follow_seg.start_frame_index, follow_seg.end_frame_index + 1):
        mf = motion_by.get(idx)
        if mf is None or mf.right_wrist_speed is None:
            continue
        speeds.append(mf.right_wrist_speed)

    if not speeds:
        frame_count = follow_seg.end_frame_index - follow_seg.start_frame_index + 1
        return 0.0, frame_count

    ratio = max(speeds) / peak
    frame_count = follow_seg.end_frame_index - follow_seg.start_frame_index + 1
    return ratio, frame_count
