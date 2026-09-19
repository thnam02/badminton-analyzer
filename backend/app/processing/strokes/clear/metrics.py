"""Forehand clear metrics extraction from FinalAnalysisState."""

from __future__ import annotations

import math

from app.schemas.angles import AngleSequence
from app.schemas.clear_metrics import ForehandClearMetrics
from app.schemas.contact import ContactEvent
from app.schemas.final_analysis import FinalAnalysisState
from app.schemas.motion import MotionSequence
from app.schemas.phases import PhaseSequence, SmashPhase
from app.schemas.pose import PoseSequence


def compute_clear_metrics_from_final(state: FinalAnalysisState) -> ForehandClearMetrics:
    return compute_clear_metrics(
        state.smoothed_pose,
        state.angles,
        state.motion,
        state.phases,
        contact=state.contact,
    )


def compute_clear_metrics(
    pose: PoseSequence,
    angles: AngleSequence,
    motion: MotionSequence,
    phases: PhaseSequence,
    *,
    contact: ContactEvent | None = None,
) -> ForehandClearMetrics:
    video = pose.video or angles.video or motion.video or phases.video
    contact_idx = (
        contact.frame_index
        if contact is not None
        else phases.estimated_contact_frame_index
    )
    contact_ts = (
        contact.timestamp
        if contact is not None
        else phases.estimated_contact_timestamp
    )
    metrics = ForehandClearMetrics(
        video=video,
        estimated_contact_frame_index=contact_idx,
        estimated_contact_timestamp=contact_ts,
        phase_confidence=float(phases.confidence),
        contact_confidence=float(contact.confidence) if contact is not None else 0.0,
    )
    if contact_idx is None:
        return metrics

    angle_by = {f.frame_index: f for f in angles.frames}
    motion_by = {f.frame_index: f for f in motion.frames}
    pose_by = {f.frame_index: f for f in pose.frames}

    contact_angle = angle_by.get(contact_idx)
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
        metrics.peak_wrist_speed = float(peak.value)

    elbow_peak = motion.peaks.get("right_elbow_angular_velocity")
    if elbow_peak is not None and elbow_peak.value is not None:
        metrics.peak_elbow_omega = abs(float(elbow_peak.value))
        if elbow_peak.frame_index is not None:
            metrics.peak_elbow_omega_offset_frames = (
                int(elbow_peak.frame_index) - int(contact_idx)
            )
    else:
        metrics.peak_elbow_omega_offset_frames = _peak_elbow_omega_offset(
            motion_by, contact_idx
        )

    metrics.preparation_elbow_angle_deg = _phase_mean_angle(
        phases, angle_by, SmashPhase.PREPARATION, "elbow"
    )
    metrics.preparation_knee_angle_deg = _phase_mean_angle(
        phases, angle_by, SmashPhase.PREPARATION, "knee"
    )
    metrics.backswing_min_elbow_angle_deg = _phase_extreme_angle(
        phases, angle_by, SmashPhase.BACKSWING, "elbow", mode="min"
    )
    metrics.backswing_wrist_y_travel = _phase_wrist_y_travel(
        phases, pose_by, SmashPhase.BACKSWING
    )
    metrics.acceleration_phase_fraction = _acceleration_fraction(phases, contact_idx)

    follow_ratio, follow_frames = _follow_through_stats(motion_by, phases)
    metrics.follow_through_speed_ratio = follow_ratio
    metrics.follow_through_frame_count = follow_frames
    metrics.follow_through_wrist_y_travel = _phase_wrist_y_travel(
        phases, pose_by, SmashPhase.FOLLOW_THROUGH
    )

    rec_seg = _first_segment(phases, SmashPhase.RECOVERY)
    if rec_seg is not None:
        metrics.recovery_frame_count = (
            int(rec_seg.end_frame_index) - int(rec_seg.start_frame_index) + 1
        )
        metrics.recovery_duration_s = float(
            max(0.0, rec_seg.end_timestamp - rec_seg.start_timestamp)
        )

    return metrics


def _first_segment(phases: PhaseSequence, phase: SmashPhase):
    for seg in phases.segments:
        if seg.phase == phase:
            return seg
    return None


def _phase_mean_angle(
    phases: PhaseSequence,
    angle_by: dict,
    phase: SmashPhase,
    joint: str,
) -> float | None:
    seg = _first_segment(phases, phase)
    if seg is None:
        return None
    vals: list[float] = []
    for idx in range(seg.start_frame_index, seg.end_frame_index + 1):
        frame = angle_by.get(idx)
        if frame is None:
            continue
        val = getattr(frame, f"right_{joint}", None)
        if val is not None and math.isfinite(float(val)):
            vals.append(float(val))
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _phase_extreme_angle(
    phases: PhaseSequence,
    angle_by: dict,
    phase: SmashPhase,
    joint: str,
    *,
    mode: str,
) -> float | None:
    seg = _first_segment(phases, phase)
    if seg is None:
        return None
    vals: list[float] = []
    for idx in range(seg.start_frame_index, seg.end_frame_index + 1):
        frame = angle_by.get(idx)
        if frame is None:
            continue
        val = getattr(frame, f"right_{joint}", None)
        if val is not None and math.isfinite(float(val)):
            vals.append(float(val))
    if not vals:
        return None
    return float(min(vals) if mode == "min" else max(vals))


def _phase_wrist_y_travel(
    phases: PhaseSequence, pose_by: dict, phase: SmashPhase
) -> float | None:
    seg = _first_segment(phases, phase)
    if seg is None:
        return None
    ys: list[float] = []
    for idx in range(seg.start_frame_index, seg.end_frame_index + 1):
        frame = pose_by.get(idx)
        if frame is None:
            continue
        wrist = frame.keypoints.get("right_wrist")
        if wrist is not None and math.isfinite(float(wrist.y)):
            ys.append(float(wrist.y))
    if len(ys) < 2:
        return None
    return float(max(ys) - min(ys))


def _peak_elbow_omega_offset(motion_by: dict, contact_idx: int) -> int | None:
    best_idx = None
    best_abs = -1.0
    for idx, frame in motion_by.items():
        omega = frame.right_elbow_angular_velocity
        if omega is None or not math.isfinite(float(omega)):
            continue
        a = abs(float(omega))
        if a > best_abs:
            best_abs = a
            best_idx = int(idx)
    if best_idx is None:
        return None
    return int(best_idx - contact_idx)


def _acceleration_fraction(phases: PhaseSequence, contact_idx: int) -> float | None:
    accel = _first_segment(phases, SmashPhase.ACCELERATION)
    prep = _first_segment(phases, SmashPhase.PREPARATION)
    if accel is None:
        return None
    accel_len = max(1, accel.end_frame_index - accel.start_frame_index + 1)
    if prep is None:
        return None
    total = max(1, contact_idx - prep.start_frame_index + 1)
    return float(accel_len / total)


def _follow_through_stats(
    motion_by: dict, phases: PhaseSequence
) -> tuple[float | None, int | None]:
    follow = _first_segment(phases, SmashPhase.FOLLOW_THROUGH)
    contact_idx = phases.estimated_contact_frame_index
    if follow is None or contact_idx is None:
        return None, None
    contact_m = motion_by.get(contact_idx)
    contact_speed = (
        float(contact_m.right_wrist_speed)
        if contact_m is not None and contact_m.right_wrist_speed is not None
        else None
    )
    speeds: list[float] = []
    for idx in range(follow.start_frame_index, follow.end_frame_index + 1):
        frame = motion_by.get(idx)
        if frame is None or frame.right_wrist_speed is None:
            continue
        speeds.append(float(frame.right_wrist_speed))
    frames = follow.end_frame_index - follow.start_frame_index + 1
    if not speeds or contact_speed is None or contact_speed <= 1e-9:
        return None, int(frames)
    return float((sum(speeds) / len(speeds)) / contact_speed), int(frames)
