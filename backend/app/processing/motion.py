"""Motion derivatives from smoothed pose + joint angles.

Independent of MMPose, temporal smoothing, and angle geometry.
Uses timestamps for Δt (no fixed FPS assumption). Does not interpolate gaps.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

from app.schemas.angles import AngleFrame, AngleSequence
from app.schemas.motion import MotionFrame, MotionSequence, PeakStats
from app.schemas.pose import Keypoint, PoseFrame, PoseSequence

_METRIC_WRIST_SPEED = "right_wrist_speed"
_METRIC_ELBOW_OMEGA = "right_elbow_angular_velocity"
_METRIC_KNEE_OMEGA = "right_knee_angular_velocity"


def compute_motion_derivatives(
    pose: PoseSequence,
    angles: AngleSequence,
    *,
    confidence_threshold: float = 0.5,
) -> MotionSequence:
    """Per-frame wrist speed and elbow/knee angular velocity, plus peaks.

    Linear speed uses consecutive valid ``right_wrist`` samples from ``pose``.
    Angular velocities use consecutive non-null angle samples from ``angles``.
    Missing values yield ``None`` (no interpolation).
    """
    pose_by_index = {frame.frame_index: frame for frame in pose.frames}
    angle_by_index = {frame.frame_index: frame for frame in angles.frames}
    # Prefer pose frame order; fall back to angles if pose empty.
    ordered_indices = [f.frame_index for f in pose.frames] or [
        f.frame_index for f in angles.frames
    ]

    out = MotionSequence(video=pose.video or angles.video)
    prev_pose: PoseFrame | None = None
    prev_angle: AngleFrame | None = None

    for frame_index in ordered_indices:
        pose_frame = pose_by_index.get(frame_index)
        angle_frame = angle_by_index.get(frame_index)
        timestamp = (
            pose_frame.timestamp
            if pose_frame is not None
            else angle_frame.timestamp
            if angle_frame is not None
            else 0.0
        )

        wrist_speed = None
        if pose_frame is not None and prev_pose is not None:
            wrist_speed = _linear_speed(
                prev_pose,
                pose_frame,
                joint="right_wrist",
                confidence_threshold=confidence_threshold,
            )

        elbow_omega = None
        knee_omega = None
        if angle_frame is not None and prev_angle is not None:
            elbow_omega = _angular_velocity(
                prev_angle.right_elbow,
                angle_frame.right_elbow,
                prev_angle.timestamp,
                angle_frame.timestamp,
            )
            knee_omega = _angular_velocity(
                prev_angle.right_knee,
                angle_frame.right_knee,
                prev_angle.timestamp,
                angle_frame.timestamp,
            )

        out.append(
            MotionFrame(
                frame_index=frame_index,
                timestamp=timestamp,
                right_wrist_speed=wrist_speed,
                right_elbow_angular_velocity=elbow_omega,
                right_knee_angular_velocity=knee_omega,
            )
        )

        if pose_frame is not None:
            prev_pose = pose_frame
        if angle_frame is not None:
            prev_angle = angle_frame

    out.peaks = {
        _METRIC_WRIST_SPEED: _peak_for(
            out.frames, lambda f: f.right_wrist_speed, use_abs=False
        ),
        _METRIC_ELBOW_OMEGA: _peak_for(
            out.frames, lambda f: f.right_elbow_angular_velocity, use_abs=True
        ),
        _METRIC_KNEE_OMEGA: _peak_for(
            out.frames, lambda f: f.right_knee_angular_velocity, use_abs=True
        ),
    }
    return out


def _usable_keypoint(
    frame: PoseFrame,
    joint: str,
    *,
    confidence_threshold: float,
) -> Keypoint | None:
    kp = frame.keypoints.get(joint)
    if kp is None or kp.confidence < confidence_threshold:
        return None
    return kp


def _delta_t(t0: float, t1: float) -> float | None:
    dt = t1 - t0
    if dt <= 0.0 or not math.isfinite(dt):
        return None
    return dt


def _linear_speed(
    prev: PoseFrame,
    curr: PoseFrame,
    *,
    joint: str,
    confidence_threshold: float,
) -> float | None:
    a = _usable_keypoint(prev, joint, confidence_threshold=confidence_threshold)
    b = _usable_keypoint(curr, joint, confidence_threshold=confidence_threshold)
    if a is None or b is None:
        return None
    dt = _delta_t(prev.timestamp, curr.timestamp)
    if dt is None:
        return None
    dist = math.hypot(b.x - a.x, b.y - a.y)
    return dist / dt


def _angular_velocity(
    prev_angle: float | None,
    curr_angle: float | None,
    t0: float,
    t1: float,
) -> float | None:
    if prev_angle is None or curr_angle is None:
        return None
    dt = _delta_t(t0, t1)
    if dt is None:
        return None
    return (curr_angle - prev_angle) / dt


def _peak_for(
    frames: Sequence[MotionFrame],
    getter: Callable[[MotionFrame], float | None],
    *,
    use_abs: bool,
) -> PeakStats:
    best_frame: MotionFrame | None = None
    best_score = float("-inf")
    best_value: float | None = None

    for frame in frames:
        value = getter(frame)
        if value is None or not math.isfinite(value):
            continue
        score = abs(value) if use_abs else value
        if score > best_score:
            best_score = score
            best_frame = frame
            best_value = value

    if best_frame is None:
        return PeakStats(value=None, frame_index=None, timestamp=None)

    # For angular metrics, peak value is the max |ω| (magnitude of peak motion).
    peak_value = abs(best_value) if use_abs and best_value is not None else best_value
    return PeakStats(
        value=peak_value,
        frame_index=best_frame.frame_index,
        timestamp=best_frame.timestamp,
    )
