"""V1 rule-based smash phase detection from pose / angles / motion.

Uses relative features (normalized wrist speed, speed trend, elbow angle /
angular velocity) — not fixed absolute velocity thresholds.
ESTIMATED_CONTACT is the peak right-wrist-speed anchor (no shuttle/racket).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from app.schemas.angles import AngleSequence
from app.schemas.motion import MotionSequence
from app.schemas.phases import PhaseSegment, PhaseSequence, SmashPhase
from app.schemas.pose import PoseSequence


@dataclass(slots=True)
class _FrameFeat:
    frame_index: int
    timestamp: float
    wrist_speed: float | None
    elbow_angle: float | None
    elbow_omega: float | None


def detect_smash_phases(
    pose: PoseSequence,
    angles: AngleSequence,
    motion: MotionSequence,
) -> PhaseSequence:
    """Detect PREPARATION → BACKSWING → ACCELERATION → ESTIMATED_CONTACT → FOLLOW_THROUGH."""
    feats = _build_features(pose, angles, motion)
    empty = PhaseSequence(video=pose.video or angles.video or motion.video)
    if not feats:
        return empty

    peak = motion.peaks.get("right_wrist_speed")
    contact_idx_in_feats = _contact_feature_index(feats, peak)
    if contact_idx_in_feats is None:
        # Fallback: argmax among available wrist speeds.
        contact_idx_in_feats = _argmax_wrist(feats)
    if contact_idx_in_feats is None:
        return empty

    peak_speed = feats[contact_idx_in_feats].wrist_speed
    if peak_speed is None or peak_speed <= 0.0 or not math.isfinite(peak_speed):
        return empty

    # Relative wrist speeds (fraction of peak).
    rel = [
        (f.wrist_speed / peak_speed)
        if f.wrist_speed is not None and math.isfinite(f.wrist_speed)
        else None
        for f in feats
    ]
    trend = _speed_trend(rel)
    omega_rel = _relative_abs_omega(feats)

    accel_start = _find_acceleration_start(
        contact_idx_in_feats, rel, trend, omega_rel
    )
    backswing_start = _find_backswing_start(
        accel_start, rel, trend, feats, omega_rel
    )
    follow_end = _find_follow_through_end(contact_idx_in_feats, rel)

    contact_feat = feats[contact_idx_in_feats]
    prep_end = max(0, backswing_start - 1)
    back_end = max(backswing_start, accel_start - 1)
    accel_end = max(accel_start, contact_idx_in_feats - 1)
    follow_start = min(len(feats) - 1, contact_idx_in_feats + 1)

    segments: list[PhaseSegment] = []
    frame_phases: dict[int, SmashPhase] = {}

    def add_range(
        phase: SmashPhase,
        i0: int,
        i1: int,
        confidence: float,
    ) -> None:
        if i0 > i1 or i0 < 0 or i1 >= len(feats):
            return
        seg = PhaseSegment(
            phase=phase,
            start_frame_index=feats[i0].frame_index,
            end_frame_index=feats[i1].frame_index,
            start_timestamp=feats[i0].timestamp,
            end_timestamp=feats[i1].timestamp,
            confidence=float(max(0.0, min(1.0, confidence))),
        )
        segments.append(seg)
        for i in range(i0, i1 + 1):
            frame_phases[feats[i].frame_index] = phase

    overall = _sequence_confidence(feats, contact_idx_in_feats, peak_speed, rel)

    if prep_end >= 0 and backswing_start > 0:
        add_range(SmashPhase.PREPARATION, 0, prep_end, overall * 0.75)
    if backswing_start <= back_end:
        add_range(SmashPhase.BACKSWING, backswing_start, back_end, overall * 0.8)
    if accel_start <= accel_end:
        add_range(SmashPhase.ACCELERATION, accel_start, accel_end, overall * 0.9)

    # Estimated contact is a single-frame segment at the peak-speed anchor.
    add_range(
        SmashPhase.ESTIMATED_CONTACT,
        contact_idx_in_feats,
        contact_idx_in_feats,
        overall,
    )

    if follow_start <= follow_end:
        add_range(
            SmashPhase.FOLLOW_THROUGH,
            follow_start,
            follow_end,
            overall * 0.85,
        )

    # Ensure every feature frame has a label (fill gaps conservatively).
    _fill_unlabeled(feats, frame_phases)

    return PhaseSequence(
        video=pose.video or motion.video,
        segments=segments,
        frame_phases=frame_phases,
        estimated_contact_frame_index=contact_feat.frame_index,
        estimated_contact_timestamp=contact_feat.timestamp,
        confidence=overall,
    )


def _build_features(
    pose: PoseSequence,
    angles: AngleSequence,
    motion: MotionSequence,
) -> list[_FrameFeat]:
    angle_by = {f.frame_index: f for f in angles.frames}
    motion_by = {f.frame_index: f for f in motion.frames}
    ordered = [f.frame_index for f in pose.frames] or [f.frame_index for f in motion.frames]
    pose_by = {f.frame_index: f for f in pose.frames}

    feats: list[_FrameFeat] = []
    for idx in ordered:
        pf = pose_by.get(idx)
        af = angle_by.get(idx)
        mf = motion_by.get(idx)
        timestamp = (
            mf.timestamp
            if mf is not None
            else af.timestamp
            if af is not None
            else pf.timestamp
            if pf is not None
            else 0.0
        )
        feats.append(
            _FrameFeat(
                frame_index=idx,
                timestamp=timestamp,
                wrist_speed=mf.right_wrist_speed if mf is not None else None,
                elbow_angle=af.right_elbow if af is not None else None,
                elbow_omega=mf.right_elbow_angular_velocity if mf is not None else None,
            )
        )
    return feats


def _contact_feature_index(feats: list[_FrameFeat], peak) -> int | None:
    if peak is None or peak.frame_index is None:
        return None
    for i, feat in enumerate(feats):
        if feat.frame_index == peak.frame_index:
            return i
    return None


def _argmax_wrist(feats: list[_FrameFeat]) -> int | None:
    best_i = None
    best_v = float("-inf")
    for i, feat in enumerate(feats):
        if feat.wrist_speed is None or not math.isfinite(feat.wrist_speed):
            continue
        if feat.wrist_speed > best_v:
            best_v = feat.wrist_speed
            best_i = i
    return best_i


def _speed_trend(rel: list[float | None]) -> list[float | None]:
    """Sign of relative-speed change vs previous valid sample (+1 rising)."""
    out: list[float | None] = [None] * len(rel)
    prev_i = None
    for i, value in enumerate(rel):
        if value is None:
            continue
        if prev_i is None:
            out[i] = 0.0
        else:
            prev = rel[prev_i]
            assert prev is not None
            delta = value - prev
            if abs(delta) < 1e-6:
                out[i] = 0.0
            else:
                out[i] = 1.0 if delta > 0 else -1.0
        prev_i = i
    return out


def _relative_abs_omega(feats: list[_FrameFeat]) -> list[float | None]:
    mags = [
        abs(f.elbow_omega)
        if f.elbow_omega is not None and math.isfinite(f.elbow_omega)
        else None
        for f in feats
    ]
    peak = max((m for m in mags if m is not None), default=None)
    if peak is None or peak <= 0:
        return [None] * len(feats)
    return [None if m is None else m / peak for m in mags]


def _find_acceleration_start(
    contact_i: int,
    rel: list[float | None],
    trend: list[float | None],
    omega_rel: list[float | None],
) -> int:
    """Walk back from contact while speed is high / still rising toward peak."""
    i = contact_i
    start = contact_i
    while i > 0:
        prev = i - 1
        r = rel[i]
        rp = rel[prev]
        # Stay in acceleration while relative speed remains substantial
        # or the trend toward contact is still rising.
        rising = trend[i] is not None and trend[i] >= 0
        high = r is not None and r >= 0.35
        prev_high = rp is not None and rp >= 0.25
        omega_hot = omega_rel[i] is not None and omega_rel[i] >= 0.35
        if high or (rising and prev_high) or (rising and omega_hot and r is not None and r >= 0.2):
            start = prev
            i = prev
            continue
        # Local trough: stop when speed drops and trend turns down.
        if rp is not None and r is not None and rp < 0.25 and (trend[i] or 0) <= 0:
            break
        if r is not None and r < 0.2:
            break
        start = prev
        i = prev
    return start


def _find_backswing_start(
    accel_start: int,
    rel: list[float | None],
    trend: list[float | None],
    feats: list[_FrameFeat],
    omega_rel: list[float | None],
) -> int:
    """Before acceleration: moderate motion / elbow cocking window."""
    if accel_start <= 0:
        return 0
    i = accel_start
    start = accel_start
    # Median elbow angle in early window as relative reference.
    early_angles = [
        f.elbow_angle
        for f in feats[: max(1, accel_start)]
        if f.elbow_angle is not None and math.isfinite(f.elbow_angle)
    ]
    elbow_ref = _percentile(early_angles, 0.5) if early_angles else None

    while i > 0:
        prev = i - 1
        r = rel[prev]
        om = omega_rel[prev]
        ang = feats[prev].elbow_angle
        modest_speed = r is not None and 0.08 <= r < 0.45
        cocking = False
        if ang is not None and elbow_ref is not None:
            # More flexed than early median (smaller interior angle).
            cocking = ang < elbow_ref
        omega_cock = om is not None and om >= 0.2
        rising_into_accel = trend[i] is not None and trend[i] >= 0 and (
            r is not None and r >= 0.08
        )
        if modest_speed or cocking or omega_cock or rising_into_accel:
            start = prev
            i = prev
            continue
        if r is not None and r < 0.08 and not cocking:
            break
        # Quiet preparation territory.
        if r is None or r < 0.05:
            break
        start = prev
        i = prev
    return start


def _find_follow_through_end(contact_i: int, rel: list[float | None]) -> int:
    end = contact_i
    for i in range(contact_i + 1, len(rel)):
        r = rel[i]
        end = i
        if r is not None and r < 0.2:
            # Include a couple frames of decay, then stop at first deep drop.
            if i + 1 < len(rel):
                nxt = rel[i + 1]
                if nxt is not None and nxt < 0.15:
                    return i
            return i
    return end


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    idx = int(round(q * (len(ordered) - 1)))
    return ordered[idx]


def _sequence_confidence(
    feats: list[_FrameFeat],
    contact_i: int,
    peak_speed: float,
    rel: list[float | None],
) -> float:
    valid = sum(1 for r in rel if r is not None)
    coverage = valid / max(1, len(rel))
    # Peak prominence: how much higher than median valid speed.
    valid_speeds = [f.wrist_speed for f in feats if f.wrist_speed is not None]
    if len(valid_speeds) < 3:
        return max(0.2, 0.4 * coverage)
    med = _percentile(valid_speeds, 0.5)
    prominence = 0.0 if med <= 0 else min(1.0, (peak_speed - med) / peak_speed)
    # Prefer a clear rise into contact.
    pre = rel[max(0, contact_i - 3) : contact_i]
    rising = 0.0
    if len(pre) >= 2:
        nums = [r for r in pre if r is not None]
        if len(nums) >= 2 and nums[-1] >= nums[0]:
            rising = 0.2
    score = 0.35 * coverage + 0.45 * prominence + rising + 0.1
    return float(max(0.15, min(0.98, score)))


def _fill_unlabeled(feats: list[_FrameFeat], frame_phases: dict[int, SmashPhase]) -> None:
    last: SmashPhase | None = None
    for feat in feats:
        if feat.frame_index in frame_phases:
            last = frame_phases[feat.frame_index]
            continue
        if last is not None:
            frame_phases[feat.frame_index] = last
        else:
            frame_phases[feat.frame_index] = SmashPhase.PREPARATION
