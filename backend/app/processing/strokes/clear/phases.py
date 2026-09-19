"""Forehand clear phase detection (V1).

Reuses the shared kinematic contact anchor (peak wrist speed / forced contact)
but applies clear-specific follow-through → recovery splitting. Acceleration /
backswing windows are derived from the same relative-feature framework as smash
with clear-tuned decay thresholds — not smash technique rules.
"""

from __future__ import annotations

from app.processing.phases import detect_smash_phases
from app.schemas.angles import AngleSequence
from app.schemas.motion import MotionSequence
from app.schemas.phases import PhaseSegment, PhaseSequence, SmashPhase
from app.schemas.pose import PoseSequence

# Clear-specific: after contact, once relative wrist speed stays below this
# fraction of peak for several frames, treat remaining motion as RECOVERY.
_CLEAR_RECOVERY_REL_SPEED = 0.20
_CLEAR_RECOVERY_HOLD_FRAMES = 3


def detect_clear_phases(
    pose: PoseSequence,
    angles: AngleSequence,
    motion: MotionSequence,
    *,
    forced_contact_frame_index: int | None = None,
) -> PhaseSequence:
    """Detect clear phases including RECOVERY after follow-through."""
    base = detect_smash_phases(
        pose,
        angles,
        motion,
        forced_contact_frame_index=forced_contact_frame_index,
    )
    if not base.segments or base.estimated_contact_frame_index is None:
        return base

    peak = motion.peaks.get("right_wrist_speed")
    peak_speed = float(peak.value) if peak is not None and peak.value else None
    if peak_speed is None or peak_speed <= 0:
        return _with_clear_notes(base)

    follow = next(
        (s for s in base.segments if s.phase == SmashPhase.FOLLOW_THROUGH),
        None,
    )
    if follow is None:
        return _with_clear_notes(base)

    motion_by = {f.frame_index: f for f in motion.frames}
    ordered = [
        idx
        for idx in range(follow.start_frame_index, follow.end_frame_index + 1)
        if idx in motion_by
    ]
    if not ordered:
        return _with_clear_notes(base)

    recovery_start: int | None = None
    hold = 0
    for idx in ordered:
        speed = motion_by[idx].right_wrist_speed
        rel = (
            float(speed) / peak_speed
            if speed is not None and peak_speed > 0
            else None
        )
        if rel is not None and rel <= _CLEAR_RECOVERY_REL_SPEED:
            hold += 1
            if hold >= _CLEAR_RECOVERY_HOLD_FRAMES:
                recovery_start = idx - _CLEAR_RECOVERY_HOLD_FRAMES + 1
                break
        else:
            hold = 0

    if recovery_start is None or recovery_start <= follow.start_frame_index:
        return _with_clear_notes(base)

    # Rebuild segments: trim follow-through, append recovery.
    new_segments: list[PhaseSegment] = []
    for seg in base.segments:
        if seg.phase != SmashPhase.FOLLOW_THROUGH:
            new_segments.append(seg)
            continue
        follow_end = max(seg.start_frame_index, recovery_start - 1)
        follow_ts_end = motion_by.get(follow_end)
        new_segments.append(
            PhaseSegment(
                phase=SmashPhase.FOLLOW_THROUGH,
                start_frame_index=seg.start_frame_index,
                end_frame_index=follow_end,
                start_timestamp=seg.start_timestamp,
                end_timestamp=(
                    follow_ts_end.timestamp if follow_ts_end else seg.end_timestamp
                ),
                confidence=seg.confidence,
            )
        )
        rec_end = seg.end_frame_index
        rec_start_frame = motion_by.get(recovery_start)
        rec_end_frame = motion_by.get(rec_end)
        new_segments.append(
            PhaseSegment(
                phase=SmashPhase.RECOVERY,
                start_frame_index=recovery_start,
                end_frame_index=rec_end,
                start_timestamp=(
                    rec_start_frame.timestamp
                    if rec_start_frame
                    else seg.end_timestamp
                ),
                end_timestamp=(
                    rec_end_frame.timestamp if rec_end_frame else seg.end_timestamp
                ),
                confidence=float(max(0.0, min(1.0, seg.confidence * 0.9))),
            )
        )

    frame_phases = dict(base.frame_phases)
    for idx in range(recovery_start, follow.end_frame_index + 1):
        frame_phases[idx] = SmashPhase.RECOVERY

    return PhaseSequence(
        video=base.video,
        segments=new_segments,
        frame_phases=frame_phases,
        estimated_contact_frame_index=base.estimated_contact_frame_index,
        estimated_contact_timestamp=base.estimated_contact_timestamp,
        confidence=base.confidence,
        notes=(
            "Forehand clear phases: ESTIMATED_CONTACT is a kinematic wrist-speed "
            "anchor; RECOVERY begins after clear-specific follow-through decay. "
            "Shuttle trajectory is not inferred."
        ),
    )


def _with_clear_notes(base: PhaseSequence) -> PhaseSequence:
    return PhaseSequence(
        video=base.video,
        segments=list(base.segments),
        frame_phases=dict(base.frame_phases),
        estimated_contact_frame_index=base.estimated_contact_frame_index,
        estimated_contact_timestamp=base.estimated_contact_timestamp,
        confidence=base.confidence,
        notes=(
            "Forehand clear phases (recovery not split — insufficient post-contact "
            "speed decay signal). ESTIMATED_CONTACT is kinematic only."
        ),
    )
