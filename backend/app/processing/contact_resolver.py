"""Resolve smash contact from kinematic estimate + optional shuttle/racket."""

from __future__ import annotations

import math
from dataclasses import dataclass

from app.config import settings
from app.schemas.contact import (
    CONTACT_TYPE_KINEMATIC,
    CONTACT_TYPE_TRACKED,
    ContactEvent,
    ContactSignalEvidence,
)
from app.schemas.motion import MotionSequence
from app.schemas.pose import PoseSequence
from app.schemas.racket import RacketTrajectory
from app.schemas.shuttle import ShuttleTrajectory


@dataclass(slots=True)
class _CandidateScore:
    frame_index: int
    timestamp: float
    total: float
    evidence: list[ContactSignalEvidence]


def resolve_contact(
    *,
    kinematic_frame_index: int,
    kinematic_timestamp: float,
    kinematic_confidence: float = 0.5,
    pose: PoseSequence | None = None,
    motion: MotionSequence | None = None,
    shuttle: ShuttleTrajectory | None = None,
    racket: RacketTrajectory | None = None,
    search_radius: int | None = None,
    min_tracked_confidence: float | None = None,
    hitting_hand: str | None = None,
) -> ContactEvent:
    """Combine peak-wrist kinematic contact with shuttle/racket when reliable.

    Falls back to ``KINEMATIC_ESTIMATE`` when trajectories are missing, sparse,
    ambiguous, or below the tracked-confidence threshold — never raises for
    unavailable shuttle/racket data.
    """
    radius = (
        settings.contact_search_radius if search_radius is None else search_radius
    )
    min_conf = (
        settings.contact_min_tracked_confidence
        if min_tracked_confidence is None
        else min_tracked_confidence
    )
    kinematic = ContactEvent(
        contact_type=CONTACT_TYPE_KINEMATIC,
        frame_index=int(kinematic_frame_index),
        timestamp=float(kinematic_timestamp),
        confidence=float(max(0.0, min(1.0, kinematic_confidence))),
        evidence=[
            ContactSignalEvidence(
                name="peak_wrist_speed",
                value=float(kinematic_frame_index),
                weight=1.0,
                score=float(max(0.0, min(1.0, kinematic_confidence))),
                notes="Kinematic ESTIMATED_CONTACT anchor (peak right-wrist speed).",
            )
        ],
        kinematic_frame_index=int(kinematic_frame_index),
        kinematic_timestamp=float(kinematic_timestamp),
        notes=(
            "Fallback to peak-wrist-speed estimate; shuttle/racket unavailable "
            "or unreliable."
        ),
    )

    shuttle_ok = _trajectory_usable(shuttle, radius, kinematic_frame_index)
    racket_ok = _trajectory_usable(racket, radius, kinematic_frame_index)
    if not shuttle_ok and not racket_ok:
        return kinematic

    hand = hitting_hand
    if hand not in ("LEFT", "RIGHT") and racket is not None:
        hand = racket.hitting_hand if racket.hitting_hand in ("LEFT", "RIGHT") else "RIGHT"
    if hand not in ("LEFT", "RIGHT"):
        hand = "RIGHT"

    shuttle_by = _index_points(shuttle)
    racket_by = _index_points(racket)
    pose_by = {f.frame_index: f for f in pose.frames} if pose is not None else {}
    motion_by = {f.frame_index: f for f in motion.frames} if motion is not None else {}

    frame_min = kinematic_frame_index - radius
    frame_max = kinematic_frame_index + radius
    # Prefer union of known frame indices in the window.
    candidate_frames = sorted(
        {
            *shuttle_by.keys(),
            *racket_by.keys(),
            *pose_by.keys(),
            *motion_by.keys(),
            kinematic_frame_index,
        }
    )
    candidate_frames = [f for f in candidate_frames if frame_min <= f <= frame_max]
    if not candidate_frames:
        return kinematic

    scored: list[_CandidateScore] = []
    for frame_index in candidate_frames:
        evidence: list[ContactSignalEvidence] = []
        weights = 0.0
        weighted = 0.0

        # 1) Minimum shuttle–racket distance (prefer close approach).
        if shuttle_ok and racket_ok:
            dist = _shuttle_racket_distance(shuttle_by.get(frame_index), racket_by.get(frame_index))
            dist_score = _distance_score(
                dist, max_dist=settings.contact_max_shuttle_racket_dist
            )
            w = 0.35
            evidence.append(
                ContactSignalEvidence(
                    name="shuttle_racket_distance",
                    value=dist,
                    weight=w,
                    score=dist_score,
                    notes="Normalized tip-to-shuttle distance (lower is better).",
                )
            )
            weighted += w * dist_score
            weights += w

        # 2) Rapid shuttle trajectory / direction change.
        if shuttle_ok:
            turn = _shuttle_turn_score(shuttle_by, frame_index)
            w = 0.25
            evidence.append(
                ContactSignalEvidence(
                    name="shuttle_direction_change",
                    value=turn,
                    weight=w,
                    score=turn,
                    notes="Local shuttle heading/speed change around the frame.",
                )
            )
            weighted += w * turn
            weights += w

        # 3) Racket proximity to hitting wrist.
        if racket_ok and pose is not None:
            prox = _racket_wrist_proximity(
                racket_by.get(frame_index),
                pose_by.get(frame_index),
                hand=hand,
                max_dist=settings.contact_max_racket_wrist_dist,
            )
            w = 0.20
            evidence.append(
                ContactSignalEvidence(
                    name="racket_wrist_proximity",
                    value=prox,
                    weight=w,
                    score=prox,
                    notes="Racket point near hitting-hand wrist.",
                )
            )
            weighted += w * prox
            weights += w

        # 4) Local wrist / racket motion (favor energetic contact window).
        motion_score = _local_motion_score(
            motion_by.get(frame_index),
            racket_by,
            frame_index,
            peak_frame=kinematic_frame_index,
        )
        w = 0.20
        evidence.append(
            ContactSignalEvidence(
                name="local_wrist_racket_motion",
                value=motion_score,
                weight=w,
                score=motion_score,
                notes="Wrist speed near kinematic peak plus racket tip motion.",
            )
        )
        weighted += w * motion_score
        weights += w

        if weights <= 1e-9:
            continue
        total = weighted / weights
        # Soft prior toward the kinematic frame (keeps tracked contact nearby).
        prior = math.exp(
            -0.5 * ((frame_index - kinematic_frame_index) / max(1.0, radius / 2.0)) ** 2
        )
        total = 0.85 * total + 0.15 * prior
        ts = _timestamp_for(
            frame_index,
            kinematic_timestamp,
            kinematic_frame_index,
            shuttle_by,
            racket_by,
            pose_by,
            motion_by,
        )
        scored.append(
            _CandidateScore(
                frame_index=frame_index,
                timestamp=ts,
                total=float(total),
                evidence=evidence,
            )
        )

    if not scored:
        return kinematic

    scored.sort(key=lambda c: c.total, reverse=True)
    best = scored[0]
    second = scored[1] if len(scored) > 1 else None

    # Ambiguous: two strong peaks nearly tied and separated in time.
    if (
        second is not None
        and abs(best.total - second.total) < settings.contact_ambiguity_margin
        and abs(best.frame_index - second.frame_index) > 1
        and best.total < min_conf + 0.15
    ):
        kinematic.notes = (
            "Ambiguous tracked contact (competing peaks); "
            "falling back to peak-wrist-speed estimate."
        )
        kinematic.evidence.append(
            ContactSignalEvidence(
                name="ambiguity",
                value=abs(best.total - second.total),
                weight=0.0,
                score=0.0,
                notes=(
                    f"Best frame {best.frame_index} score={best.total:.3f}; "
                    f"runner-up {second.frame_index} score={second.total:.3f}."
                ),
            )
        )
        return kinematic

    if best.total < min_conf:
        kinematic.notes = (
            "Tracked signals below confidence threshold; "
            "falling back to peak-wrist-speed estimate."
        )
        kinematic.evidence.extend(best.evidence)
        return kinematic

    # Require at least one strong geometric cue when both modalities exist.
    if shuttle_ok and racket_ok:
        dist_ev = next(
            (e for e in best.evidence if e.name == "shuttle_racket_distance"),
            None,
        )
        if dist_ev is None or (dist_ev.score is not None and dist_ev.score < 0.35):
            kinematic.notes = (
                "Shuttle–racket proximity weak at candidate; "
                "falling back to peak-wrist-speed estimate."
            )
            kinematic.evidence.extend(best.evidence)
            return kinematic

    conf = float(max(0.0, min(1.0, 0.55 * best.total + 0.45 * kinematic.confidence)))
    return ContactEvent(
        contact_type=CONTACT_TYPE_TRACKED,
        frame_index=best.frame_index,
        timestamp=best.timestamp,
        confidence=conf,
        evidence=best.evidence,
        kinematic_frame_index=int(kinematic_frame_index),
        kinematic_timestamp=float(kinematic_timestamp),
        notes=(
            "TRACKED_CONTACT from shuttle/racket evidence near the "
            "kinematic peak-wrist-speed estimate."
        ),
    )


def _trajectory_usable(traj, radius: int, center: int) -> bool:
    if traj is None or not getattr(traj, "frames", None):
        return False
    visible = 0
    for point in traj.frames:
        if abs(point.frame_index - center) > radius:
            continue
        if point.visible and point.x is not None and point.y is not None:
            visible += 1
    return visible >= settings.contact_min_visible_frames


def _index_points(traj) -> dict[int, object]:
    if traj is None:
        return {}
    return {
        p.frame_index: p
        for p in traj.frames
        if p.visible and p.x is not None and p.y is not None
    }


def _shuttle_racket_distance(shuttle, racket) -> float | None:
    if shuttle is None or racket is None:
        return None
    if shuttle.x is None or shuttle.y is None or racket.x is None or racket.y is None:
        return None
    return float(math.hypot(float(shuttle.x) - float(racket.x), float(shuttle.y) - float(racket.y)))


def _distance_score(dist: float | None, *, max_dist: float) -> float:
    if dist is None or not math.isfinite(dist):
        return 0.0
    if max_dist <= 0:
        return 0.0
    return float(max(0.0, min(1.0, 1.0 - dist / max_dist)))


def _shuttle_turn_score(shuttle_by: dict[int, object], frame_index: int) -> float:
    prev = shuttle_by.get(frame_index - 1)
    cur = shuttle_by.get(frame_index)
    nxt = shuttle_by.get(frame_index + 1)
    if prev is None or cur is None or nxt is None:
        # Try ±2 if immediate neighbors missing.
        prev = prev or shuttle_by.get(frame_index - 2)
        nxt = nxt or shuttle_by.get(frame_index + 2)
    if prev is None or cur is None or nxt is None:
        return 0.0
    v1x = float(cur.x) - float(prev.x)
    v1y = float(cur.y) - float(prev.y)
    v2x = float(nxt.x) - float(cur.x)
    v2y = float(nxt.y) - float(cur.y)
    n1 = math.hypot(v1x, v1y)
    n2 = math.hypot(v2x, v2y)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    # Direction change: 1 - cosine similarity, plus speed-ratio change.
    cos = max(-1.0, min(1.0, (v1x * v2x + v1y * v2y) / (n1 * n2)))
    turn = 0.5 * (1.0 - cos)
    speed_change = abs(n2 - n1) / max(n1, n2)
    return float(max(0.0, min(1.0, 0.65 * turn + 0.35 * speed_change)))


def _racket_wrist_proximity(
    racket,
    pose_frame,
    *,
    hand: str,
    max_dist: float,
) -> float:
    if racket is None or pose_frame is None:
        return 0.0
    wrist_name = "right_wrist" if hand == "RIGHT" else "left_wrist"
    wrist = pose_frame.keypoints.get(wrist_name)
    if wrist is None or racket.x is None or racket.y is None:
        return 0.0
    dist = math.hypot(float(racket.x) - float(wrist.x), float(racket.y) - float(wrist.y))
    return _distance_score(dist, max_dist=max_dist)


def _local_motion_score(
    motion_frame,
    racket_by: dict[int, object],
    frame_index: int,
    *,
    peak_frame: int,
) -> float:
    wrist_term = 0.0
    if motion_frame is not None and motion_frame.right_wrist_speed is not None:
        # Relative closeness to the kinematic peak frame as a proxy when we
        # lack a global peak value here; still rewards energetic frames nearby.
        offset = abs(frame_index - peak_frame)
        wrist_term = math.exp(-0.35 * offset)
        # Boost if absolute speed looks non-trivial.
        speed = float(motion_frame.right_wrist_speed)
        if math.isfinite(speed) and speed > 0:
            wrist_term = min(1.0, 0.5 * wrist_term + 0.5 * min(1.0, speed * 2.0))

    racket_term = 0.0
    cur = racket_by.get(frame_index)
    prev = racket_by.get(frame_index - 1)
    if cur is not None and prev is not None and cur.x is not None and prev.x is not None:
        step = math.hypot(float(cur.x) - float(prev.x), float(cur.y) - float(prev.y))
        racket_term = min(1.0, step / 0.05)

    if wrist_term <= 0 and racket_term <= 0:
        return 0.0
    return float(max(0.0, min(1.0, 0.7 * wrist_term + 0.3 * racket_term)))


def _timestamp_for(
    frame_index: int,
    kinematic_timestamp: float,
    kinematic_frame_index: int,
    shuttle_by,
    racket_by,
    pose_by,
    motion_by,
) -> float:
    for source in (shuttle_by, racket_by, pose_by, motion_by):
        point = source.get(frame_index)
        if point is not None and getattr(point, "timestamp", None) is not None:
            return float(point.timestamp)
    # Assume constant fps from kinematic pair.
    if kinematic_frame_index == frame_index:
        return float(kinematic_timestamp)
    # Unknown fps — linear guess at 30fps offset.
    return float(kinematic_timestamp + (frame_index - kinematic_frame_index) / 30.0)
