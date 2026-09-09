"""Tests for ContactResolver — tracked, ambiguous, and kinematic fallback."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.processing.contact_resolver import resolve_contact
from app.processing.phases import detect_smash_phases
from app.processing.stroke_metrics import compute_stroke_metrics, extract_stroke_metrics
from app.schemas.contact import CONTACT_TYPE_KINEMATIC, CONTACT_TYPE_TRACKED
from app.schemas.motion import MotionFrame, MotionSequence, PeakStats
from app.schemas.pose import Keypoint, PoseFrame, PoseSequence
from app.schemas.racket import RacketPoint, RacketTrajectory
from app.schemas.shuttle import ShuttlePoint, ShuttleTrajectory
from tests.test_phases import _synthetic_smash


def _shuttle_traj(
    *,
    contact: int,
    n: int,
    tip_at_contact: tuple[float, float] = (0.62, 0.40),
    turn: bool = True,
) -> ShuttleTrajectory:
    frames: list[ShuttlePoint] = []
    for i in range(n):
        # Approach from upper-left, then reverse after contact when turn=True.
        if i < contact:
            x = tip_at_contact[0] - 0.04 * (contact - i)
            y = tip_at_contact[1] - 0.03 * (contact - i)
        elif turn:
            x = tip_at_contact[0] + 0.05 * (i - contact)
            y = tip_at_contact[1] + 0.04 * (i - contact)
        else:
            x = tip_at_contact[0] + 0.01 * (i - contact)
            y = tip_at_contact[1] - 0.01 * (i - contact)
        frames.append(
            ShuttlePoint(
                frame_index=i,
                timestamp=i * 0.05,
                x=x,
                y=y,
                visible=True,
                confidence=0.95,
            )
        )
    return ShuttleTrajectory(
        video="smash.mp4",
        backend="test",
        fps=20.0,
        width=1280,
        height=720,
        frames=frames,
        missing_frame_indices=[],
    )


def _racket_traj(
    *,
    contact: int,
    n: int,
    tip_at_contact: tuple[float, float] = (0.62, 0.40),
) -> RacketTrajectory:
    frames: list[RacketPoint] = []
    for i in range(n):
        # Racket tip near shuttle at contact; wrist-ish slightly behind.
        offset = 0.01 * abs(i - contact)
        frames.append(
            RacketPoint(
                frame_index=i,
                timestamp=i * 0.05,
                x=tip_at_contact[0] - offset * 0.2,
                y=tip_at_contact[1] + offset * 0.1,
                visible=True,
                confidence=0.9,
                hand="RIGHT",
            )
        )
    return RacketTrajectory(
        video="smash.mp4",
        backend="test",
        fps=20.0,
        width=1280,
        height=720,
        hitting_hand="RIGHT",
        frames=frames,
        missing_frame_indices=[],
    )


def _pose_near_racket(n: int, contact: int, tip=(0.62, 0.40)) -> PoseSequence:
    frames = []
    for i in range(n):
        frames.append(
            PoseFrame(
                frame_index=i,
                timestamp=i * 0.05,
                keypoints={
                    "right_wrist": Keypoint(tip[0] - 0.05, tip[1] + 0.02, 0.95),
                    "right_elbow": Keypoint(tip[0] - 0.12, tip[1] + 0.04, 0.9),
                },
            )
        )
    return PoseSequence(video="smash.mp4", frames=frames)


def test_resolve_tracked_contact_from_shuttle_racket() -> None:
    pose, angles, motion, kinematic = _synthetic_smash(n=40, contact=28)
    tip = (0.60, 0.42)
    # Put clear min-distance + turn at frame 30 (near kinematic).
    tracked_frame = 30
    shuttle = _shuttle_traj(contact=tracked_frame, n=40, tip_at_contact=tip, turn=True)
    racket = _racket_traj(contact=tracked_frame, n=40, tip_at_contact=tip)
    pose_ctx = _pose_near_racket(40, tracked_frame, tip)

    event = resolve_contact(
        kinematic_frame_index=kinematic,
        kinematic_timestamp=kinematic * 0.05,
        kinematic_confidence=0.8,
        pose=pose_ctx,
        motion=motion,
        shuttle=shuttle,
        racket=racket,
        search_radius=8,
        min_tracked_confidence=0.45,
    )
    assert event.contact_type == CONTACT_TYPE_TRACKED
    assert event.frame_index == tracked_frame
    assert event.kinematic_frame_index == kinematic
    assert event.confidence > 0.45
    names = {e.name for e in event.evidence}
    assert "shuttle_racket_distance" in names
    assert "shuttle_direction_change" in names


def test_resolve_fallback_when_trajectories_missing() -> None:
    event = resolve_contact(
        kinematic_frame_index=22,
        kinematic_timestamp=1.1,
        kinematic_confidence=0.7,
        pose=None,
        motion=None,
        shuttle=None,
        racket=None,
    )
    assert event.contact_type == CONTACT_TYPE_KINEMATIC
    assert event.frame_index == 22
    assert event.timestamp == pytest.approx(1.1)


def test_resolve_ambiguous_competing_peaks_falls_back() -> None:
    n = 40
    kinematic = 20
    # Two equally good close approaches far apart inside the window.
    frames_s: list[ShuttlePoint] = []
    frames_r: list[RacketPoint] = []
    for i in range(n):
        # Peaks at 16 and 24 with similar geometry, weak turn everywhere.
        if abs(i - 16) <= 1:
            sx, sy = 0.50, 0.50
        elif abs(i - 24) <= 1:
            sx, sy = 0.70, 0.50
        else:
            sx, sy = 0.2, 0.2
        frames_s.append(
            ShuttlePoint(i, i * 0.05, sx, sy, True, 0.9)
        )
        frames_r.append(
            RacketPoint(i, i * 0.05, sx, sy, None, True, 0.9, "RIGHT")
        )
    shuttle = ShuttleTrajectory("v.mp4", "t", 20.0, 100, 100, frames_s, [])
    racket = RacketTrajectory(
        "v.mp4", "t", 20.0, 100, 100, "RIGHT", frames_r, []
    )
    pose = _pose_near_racket(n, kinematic, (0.6, 0.5))
    motion = MotionSequence(
        video="v.mp4",
        frames=[
            MotionFrame(i, i * 0.05, right_wrist_speed=0.5)
            for i in range(n)
        ],
        peaks={"right_wrist_speed": PeakStats(1.0, kinematic, kinematic * 0.05)},
    )
    event = resolve_contact(
        kinematic_frame_index=kinematic,
        kinematic_timestamp=kinematic * 0.05,
        kinematic_confidence=0.6,
        pose=pose,
        motion=motion,
        shuttle=shuttle,
        racket=racket,
        search_radius=8,
        min_tracked_confidence=0.55,
    )
    assert event.contact_type == CONTACT_TYPE_KINEMATIC
    assert event.frame_index == kinematic
    assert any(e.name == "ambiguity" for e in event.evidence) or "Ambiguous" in event.notes or "threshold" in event.notes.lower() or "Fallback" in event.notes or "falling back" in event.notes


def test_metrics_snap_to_resolved_contact() -> None:
    pose, angles, motion, _ = _synthetic_smash(n=40, contact=28)
    phases = detect_smash_phases(pose, angles, motion)
    assert phases.estimated_contact_frame_index == 28

    tracked = resolve_contact(
        kinematic_frame_index=28,
        kinematic_timestamp=28 * 0.05,
        kinematic_confidence=0.85,
        pose=_pose_near_racket(40, 31, (0.61, 0.41)),
        motion=motion,
        shuttle=_shuttle_traj(contact=31, n=40, tip_at_contact=(0.61, 0.41)),
        racket=_racket_traj(contact=31, n=40, tip_at_contact=(0.61, 0.41)),
        min_tracked_confidence=0.4,
    )
    assert tracked.contact_type == CONTACT_TYPE_TRACKED
    assert tracked.frame_index == 31

    snapped_phases = detect_smash_phases(
        pose, angles, motion, forced_contact_frame_index=tracked.frame_index
    )
    assert snapped_phases.estimated_contact_frame_index == 31

    metrics = compute_stroke_metrics(
        pose, angles, motion, snapped_phases, contact=tracked
    )
    assert metrics.estimated_contact_frame_index == 31
    # Contact elbow comes from frame 31, not kinematic 28.
    angle_31 = next(f for f in angles.frames if f.frame_index == 31)
    assert metrics.contact_elbow_angle_deg == pytest.approx(angle_31.right_elbow)

    phase_metrics = extract_stroke_metrics(
        snapped_phases, angles, motion, contact=tracked
    )
    assert phase_metrics.estimated_contact.frame_index == 31


def test_sparse_shuttle_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.config import settings

    monkeypatch.setattr(settings, "contact_min_visible_frames", 5)
    shuttle = ShuttleTrajectory(
        video="v.mp4",
        backend="t",
        fps=30.0,
        width=100,
        height=100,
        frames=[
            ShuttlePoint(20, 1.0, 0.5, 0.5, True, 1.0),
            ShuttlePoint(21, 1.05, None, None, False, 0.0),
        ],
        missing_frame_indices=[21],
    )
    event = resolve_contact(
        kinematic_frame_index=20,
        kinematic_timestamp=1.0,
        kinematic_confidence=0.9,
        shuttle=shuttle,
        racket=None,
    )
    assert event.contact_type == CONTACT_TYPE_KINEMATIC
    assert event.frame_index == 20
