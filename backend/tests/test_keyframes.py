"""Tests for smash keyframe selection (index alignment; no CV inference)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from app.processing.keyframes import extract_keyframes, select_keyframe_indices
from app.processing.phases import detect_smash_phases
from app.schemas.keyframes import CONTACT_MINUS_2, CONTACT_PLUS_2
from app.schemas.phases import PhaseSegment, PhaseSequence, SmashPhase
from app.schemas.pose import PoseFrame, PoseSequence
from tests.test_phases import _synthetic_smash


def _pose_only(n: int = 40, dt: float = 0.05) -> PoseSequence:
    frames = [
        PoseFrame(frame_index=i, timestamp=i * dt, keypoints={})
        for i in range(n)
    ]
    return PoseSequence(video="clip.mp4", frames=frames)


def _manual_phases(contact: int = 28) -> PhaseSequence:
    """Deterministic phase windows for selection tests (no motion heuristics)."""
    segments = [
        PhaseSegment(
            phase=SmashPhase.PREPARATION,
            start_frame_index=0,
            end_frame_index=7,
            start_timestamp=0.0,
            end_timestamp=0.35,
            confidence=0.7,
        ),
        PhaseSegment(
            phase=SmashPhase.BACKSWING,
            start_frame_index=8,
            end_frame_index=17,
            start_timestamp=0.4,
            end_timestamp=0.85,
            confidence=0.8,
        ),
        PhaseSegment(
            phase=SmashPhase.ACCELERATION,
            start_frame_index=18,
            end_frame_index=27,
            start_timestamp=0.9,
            end_timestamp=1.35,
            confidence=0.9,
        ),
        PhaseSegment(
            phase=SmashPhase.ESTIMATED_CONTACT,
            start_frame_index=contact,
            end_frame_index=contact,
            start_timestamp=contact * 0.05,
            end_timestamp=contact * 0.05,
            confidence=0.95,
        ),
        PhaseSegment(
            phase=SmashPhase.FOLLOW_THROUGH,
            start_frame_index=contact + 1,
            end_frame_index=39,
            start_timestamp=(contact + 1) * 0.05,
            end_timestamp=1.95,
            confidence=0.85,
        ),
    ]
    frame_phases: dict[int, SmashPhase] = {}
    for seg in segments:
        for i in range(seg.start_frame_index, seg.end_frame_index + 1):
            frame_phases[i] = seg.phase
    return PhaseSequence(
        video="clip.mp4",
        segments=segments,
        frame_phases=frame_phases,
        estimated_contact_frame_index=contact,
        estimated_contact_timestamp=contact * 0.05,
        confidence=0.9,
    )


def test_selects_midpoint_for_phase_windows() -> None:
    pose = _pose_only()
    phases = _manual_phases(contact=28)
    selected = select_keyframe_indices(phases, pose, include_contact_neighbors=False)
    by_phase = {s.phase: s for s in selected}

    assert by_phase["PREPARATION"].frame_index == 3  # (0+7)//2
    assert by_phase["BACKSWING"].frame_index == 12  # (8+17)//2
    assert by_phase["ACCELERATION"].frame_index == 22  # (18+27)//2
    assert by_phase["FOLLOW_THROUGH"].frame_index == 34  # (29+39)//2
    assert by_phase["ESTIMATED_CONTACT"].frame_index == 28


def test_estimated_contact_matches_phase_anchor() -> None:
    pose = _pose_only()
    phases = _manual_phases(contact=28)
    selected = select_keyframe_indices(phases, pose, include_contact_neighbors=True)
    contact = next(s for s in selected if s.phase == SmashPhase.ESTIMATED_CONTACT.value)
    assert contact.frame_index == phases.estimated_contact_frame_index
    assert contact.timestamp == pytest.approx(phases.estimated_contact_timestamp)
    assert contact.confidence == pytest.approx(0.95)


def test_contact_neighbors_optional() -> None:
    pose = _pose_only()
    phases = _manual_phases(contact=28)

    with_neighbors = select_keyframe_indices(
        phases, pose, include_contact_neighbors=True
    )
    without = select_keyframe_indices(phases, pose, include_contact_neighbors=False)

    labels_with = {s.phase for s in with_neighbors}
    labels_without = {s.phase for s in without}
    assert CONTACT_MINUS_2 in labels_with
    assert CONTACT_PLUS_2 in labels_with
    assert CONTACT_MINUS_2 not in labels_without
    assert CONTACT_PLUS_2 not in labels_without

    minus = next(s for s in with_neighbors if s.phase == CONTACT_MINUS_2)
    plus = next(s for s in with_neighbors if s.phase == CONTACT_PLUS_2)
    assert minus.frame_index == 26
    assert plus.frame_index == 30


def test_neighbors_omitted_near_clip_edges() -> None:
    pose = _pose_only(n=5)
    phases = PhaseSequence(
        video="short.mp4",
        segments=[
            PhaseSegment(
                phase=SmashPhase.ESTIMATED_CONTACT,
                start_frame_index=1,
                end_frame_index=1,
                start_timestamp=0.05,
                end_timestamp=0.05,
                confidence=0.9,
            )
        ],
        frame_phases={1: SmashPhase.ESTIMATED_CONTACT},
        estimated_contact_frame_index=1,
        estimated_contact_timestamp=0.05,
        confidence=0.9,
    )
    selected = select_keyframe_indices(phases, pose, include_contact_neighbors=True)
    labels = {s.phase for s in selected}
    # contact-2 = -1 invalid; contact+2 = 3 valid
    assert CONTACT_MINUS_2 not in labels
    assert CONTACT_PLUS_2 in labels
    assert SmashPhase.ESTIMATED_CONTACT.value in labels


def test_indices_exist_in_pose_and_match_timestamps() -> None:
    pose = _pose_only()
    phases = _manual_phases()
    selected = select_keyframe_indices(phases, pose, include_contact_neighbors=True)
    pose_ts = {f.frame_index: f.timestamp for f in pose.frames}
    for item in selected:
        assert item.frame_index in pose_ts
        assert item.timestamp == pytest.approx(pose_ts[item.frame_index])
        assert 0.0 <= item.confidence <= 1.0


def test_selection_aligned_with_detected_phases() -> None:
    pose, angles, motion, contact = _synthetic_smash()
    phases = detect_smash_phases(pose, angles, motion)
    selected = select_keyframe_indices(phases, pose, include_contact_neighbors=True)
    by_phase = {s.phase: s for s in selected}

    assert by_phase[SmashPhase.ESTIMATED_CONTACT.value].frame_index == contact
    for phase in (
        SmashPhase.PREPARATION,
        SmashPhase.BACKSWING,
        SmashPhase.ACCELERATION,
        SmashPhase.FOLLOW_THROUGH,
    ):
        if phase.value not in by_phase:
            continue
        idx = by_phase[phase.value].frame_index
        assert phases.phase_at(idx) is phase or (
            phase is SmashPhase.FOLLOW_THROUGH and idx > contact
        )


def test_extract_keyframes_writes_files(tmp_path: Path) -> None:
    # Tiny synthetic video with known frame count.
    video_path = tmp_path / "src.mp4"
    n = 40
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        20.0,
        (64, 48),
    )
    assert writer.isOpened()
    for i in range(n):
        frame = np.full((48, 64, 3), i % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    pose = _pose_only(n=n)
    phases = _manual_phases(contact=28)
    out_dir = tmp_path / "keyframes"
    result = extract_keyframes(
        video_path,
        phases,
        pose,
        out_dir,
        include_contact_neighbors=True,
    )

    assert result.count >= 5
    assert Path(result.output_dir) == out_dir
    for kf in result.keyframes:
        path = Path(kf.file_path)
        assert path.exists()
        assert path.parent == out_dir
        assert kf.phase in path.name
        assert f"{kf.frame_index:06d}" in path.name
        image = cv2.imread(str(path))
        assert image is not None
        assert image.shape[0] == 48 and image.shape[1] == 64

    payload = result.to_dict()
    assert "keyframes" in payload
    assert payload["count"] == result.count
