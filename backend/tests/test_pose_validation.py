"""Tests for offline PoseValidator (NPE, PCK, missing rate, phase breakdown)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from app.schemas.phases import PhaseSegment, PhaseSequence, SmashPhase
from app.schemas.pose import Keypoint, PoseFrame, PoseSequence
from app.validation.annotations import (
    AnnotatedKeypoint,
    AnnotatedPoseFrame,
    PoseValidationAnnotationSet,
    load_annotation_set,
)
from app.validation.joints import BADMINTON_CRITICAL_JOINTS
from app.validation.metrics import (
    missing_rate,
    normalized_pixel_error,
    pck_score,
)
from app.validation.report import POSE_VALIDATION_REPORT_VERSION
from app.validation.validator import (
    PoseValidator,
    export_pose_validation_report,
    validate_pose,
)


def test_normalized_pixel_error_known_coords() -> None:
    # Axis-aligned 0.3 in x → error 0.3
    assert normalized_pixel_error(0.5, 0.5, 0.2, 0.5) == pytest.approx(0.3)
    # 3-4-5 triangle in normalized space
    assert normalized_pixel_error(0.0, 0.0, 0.3, 0.4) == pytest.approx(0.5)
    assert normalized_pixel_error(0.1, 0.1, 0.1, 0.1) == pytest.approx(0.0)


def test_pck_and_missing_rate_calculations() -> None:
    errors = [0.02, 0.04, 0.08, 0.15]
    assert pck_score(errors, threshold=0.05) == pytest.approx(0.5)  # 2/4
    assert pck_score(errors, threshold=0.10) == pytest.approx(0.75)  # 3/4
    assert pck_score(errors, threshold=0.20) == pytest.approx(1.0)
    assert pck_score([], threshold=0.05) is None

    assert missing_rate(annotated=10, detected=7) == pytest.approx(0.3)
    assert missing_rate(annotated=0, detected=0) is None


def test_validator_per_joint_error_and_pck() -> None:
    """Synthetic GT at known positions; preds offset by exact deltas."""
    # GT
    ann = PoseValidationAnnotationSet(
        video="synth.mp4",
        frames=[
            AnnotatedPoseFrame(
                frame_index=0,
                phase="ACCELERATION",
                keypoints={
                    "right_wrist": AnnotatedKeypoint(0.50, 0.40),
                    "right_elbow": AnnotatedKeypoint(0.40, 0.40),
                    "right_shoulder": AnnotatedKeypoint(0.30, 0.40),
                },
            )
        ],
    )
    # Pred: wrist +0.03 in x, elbow exact, shoulder missing
    pred = PoseSequence(
        video="synth.mp4",
        frames=[
            PoseFrame(
                frame_index=0,
                timestamp=0.0,
                keypoints={
                    "right_wrist": Keypoint(0.53, 0.40, 0.9),
                    "right_elbow": Keypoint(0.40, 0.40, 0.8),
                    # right_shoulder omitted → missing
                },
            )
        ],
    )

    report = PoseValidator(pck_thresholds=(0.05, 0.10)).validate(pred, ann)
    frame = report.frames[0]
    by_joint = {d.joint: d for d in frame.joint_errors}

    assert by_joint["right_wrist"].error == pytest.approx(0.03)
    assert by_joint["right_elbow"].error == pytest.approx(0.0)
    assert by_joint["right_shoulder"].predicted is False
    assert by_joint["right_shoulder"].error is None

    assert frame.missing_detection_rate == pytest.approx(1 / 3)
    assert frame.mean_error == pytest.approx(0.015)  # (0.03 + 0.0) / 2
    assert frame.mean_confidence == pytest.approx(0.85)

    assert report.overall is not None
    # Detected errors: 0.03, 0.0 → PCK@0.05 = 2/2
    assert report.overall.pck["0.05"] == pytest.approx(1.0)
    assert report.overall.missing_detection_rate == pytest.approx(1 / 3)

    # Badminton-critical group includes these joints
    assert report.badminton_critical is not None
    assert report.badminton_critical.sample_count == 3
    critical_names = {j.joint for j in report.badminton_critical.per_joint}
    assert critical_names <= set(BADMINTON_CRITICAL_JOINTS)


def test_phase_breakdown_separates_acceleration_and_contact() -> None:
    ann = PoseValidationAnnotationSet(
        video="v.mp4",
        frames=[
            AnnotatedPoseFrame(
                frame_index=10,
                keypoints={"right_wrist": AnnotatedKeypoint(0.5, 0.5)},
            ),
            AnnotatedPoseFrame(
                frame_index=20,
                keypoints={"right_wrist": AnnotatedKeypoint(0.5, 0.5)},
            ),
        ],
    )
    pred = PoseSequence(
        video="v.mp4",
        frames=[
            PoseFrame(
                frame_index=10,
                timestamp=0.5,
                keypoints={"right_wrist": Keypoint(0.52, 0.5, 0.9)},  # err 0.02
            ),
            PoseFrame(
                frame_index=20,
                timestamp=1.0,
                keypoints={"right_wrist": Keypoint(0.60, 0.5, 0.9)},  # err 0.10
            ),
        ],
    )
    phases = PhaseSequence(
        video="v.mp4",
        segments=[
            PhaseSegment(
                SmashPhase.ACCELERATION, 8, 15, 0.4, 0.75, 1.0
            ),
            PhaseSegment(
                SmashPhase.ESTIMATED_CONTACT, 20, 20, 1.0, 1.0, 1.0
            ),
        ],
        frame_phases={
            10: SmashPhase.ACCELERATION,
            20: SmashPhase.ESTIMATED_CONTACT,
        },
        estimated_contact_frame_index=20,
        estimated_contact_timestamp=1.0,
    )

    report = validate_pose(pred, ann, phases=phases, pck_thresholds=(0.05,))
    by_phase = {p.phase: p for p in report.by_phase}
    assert "ACCELERATION" in by_phase
    assert "ESTIMATED_CONTACT" in by_phase
    assert by_phase["ACCELERATION"].mean_error == pytest.approx(0.02)
    assert by_phase["ESTIMATED_CONTACT"].mean_error == pytest.approx(0.10)
    # Contact frame fails PCK@0.05; accel passes
    assert by_phase["ACCELERATION"].pck["0.05"] == pytest.approx(1.0)
    assert by_phase["ESTIMATED_CONTACT"].pck["0.05"] == pytest.approx(0.0)


def test_confidence_threshold_counts_as_missing() -> None:
    ann = PoseValidationAnnotationSet(
        video="v.mp4",
        frames=[
            AnnotatedPoseFrame(
                frame_index=0,
                keypoints={"right_wrist": AnnotatedKeypoint(0.5, 0.5)},
            )
        ],
    )
    pred = PoseSequence(
        video="v.mp4",
        frames=[
            PoseFrame(
                frame_index=0,
                timestamp=0.0,
                keypoints={"right_wrist": Keypoint(0.5, 0.5, 0.2)},
            )
        ],
    )
    report = PoseValidator(confidence_threshold=0.5).validate(pred, ann)
    detail = report.frames[0].joint_errors[0]
    assert detail.predicted is False
    assert report.frames[0].missing_detection_rate == pytest.approx(1.0)


def test_export_pose_validation_json(tmp_path: Path) -> None:
    ann = PoseValidationAnnotationSet(
        video="clip.mp4",
        frames=[
            AnnotatedPoseFrame(
                frame_index=1,
                keypoints={
                    "left_hip": AnnotatedKeypoint(0.4, 0.6),
                    "right_hip": AnnotatedKeypoint(0.5, 0.6),
                },
            )
        ],
    )
    pred = PoseSequence(
        video="clip.mp4",
        frames=[
            PoseFrame(
                frame_index=1,
                timestamp=0.05,
                keypoints={
                    "left_hip": Keypoint(0.41, 0.60, 0.95),
                    "right_hip": Keypoint(0.50, 0.61, 0.95),
                },
            )
        ],
    )
    report = validate_pose(pred, ann)
    out = tmp_path / "pose_validation.json"
    export_pose_validation_report(report, out)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["pose_validation_report_version"] == POSE_VALIDATION_REPORT_VERSION
    assert data["annotated_frame_count"] == 1
    assert len(data["frames"]) == 1
    assert data["frames"][0]["frame_index"] == 1
    assert "joint_errors" in data["frames"][0]
    assert data["overall"]["mean_error"] is not None
    assert data["badminton_critical"]["sample_count"] == 2


def test_load_example_annotation_set() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "pose_validation"
        / "example_annotations.json"
    )
    assert path.is_file()
    loaded = load_annotation_set(path)
    assert loaded.video == "example_smash.mp4"
    assert len(loaded.frames) == 1
    assert "right_wrist" in loaded.frames[0].keypoints


def test_perfect_match_zero_error() -> None:
    joints = {
        "right_shoulder": (0.3, 0.3),
        "right_elbow": (0.4, 0.35),
        "right_wrist": (0.5, 0.4),
        "right_hip": (0.35, 0.55),
        "right_knee": (0.36, 0.7),
        "right_ankle": (0.37, 0.85),
    }
    ann = PoseValidationAnnotationSet(
        video="v.mp4",
        frames=[
            AnnotatedPoseFrame(
                frame_index=5,
                keypoints={
                    n: AnnotatedKeypoint(x, y) for n, (x, y) in joints.items()
                },
            )
        ],
    )
    pred = PoseSequence(
        video="v.mp4",
        frames=[
            PoseFrame(
                frame_index=5,
                timestamp=0.25,
                keypoints={
                    n: Keypoint(x, y, 0.99) for n, (x, y) in joints.items()
                },
            )
        ],
    )
    report = validate_pose(pred, ann, pck_thresholds=(0.05,))
    assert report.overall is not None
    assert report.overall.mean_error == pytest.approx(0.0)
    assert report.overall.pck["0.05"] == pytest.approx(1.0)
    assert report.overall.missing_detection_rate == pytest.approx(0.0)
    # Diagonal known offset check for one joint via hypot
    err = normalized_pixel_error(0.0, 0.0, 1 / math.sqrt(2), 1 / math.sqrt(2))
    assert err == pytest.approx(1.0)
