"""Tests for offline AngleValidator (MAE, median, p90, phases, confidence)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from app.processing.angles import angle_at_vertex, compute_joint_angle
from app.schemas.angles import AngleFrame, AngleSequence
from app.schemas.phases import PhaseSegment, PhaseSequence, SmashPhase
from app.schemas.pose import Keypoint, PoseFrame, PoseSequence
from app.validation.angle_annotations import (
    AnnotatedAngleFrame,
    AngleValidationAnnotationSet,
    load_angle_annotation_set,
)
from app.validation.angle_metrics import (
    invalid_rate,
    mean_absolute_error,
    median_absolute_error,
    percentile_absolute_error,
)
from app.validation.angle_report import ANGLE_VALIDATION_REPORT_VERSION
from app.validation.angle_validator import (
    AngleValidator,
    export_angle_validation_report,
    resolve_ground_truth_angles,
    validate_angles,
)
from app.validation.annotations import AnnotatedKeypoint


def _kp(x: float, y: float, conf: float = 0.95) -> Keypoint:
    return Keypoint(x=x, y=y, confidence=conf)


def test_angle_metric_helpers() -> None:
    errs = [2.0, 4.0, 6.0, 8.0, 100.0]
    assert mean_absolute_error(errs) == pytest.approx(24.0)
    assert median_absolute_error(errs) == pytest.approx(6.0)
    # p90 of 5 samples → index 3.6 → interpolate 8 and 100
    p90 = percentile_absolute_error(errs, percentile=90.0)
    assert p90 == pytest.approx(8.0 + 0.6 * (100.0 - 8.0))
    assert invalid_rate(annotated=10, valid_predictions=7) == pytest.approx(0.3)
    assert mean_absolute_error([]) is None


def test_exact_geometry_gt_from_keypoints_matches_formula() -> None:
    """90° elbow from known keypoints — GT derivation uses production formula."""
    ann = AnnotatedAngleFrame(
        frame_index=0,
        keypoints={
            "right_shoulder": AnnotatedKeypoint(0.0, 0.0),
            "right_elbow": AnnotatedKeypoint(1.0, 0.0),
            "right_wrist": AnnotatedKeypoint(1.0, 1.0),
        },
    )
    gt = resolve_ground_truth_angles(ann)
    assert gt["right_elbow"] == pytest.approx(90.0)
    # Same as compute_joint_angle / angle_at_vertex
    assert angle_at_vertex(_kp(0, 0), _kp(1, 0), _kp(1, 1)) == pytest.approx(90.0)


def test_direct_gt_angles_preferred_over_keypoints() -> None:
    ann = AnnotatedAngleFrame(
        frame_index=0,
        angles={"right_elbow": 120.0},
        keypoints={
            "right_shoulder": AnnotatedKeypoint(0.0, 0.0),
            "right_elbow": AnnotatedKeypoint(1.0, 0.0),
            "right_wrist": AnnotatedKeypoint(1.0, 1.0),  # would be 90°
        },
    )
    gt = resolve_ground_truth_angles(ann)
    assert gt["right_elbow"] == pytest.approx(120.0)


def test_validator_mae_median_p90_and_missing() -> None:
    """Predictions offset from exact GT by known amounts."""
    # GT elbows: 90 and 180; preds: 95 and missing → errors [5]
    ann = AngleValidationAnnotationSet(
        video="synth.mp4",
        frames=[
            AnnotatedAngleFrame(
                frame_index=0,
                phase="ACCELERATION",
                pose_confidence=0.9,
                angles={"right_elbow": 90.0, "right_knee": 180.0},
            ),
            AnnotatedAngleFrame(
                frame_index=1,
                phase="ESTIMATED_CONTACT",
                pose_confidence=0.4,
                angles={"right_elbow": 90.0},
            ),
        ],
    )
    preds = AngleSequence(
        video="synth.mp4",
        frames=[
            AngleFrame(0, 0.0, right_elbow=95.0, right_knee=170.0, right_shoulder=None),
            AngleFrame(1, 0.05, right_elbow=None, right_knee=None, right_shoulder=None),
        ],
    )
    report = AngleValidator().validate(preds, ann)
    assert report.overall is not None
    # Valid errors: |95-90|=5, |170-180|=10  (frame1 elbow missing)
    assert report.overall.mae == pytest.approx(7.5)
    assert report.overall.median_absolute_error == pytest.approx(7.5)
    assert report.overall.sample_count == 3  # 2 + 1 annotated
    assert report.overall.invalid_rate == pytest.approx(1 / 3)

    by_name = {a.name: a for a in report.overall.per_angle}
    assert by_name["right_elbow"].mae == pytest.approx(5.0)
    assert by_name["right_elbow"].invalid_rate == pytest.approx(0.5)
    assert by_name["right_knee"].mae == pytest.approx(10.0)


def test_left_and_right_angles_via_smoothed_pose() -> None:
    """Left angles come from smoothed pose using the same geometric formula."""
    # Left elbow 90°: shoulder(0,0), elbow(1,0), wrist(1,1)
    # Right elbow 60° known geometry
    distal_x = 0.5
    distal_y = math.sqrt(3) / 2
    ann = AngleValidationAnnotationSet(
        video="v.mp4",
        frames=[
            AnnotatedAngleFrame(
                frame_index=0,
                angles={"left_elbow": 90.0, "right_elbow": 60.0},
            )
        ],
    )
    # Pipeline only has right_elbow (slightly wrong)
    preds = AngleSequence(
        video="v.mp4",
        frames=[
            AngleFrame(0, 0.0, right_elbow=62.0, right_knee=None, right_shoulder=None),
        ],
    )
    pose = PoseSequence(
        video="v.mp4",
        frames=[
            PoseFrame(
                frame_index=0,
                timestamp=0.0,
                keypoints={
                    "left_shoulder": _kp(0.0, 0.0),
                    "left_elbow": _kp(1.0, 0.0),
                    "left_wrist": _kp(1.0, 1.0),
                    "right_shoulder": _kp(0.0, 0.0),
                    "right_elbow": _kp(0.0, 0.0),  # unused for left
                    "right_wrist": _kp(0.0, 0.0),
                    # right pipeline angle comes from AngleFrame, not these
                },
            )
        ],
    )
    # Fix: right pred is from AngleFrame; left from pose
    report = validate_angles(preds, ann, smoothed_pose=pose)
    by = {e.angle_name: e for e in report.frames[0].angle_errors}
    assert by["right_elbow"].absolute_error == pytest.approx(2.0)
    assert by["left_elbow"].pred_degrees == pytest.approx(90.0)
    assert by["left_elbow"].absolute_error == pytest.approx(0.0)

    # Sanity: production formula agrees
    assert compute_joint_angle(
        pose.frames[0].keypoints,
        "left_shoulder",
        "left_elbow",
        "left_wrist",
        confidence_threshold=0.5,
    ) == pytest.approx(90.0)
    del distal_x, distal_y


def test_phase_breakdown_orders_smash_phases() -> None:
    ann = AngleValidationAnnotationSet(
        video="v.mp4",
        frames=[
            AnnotatedAngleFrame(
                frame_index=5, angles={"right_elbow": 100.0}
            ),
            AnnotatedAngleFrame(
                frame_index=20, angles={"right_elbow": 100.0}
            ),
        ],
    )
    preds = AngleSequence(
        video="v.mp4",
        frames=[
            AngleFrame(5, 0.25, right_elbow=102.0),
            AngleFrame(20, 1.0, right_elbow=110.0),
        ],
    )
    phases = PhaseSequence(
        video="v.mp4",
        segments=[
            PhaseSegment(SmashPhase.PREPARATION, 0, 10, 0.0, 0.5, 1.0),
            PhaseSegment(SmashPhase.ESTIMATED_CONTACT, 20, 20, 1.0, 1.0, 1.0),
        ],
        frame_phases={
            5: SmashPhase.PREPARATION,
            20: SmashPhase.ESTIMATED_CONTACT,
        },
        estimated_contact_frame_index=20,
        estimated_contact_timestamp=1.0,
    )
    report = validate_angles(preds, ann, phases=phases)
    names = [p.phase for p in report.by_phase]
    assert names.index("PREPARATION") < names.index("ESTIMATED_CONTACT")
    by = {p.phase: p for p in report.by_phase}
    assert by["PREPARATION"].mae == pytest.approx(2.0)
    assert by["ESTIMATED_CONTACT"].mae == pytest.approx(10.0)


def test_confidence_bins_separate_low_and_high() -> None:
    ann = AngleValidationAnnotationSet(
        video="v.mp4",
        frames=[
            AnnotatedAngleFrame(
                frame_index=0,
                pose_confidence=0.3,
                angles={"right_elbow": 90.0},
            ),
            AnnotatedAngleFrame(
                frame_index=1,
                pose_confidence=0.9,
                angles={"right_elbow": 90.0},
            ),
        ],
    )
    preds = AngleSequence(
        video="v.mp4",
        frames=[
            AngleFrame(0, 0.0, right_elbow=110.0),  # err 20 at low conf
            AngleFrame(1, 0.05, right_elbow=91.0),  # err 1 at high conf
        ],
    )
    report = AngleValidator().validate(preds, ann)
    by = {b.label: b for b in report.by_confidence}
    assert by["low_<0.5"].mae == pytest.approx(20.0)
    assert by["high_>=0.75"].mae == pytest.approx(1.0)
    assert by["mid_0.5_0.75"].sample_count == 0


def test_export_angle_validation_json(tmp_path: Path) -> None:
    ann = AngleValidationAnnotationSet(
        video="clip.mp4",
        frames=[
            AnnotatedAngleFrame(
                frame_index=3,
                angles={"right_shoulder": 90.0, "left_knee": 170.0},
            )
        ],
    )
    preds = AngleSequence(
        video="clip.mp4",
        frames=[AngleFrame(3, 0.15, right_shoulder=92.0)],
    )
    # left_knee needs pose
    pose = PoseSequence(
        video="clip.mp4",
        frames=[
            PoseFrame(
                frame_index=3,
                timestamp=0.15,
                keypoints={
                    "left_hip": _kp(0.2, 0.2),
                    "left_knee": _kp(0.2, 0.5),
                    "left_ankle": _kp(0.5, 0.5),  # 90° not 170 — large error
                },
            )
        ],
    )
    report = validate_angles(preds, ann, smoothed_pose=pose)
    out = tmp_path / "angle_validation.json"
    export_angle_validation_report(report, out)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["angle_validation_report_version"] == ANGLE_VALIDATION_REPORT_VERSION
    assert data["annotated_frame_count"] == 1
    assert len(data["frames"]) == 1
    assert "angle_errors" in data["frames"][0]
    assert data["overall"]["mae"] is not None
    assert "by_confidence" in data
    assert "by_phase" in data


def test_load_example_angle_annotations() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "pose_validation"
        / "example_angle_annotations.json"
    )
    loaded = load_angle_annotation_set(path)
    assert loaded.video == "example_smash.mp4"
    assert "right_elbow" in loaded.frames[0].angles


def test_shoulder_and_knee_exact_geometry_end_to_end() -> None:
    """Full pipeline: exact GT angles; preds match via AngleSequence / pose."""
    ann = AngleValidationAnnotationSet(
        video="v.mp4",
        frames=[
            AnnotatedAngleFrame(
                frame_index=0,
                angles={
                    "right_shoulder": 90.0,
                    "right_knee": 90.0,
                    "left_elbow": 180.0,
                },
                keypoints={
                    # Geometry for left elbow (collinear → 180°) drawn in debug/pose path
                    "left_shoulder": AnnotatedKeypoint(0.0, 0.0),
                    "left_elbow": AnnotatedKeypoint(0.5, 0.0),
                    "left_wrist": AnnotatedKeypoint(1.0, 0.0),
                    "right_hip": AnnotatedKeypoint(0.4, 0.8),
                    "right_shoulder": AnnotatedKeypoint(0.4, 0.4),
                    "right_elbow": AnnotatedKeypoint(0.7, 0.4),
                    "right_knee": AnnotatedKeypoint(0.2, 0.5),
                    "right_ankle": AnnotatedKeypoint(0.5, 0.5),
                },
            )
        ],
    )
    # Direct angles win; also verify keypoint-derived left elbow matches formula.
    derived = resolve_ground_truth_angles(
        AnnotatedAngleFrame(
            frame_index=0,
            keypoints={
                "left_shoulder": AnnotatedKeypoint(0.0, 0.0),
                "left_elbow": AnnotatedKeypoint(0.5, 0.0),
                "left_wrist": AnnotatedKeypoint(1.0, 0.0),
            },
        )
    )
    assert derived["left_elbow"] == pytest.approx(180.0)

    preds = AngleSequence(
        video="v.mp4",
        frames=[
            AngleFrame(
                0,
                0.0,
                right_elbow=None,
                right_knee=90.0,
                right_shoulder=90.0,
            )
        ],
    )
    pose = PoseSequence(
        video="v.mp4",
        frames=[
            PoseFrame(
                frame_index=0,
                timestamp=0.0,
                keypoints={
                    name: _kp(kp.x, kp.y)
                    for name, kp in ann.frames[0].keypoints.items()
                },
            )
        ],
    )
    report = validate_angles(preds, ann, smoothed_pose=pose)
    by = {e.angle_name: e for e in report.frames[0].angle_errors}
    assert by["right_shoulder"].absolute_error == pytest.approx(0.0)
    assert by["right_knee"].absolute_error == pytest.approx(0.0)
    assert by["left_elbow"].absolute_error == pytest.approx(0.0)
