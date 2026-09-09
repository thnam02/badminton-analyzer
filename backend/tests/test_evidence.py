"""Tests for EvidencePackager — assembly only; issue values match StrokeMetrics."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.processing.technique import evaluate_technique
from app.processing.reference_profiles import build_provisional_smash_right_side
from app.schemas.evidence import (
    CONTACT_TYPE_ESTIMATED,
    EVIDENCE_VERSION,
    STROKE_TYPE_SMASH,
)
from app.schemas.keyframes import Keyframe, KeyframeSet
from app.schemas.technique import TechniqueIssue
from app.schemas.video_quality import VideoQualityMetrics, VideoQualityReport
from app.services.evidence_packager import (
    ISSUE_PRIMARY_METRIC,
    EvidencePackager,
    issue_source_metric,
    package_evidence,
)
from tests.test_technique import _build_pipeline


def _quality(video: str = "smash.mp4") -> VideoQualityReport:
    return VideoQualityReport(
        video=video,
        usable=True,
        analysis_confidence=0.85,
        metrics=VideoQualityMetrics(
            fps=30.0,
            width=1280,
            height=720,
            full_body_coverage=0.95,
            racket_arm_visibility=0.98,
            mean_pose_confidence=0.9,
            missing_keypoint_fraction=0.05,
            interpolated_keypoint_fraction=0.02,
            player_size_ratio=0.5,
            camera_stability=0.8,
        ),
        warnings=[],
    )


def _keyframes(video: str = "smash.mp4") -> KeyframeSet:
    return KeyframeSet(
        video=video,
        output_dir="/tmp/keyframes",
        keyframes=[
            Keyframe(
                phase="ESTIMATED_CONTACT",
                frame_index=28,
                timestamp=1.4,
                file_path="/tmp/keyframes/ESTIMATED_CONTACT_f000028.jpg",
                confidence=0.9,
            )
        ],
    )


def test_package_contains_versioned_coaching_fields(tmp_path: Path) -> None:
    _, _, _, phases, metrics = _build_pipeline()
    technique = evaluate_technique(metrics, profile=build_provisional_smash_right_side())
    quality = _quality()
    keyframes = _keyframes()

    package = EvidencePackager().package(
        video_quality=quality,
        phases=phases,
        metrics=metrics,
        technique=technique,
        keyframes=keyframes,
        handedness=None,
    )

    assert package.evidence_version == EVIDENCE_VERSION
    assert package.stroke_type == STROKE_TYPE_SMASH
    assert package.handedness is None
    assert package.contact.contact_type == CONTACT_TYPE_ESTIMATED
    assert package.contact.frame_index == phases.estimated_contact_frame_index
    assert package.contact.confidence > 0
    assert package.phase_boundaries == [seg.to_dict() for seg in phases.segments]
    assert "frame_phases" not in package.to_dict()
    assert package.metrics == metrics.to_dict()
    assert package.video_quality == quality.to_dict()
    assert package.technique_issues == [i.to_dict() for i in technique.issues]
    assert package.keyframes == [kf.to_dict() for kf in keyframes.keyframes]
    assert "frames" not in package.to_dict()  # no raw pose sequence
    assert "keypoints" not in package.to_dict()

    out = tmp_path / "evidence.json"
    package.save_json(out)
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert '"evidence_version"' in text
    assert EVIDENCE_VERSION in text


def test_all_issue_measured_values_match_stroke_metrics() -> None:
    _, _, _, phases, metrics = _build_pipeline()
    # Force every known rule to fire so each measured_value is checked.
    metrics.contact_elbow_angle_deg = 120.0
    metrics.knee_contribution_deg = 4.0
    metrics.peak_elbow_omega_offset_frames = 10
    metrics.acceleration_phase_fraction = 0.02
    metrics.contact_wrist_y_normalized = 0.85
    metrics.follow_through_speed_ratio = 0.05
    metrics.follow_through_frame_count = 0

    technique = evaluate_technique(
        metrics,
        profile=build_provisional_smash_right_side(),
    )
    assert technique.issue_count >= 4

    package = package_evidence(
        video_quality=_quality(),
        phases=phases,
        metrics=metrics,
        technique=technique,
        keyframes=_keyframes(),
    )

    packaged_by_code = {i["code"]: i for i in package.technique_issues}
    for issue in technique.issues:
        field_name = issue_source_metric(issue)
        if not field_name:
            continue
        expected = getattr(metrics, field_name)
        assert expected is not None
        assert issue.measured_value == pytest.approx(float(expected))
        assert packaged_by_code[issue.code]["measured_value"] == pytest.approx(
            float(expected)
        )
        # Packaged metrics mirror the same underlying values.
        assert package.metrics[field_name] == pytest.approx(float(expected))
        assert packaged_by_code[issue.code].get("reference_profile_id")


def test_issue_source_metric_covers_known_codes() -> None:
    from app.schemas.phases import SmashPhase
    from app.schemas.stroke_metrics import StrokeMetrics
    from app.schemas.technique import IssueSeverity, ReferenceRange

    for code, field_name in ISSUE_PRIMARY_METRIC.items():
        if code == "POOR_ARM_ACCELERATION_TIMING":
            unit = "frames"
        elif code == "WEAK_FOLLOW_THROUGH":
            unit = "speed_ratio"
        else:
            unit = "deg"
        issue = TechniqueIssue(
            code=code,
            phase=SmashPhase.ESTIMATED_CONTACT,
            severity=IssueSeverity.LOW,
            confidence=0.5,
            measured_value=1.0,
            reference_range=ReferenceRange(min=0.0, max=1.0),
            unit=unit,
        )
        resolved = issue_source_metric(issue)
        assert resolved == field_name
        assert hasattr(StrokeMetrics, resolved)


def test_handedness_passed_through_when_available() -> None:
    _, _, _, phases, metrics = _build_pipeline()
    technique = evaluate_technique(metrics, profile=build_provisional_smash_right_side())
    package = package_evidence(
        video_quality=_quality(),
        phases=phases,
        metrics=metrics,
        technique=technique,
        keyframes=_keyframes(),
        handedness="RIGHT",
    )
    assert package.handedness == "RIGHT"
    assert package.stroke_type == STROKE_TYPE_SMASH


def test_unusable_quality_zeros_analysis_confidence() -> None:
    _, _, _, phases, metrics = _build_pipeline()
    technique = evaluate_technique(metrics, profile=build_provisional_smash_right_side())
    quality = _quality()
    quality.usable = False
    quality.analysis_confidence = 0.4
    package = package_evidence(
        video_quality=quality,
        phases=phases,
        metrics=metrics,
        technique=technique,
        keyframes=_keyframes(),
    )
    assert package.analysis_confidence == pytest.approx(0.0)
