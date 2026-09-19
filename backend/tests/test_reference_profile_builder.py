"""Tests for deterministic reference-profile builder statistics and grouping."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.processing.reference_profile_builder import (
    METRIC_CONTACT_ELBOW,
    METRIC_PEAK_WRIST_SPEED,
    ReferenceProfileBuilder,
    build_reference_profiles,
    export_built_reference_profiles,
    summarize_metric_values,
)
from app.schemas.built_reference import REFERENCE_PROFILE_BUILD_VERSION
from app.schemas.reference_dataset import (
    CAMERA_VIEW_REAR_45,
    CAMERA_VIEW_SIDE_45,
    PROTOCOL_STROKE_TYPE,
    AnalysisArtifactRefs,
    CameraSetup,
    Player,
    RecordingSession,
    ReferenceDatasetManifest,
    StrokeSample,
)


def _stroke(
    stroke_id: str,
    *,
    handedness: str = "RIGHT",
    camera_view: str = CAMERA_VIEW_SIDE_45,
    skill_group: str = "intermediate",
    acceptance: str = "accepted",
    analysis_id: str = "",
) -> StrokeSample:
    return StrokeSample(
        stroke_id=stroke_id,
        session_id="sess",
        anonymized_player_id="p1",
        stroke_type=PROTOCOL_STROKE_TYPE,
        take_number=int(stroke_id.rsplit("_", 1)[-1].lstrip("t") or 1)
        if "_t" in stroke_id
        else 1,
        camera_view=camera_view,
        fps=60,
        width=1920,
        height=1080,
        handedness=handedness,
        skill_group=skill_group,
        acceptance_status=acceptance,
        artifact_refs=AnalysisArtifactRefs(analysis_id=analysis_id),
    )


def test_summarize_metric_values_known_percentiles() -> None:
    """[1,2,3,4,5] → exact median/mean/P10/P25/P75/P90/IQR/std."""
    dist = summarize_metric_values(
        metric_id="demo",
        unit="unit",
        direction="in_range",
        values=[1, 2, 3, 4, 5],
        sample_count=5,
    )
    assert dist.observed_count == 5
    assert dist.missing_rate == pytest.approx(0.0)
    assert dist.median == pytest.approx(3.0)
    assert dist.mean == pytest.approx(3.0)
    assert dist.p10 == pytest.approx(1.4)
    assert dist.p25 == pytest.approx(2.0)
    assert dist.p75 == pytest.approx(4.0)
    assert dist.p90 == pytest.approx(4.6)
    assert dist.iqr == pytest.approx(2.0)
    assert dist.std == pytest.approx((2.5) ** 0.5)


def test_summarize_missing_rate_with_partial_observations() -> None:
    dist = summarize_metric_values(
        metric_id=METRIC_CONTACT_ELBOW,
        unit="deg",
        direction="higher_is_better",
        values=[150.0, 160.0],
        sample_count=4,
    )
    assert dist.sample_count == 4
    assert dist.observed_count == 2
    assert dist.missing_rate == pytest.approx(0.5)
    assert dist.median == pytest.approx(155.0)


def test_builder_groups_by_stroke_hand_view_skill() -> None:
    strokes = [
        _stroke("a_t1", skill_group="beginner", camera_view=CAMERA_VIEW_SIDE_45),
        _stroke("a_t2", skill_group="beginner", camera_view=CAMERA_VIEW_SIDE_45),
        _stroke("b_t1", skill_group="intermediate", camera_view=CAMERA_VIEW_SIDE_45),
        _stroke(
            "c_t1",
            skill_group="beginner",
            camera_view=CAMERA_VIEW_REAR_45,
            handedness="LEFT",
        ),
    ]
    manifest = ReferenceDatasetManifest(strokes=strokes)
    metrics = {
        "a_t1": {METRIC_CONTACT_ELBOW: 150.0, METRIC_PEAK_WRIST_SPEED: 1.0},
        "a_t2": {METRIC_CONTACT_ELBOW: 170.0, METRIC_PEAK_WRIST_SPEED: 3.0},
        "b_t1": {METRIC_CONTACT_ELBOW: 160.0, METRIC_PEAK_WRIST_SPEED: 2.0},
        "c_t1": {METRIC_CONTACT_ELBOW: 155.0, METRIC_PEAK_WRIST_SPEED: 2.5},
    }
    result = build_reference_profiles(
        manifest,
        metrics_by_stroke_id=metrics,
        exclude_low_quality=False,
    )
    assert result.total_included_strokes == 4
    assert len(result.profiles) == 3

    by_id = {p.profile_id: p for p in result.profiles}
    beginner_side = next(
        p
        for p in result.profiles
        if p.skill_level == "beginner" and p.camera_view == CAMERA_VIEW_SIDE_45
    )
    assert beginner_side.stroke_count == 2
    elbow = beginner_side.metrics[METRIC_CONTACT_ELBOW]
    assert elbow.median == pytest.approx(160.0)
    assert elbow.mean == pytest.approx(160.0)
    assert elbow.p10 == pytest.approx(152.0)  # 150 + 0.1*(170-150)
    assert elbow.observed_count == 2

    left_rear = next(p for p in result.profiles if p.handedness == "LEFT")
    assert left_rear.camera_view == CAMERA_VIEW_REAR_45
    assert left_rear.stroke_count == 1

    # Deterministic profile ordering / ids.
    assert [p.profile_id for p in result.profiles] == sorted(by_id.keys())


def test_builder_excludes_invalid_and_low_quality() -> None:
    strokes = [
        _stroke("ok_t1", acceptance="accepted"),
        _stroke("bad_t1", acceptance="rejected"),
        _stroke("lowq_t1", acceptance="accepted"),
        _stroke("pending_t1", acceptance="pending"),
    ]
    manifest = ReferenceDatasetManifest(strokes=strokes)
    metrics = {
        "ok_t1": {METRIC_CONTACT_ELBOW: 160.0},
        "bad_t1": {METRIC_CONTACT_ELBOW: 100.0},
        "lowq_t1": {METRIC_CONTACT_ELBOW: 140.0},
        "pending_t1": {METRIC_CONTACT_ELBOW: 150.0},
    }
    quality = {
        "ok_t1": {"usable": True, "analysis_confidence": 0.9},
        "lowq_t1": {"usable": True, "analysis_confidence": 0.2},
    }
    result = build_reference_profiles(
        manifest,
        metrics_by_stroke_id=metrics,
        quality_by_stroke_id=quality,
        exclude_low_quality=True,
        min_analysis_confidence=0.5,
    )
    assert result.total_included_strokes == 1
    assert result.exclusion_reasons["acceptance_rejected"] == 1
    assert result.exclusion_reasons["acceptance_pending"] == 1
    assert result.exclusion_reasons["quality_low_confidence"] == 1
    assert result.profiles[0].included_stroke_ids == ["ok_t1"]


def test_builder_is_deterministic() -> None:
    strokes = [
        _stroke("z_t1", skill_group="advanced"),
        _stroke("a_t1", skill_group="advanced"),
        _stroke("m_t1", skill_group="advanced"),
    ]
    manifest = ReferenceDatasetManifest(strokes=strokes)
    metrics = {
        "z_t1": {METRIC_CONTACT_ELBOW: 150.0, METRIC_PEAK_WRIST_SPEED: 1.5},
        "a_t1": {METRIC_CONTACT_ELBOW: 170.0, METRIC_PEAK_WRIST_SPEED: 2.5},
        "m_t1": {METRIC_CONTACT_ELBOW: 160.0, METRIC_PEAK_WRIST_SPEED: None},
    }
    # Intentionally omit wrist speed on one row via None → missing_rate.
    metrics["m_t1"] = {METRIC_CONTACT_ELBOW: 160.0}

    first = build_reference_profiles(
        manifest, metrics_by_stroke_id=metrics, exclude_low_quality=False
    )
    second = build_reference_profiles(
        manifest, metrics_by_stroke_id=metrics, exclude_low_quality=False
    )
    assert first.to_dict() == second.to_dict()
    assert first.build_version == REFERENCE_PROFILE_BUILD_VERSION

    wrist = first.profiles[0].metrics[METRIC_PEAK_WRIST_SPEED]
    assert wrist.observed_count == 2
    assert wrist.missing_rate == pytest.approx(1 / 3)


def test_loads_metrics_from_dataset_export_artifact(tmp_path: Path) -> None:
    analysis_id = "abc123"
    export_path = tmp_path / f"{analysis_id}_dataset.json"
    export_path.write_text(
        json.dumps(
            {
                "dataset_export_version": "1.0.0",
                "analysis_id": analysis_id,
                "pose_metrics": {
                    METRIC_CONTACT_ELBOW: 158.0,
                    METRIC_PEAK_WRIST_SPEED: 2.2,
                    "peak_elbow_angular_velocity": 400.0,
                },
            }
        ),
        encoding="utf-8",
    )
    stroke = _stroke("s1", analysis_id=analysis_id)
    stroke.artifact_refs.dataset_export_path = str(export_path)
    manifest = ReferenceDatasetManifest(strokes=[stroke])
    result = ReferenceProfileBuilder(
        exclude_low_quality=False,
        artifact_root=tmp_path,
    ).build(manifest)
    assert result.total_included_strokes == 1
    assert result.profiles[0].metrics[METRIC_CONTACT_ELBOW].median == pytest.approx(
        158.0
    )
    assert result.profiles[0].metrics[
        "peak_elbow_angular_velocity"
    ].median == pytest.approx(400.0)


def test_export_and_runtime_projection(tmp_path: Path) -> None:
    strokes = [_stroke("s1"), _stroke("s2")]
    manifest = ReferenceDatasetManifest(
        players=[
            Player(
                anonymized_player_id="p1",
                skill_group="intermediate",
                handedness="RIGHT",
            )
        ],
        sessions=[
            RecordingSession(
                session_id="sess",
                anonymized_player_id="p1",
                camera_setup=CameraSetup(
                    camera_view=CAMERA_VIEW_SIDE_45,
                    fps=60,
                    width=1920,
                    height=1080,
                ),
            )
        ],
        strokes=strokes,
    )
    metrics = {
        "s1": {METRIC_CONTACT_ELBOW: 150.0},
        "s2": {METRIC_CONTACT_ELBOW: 170.0},
    }
    result = build_reference_profiles(
        manifest, metrics_by_stroke_id=metrics, exclude_low_quality=False
    )
    out = tmp_path / "reference_profiles_built.json"
    export_built_reference_profiles(result, out)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["reference_profile_build_version"] == REFERENCE_PROFILE_BUILD_VERSION
    assert data["profile_count"] == 1

    runtime = result.profiles[0].to_reference_profile()
    assert runtime.handedness == "RIGHT"
    assert runtime.camera_view == CAMERA_VIEW_SIDE_45
    band = runtime.metrics[METRIC_CONTACT_ELBOW]
    assert band.median == pytest.approx(160.0)
    assert band.lower_percentile == pytest.approx(152.0)
    assert band.upper_percentile == pytest.approx(168.0)
    assert band.provisional is True
