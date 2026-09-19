"""Synthetic tests for phase / contact timing validation metrics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.schemas.contact import CONTACT_TYPE_KINEMATIC, CONTACT_TYPE_TRACKED, ContactEvent
from app.schemas.phases import PhaseSegment, PhaseSequence, SmashPhase
from app.validation.phase_contact_annotations import (
    BoundaryAnnotation,
    PhaseContactAnnotationSet,
    VideoPhaseContactAnnotation,
    load_phase_contact_annotation_set,
)
from app.validation.phase_contact_debug_viz import render_worst_timing_timelines
from app.validation.phase_contact_metrics import (
    absolute_frame_error,
    absolute_time_error_ms,
    frame_error_to_ms,
    mean_or_none,
    median_or_none,
    percentile_or_none,
    within_tolerance_rate,
)
from app.validation.phase_contact_report import PHASE_CONTACT_VALIDATION_REPORT_VERSION
from app.validation.phase_contact_validator import (
    PhaseContactValidator,
    export_phase_contact_validation_report,
    extract_predicted_boundaries,
    validate_phase_contact,
)


def _seg(
    phase: SmashPhase,
    start: int,
    end: int,
    *,
    fps: float = 30.0,
) -> PhaseSegment:
    return PhaseSegment(
        phase=phase,
        start_frame_index=start,
        end_frame_index=end,
        start_timestamp=start / fps,
        end_timestamp=end / fps,
        confidence=1.0,
    )


def _phases(
    *,
    prep_end: int = 10,
    back_end: int = 18,
    accel_start: int = 19,
    contact: int = 28,
    follow_end: int = 39,
    fps: float = 30.0,
    video: str = "clip.mp4",
) -> PhaseSequence:
    segments = [
        _seg(SmashPhase.PREPARATION, 0, prep_end, fps=fps),
        _seg(SmashPhase.BACKSWING, prep_end + 1, back_end, fps=fps),
        _seg(SmashPhase.ACCELERATION, accel_start, contact - 1, fps=fps),
        _seg(SmashPhase.ESTIMATED_CONTACT, contact, contact, fps=fps),
        _seg(SmashPhase.FOLLOW_THROUGH, contact + 1, follow_end, fps=fps),
    ]
    frame_phases = {}
    for seg in segments:
        for i in range(seg.start_frame_index, seg.end_frame_index + 1):
            frame_phases[i] = seg.phase
    return PhaseSequence(
        video=video,
        segments=segments,
        frame_phases=frame_phases,
        estimated_contact_frame_index=contact,
        estimated_contact_timestamp=contact / fps,
        confidence=0.9,
    )


def _ann(
    video: str = "clip.mp4",
    *,
    fps: float = 30.0,
    contact: int = 28,
    conf: float | None = 0.8,
    **boundary_frames: int,
) -> VideoPhaseContactAnnotation:
    defaults = {
        "PREPARATION_END": 10,
        "BACKSWING_END": 18,
        "ACCELERATION_START": 19,
        "CONTACT": contact,
        "FOLLOW_THROUGH_END": 39,
    }
    defaults.update(boundary_frames)
    boundaries = {
        name: BoundaryAnnotation(frame_index=f, timestamp=f / fps)
        for name, f in defaults.items()
    }
    return VideoPhaseContactAnnotation(
        video=video,
        fps=fps,
        boundaries=boundaries,
        analysis_confidence=conf,
    )


def test_timing_metric_helpers_known_values() -> None:
    assert absolute_frame_error(30, 28) == 2
    assert absolute_time_error_ms(1.0, 0.9) == pytest.approx(100.0)
    assert frame_error_to_ms(3, fps=30.0) == pytest.approx(100.0)
    assert mean_or_none([1.0, 3.0, 5.0]) == pytest.approx(3.0)
    assert median_or_none([1.0, 3.0, 100.0]) == pytest.approx(3.0)
    assert percentile_or_none([1.0, 2.0, 3.0, 4.0, 5.0], percentile=90.0) == pytest.approx(
        4.6
    )
    assert within_tolerance_rate([0.0, 1.0, 2.0, 5.0], tolerance=2.0) == pytest.approx(
        0.75
    )
    assert within_tolerance_rate([], tolerance=1.0) is None


def test_extract_predicted_boundaries_from_phases_and_contact() -> None:
    phases = _phases(contact=28)
    contact = ContactEvent(
        contact_type=CONTACT_TYPE_TRACKED,
        frame_index=31,
        timestamp=31 / 30.0,
        confidence=0.9,
        kinematic_frame_index=28,
        kinematic_timestamp=28 / 30.0,
    )
    pred = extract_predicted_boundaries(phases, contact)
    assert pred["PREPARATION_END"][0] == 10
    assert pred["BACKSWING_END"][0] == 18
    assert pred["ACCELERATION_START"][0] == 19
    assert pred["FOLLOW_THROUGH_END"][0] == 39
    # Resolved contact overrides phase estimate
    assert pred["CONTACT"][0] == 31
    assert pred["CONTACT"][1] == pytest.approx(31 / 30.0)


def test_validate_video_exact_match_zero_error() -> None:
    phases = _phases(contact=28)
    contact = ContactEvent(
        contact_type=CONTACT_TYPE_KINEMATIC,
        frame_index=28,
        timestamp=28 / 30.0,
        confidence=0.8,
        kinematic_frame_index=28,
        kinematic_timestamp=28 / 30.0,
    )
    result = PhaseContactValidator().validate_video(_ann(), phases, contact)
    assert result.contact_frame_error == 0
    assert result.contact_time_error_ms == pytest.approx(0.0)
    for err in result.boundary_errors:
        assert err.predicted
        assert err.absolute_frame_error == 0
        assert err.absolute_time_error_ms == pytest.approx(0.0)


def test_validate_known_frame_and_ms_errors() -> None:
    """Pred contact at 31 vs GT 28 → 3 frames = 100 ms at 30 FPS."""
    phases = _phases(contact=31)
    contact = ContactEvent(
        contact_type=CONTACT_TYPE_TRACKED,
        frame_index=31,
        timestamp=31 / 30.0,
        confidence=0.9,
        kinematic_frame_index=28,
        kinematic_timestamp=28 / 30.0,
    )
    # Shift prep end by 1 frame in prediction
    phases.segments[0] = _seg(SmashPhase.PREPARATION, 0, 11)
    result = PhaseContactValidator().validate_video(
        _ann(contact=28), phases, contact
    )
    by = {b.boundary: b for b in result.boundary_errors}
    assert by["CONTACT"].absolute_frame_error == 3
    assert by["CONTACT"].absolute_time_error_ms == pytest.approx(100.0)
    assert by["PREPARATION_END"].absolute_frame_error == 1
    assert result.contact_type == CONTACT_TYPE_TRACKED


def test_aggregate_tolerances_and_contact_type_split() -> None:
    fps = 30.0
    ann_set = PhaseContactAnnotationSet(
        videos=[
            _ann("a.mp4", fps=fps, contact=28, conf=0.9),
            _ann("b.mp4", fps=fps, contact=28, conf=0.4),
            _ann("c.mp4", fps=60.0, contact=40, conf=0.85),
        ]
    )
    preds = {
        "a.mp4": (
            _phases(contact=28, video="a.mp4"),
            ContactEvent(
                CONTACT_TYPE_KINEMATIC,
                28,
                28 / fps,
                0.8,
                kinematic_frame_index=28,
                kinematic_timestamp=28 / fps,
            ),
        ),
        "b.mp4": (
            _phases(contact=30, video="b.mp4"),  # +2 frames
            ContactEvent(
                CONTACT_TYPE_TRACKED,
                30,
                30 / fps,
                0.9,
                kinematic_frame_index=28,
                kinematic_timestamp=28 / fps,
            ),
        ),
        "c.mp4": (
            _phases(
                prep_end=10,
                back_end=18,
                accel_start=19,
                contact=42,
                follow_end=50,
                fps=60.0,
                video="c.mp4",
            ),
            ContactEvent(
                CONTACT_TYPE_TRACKED,
                42,
                42 / 60.0,
                0.9,
                kinematic_frame_index=40,
                kinematic_timestamp=40 / 60.0,
            ),  # +2 frames at 60fps = 33.33ms
        ),
    }
    # Fix GT for c to match annotation contact=40
    report = validate_phase_contact(ann_set, preds)
    assert report.overall_contact is not None
    # Frame errors: 0, 2, 2 → MAE = 4/3
    assert report.overall_contact.mae_frames == pytest.approx(4 / 3)
    assert report.overall_contact.median_frames == pytest.approx(2.0)
    assert report.overall_contact.within_frame_tol["±1"] == pytest.approx(1 / 3)
    assert report.overall_contact.within_frame_tol["±2"] == pytest.approx(1.0)

    # Time: 0, 66.67ms, 33.33ms
    assert report.overall_contact.within_time_tol_ms["±33ms"] == pytest.approx(
        1 / 3, abs=1e-6
    )
    assert report.overall_contact.within_time_tol_ms["±50ms"] == pytest.approx(
        2 / 3, abs=1e-6
    )
    assert report.overall_contact.within_time_tol_ms["±100ms"] == pytest.approx(1.0)

    by_type = {g.key: g for g in report.by_contact_type}
    assert CONTACT_TYPE_KINEMATIC in by_type
    assert CONTACT_TYPE_TRACKED in by_type
    assert by_type[CONTACT_TYPE_KINEMATIC].contact.mae_frames == pytest.approx(0.0)
    assert by_type[CONTACT_TYPE_TRACKED].contact.mae_frames == pytest.approx(2.0)

    by_fps = {g.key: g for g in report.by_fps}
    assert "fps_30" in by_fps
    assert "fps_60" in by_fps
    assert by_fps["fps_60"].contact.mae_ms == pytest.approx(1000 / 60 * 2)

    by_conf = {g.key: g for g in report.by_analysis_confidence}
    assert by_conf["low_<0.5"].sample_count == 1
    assert by_conf["high_>=0.75"].sample_count == 2


def test_missing_contact_rate_when_prediction_absent() -> None:
    ann_set = PhaseContactAnnotationSet(
        videos=[_ann("missing.mp4"), _ann("ok.mp4")]
    )
    phases = _phases(video="ok.mp4")
    contact = ContactEvent(
        CONTACT_TYPE_KINEMATIC,
        28,
        28 / 30.0,
        0.8,
        kinematic_frame_index=28,
        kinematic_timestamp=28 / 30.0,
    )
    report = validate_phase_contact(
        ann_set, {"ok.mp4": (phases, contact)}
    )
    assert report.contact_missing_rate == pytest.approx(0.5)
    missing = next(v for v in report.videos if v.video == "missing.mp4")
    assert missing.contact_predicted is False
    assert all(not b.predicted for b in missing.boundary_errors)


def test_export_and_debug_timeline(tmp_path: Path) -> None:
    ann_set = PhaseContactAnnotationSet(videos=[_ann("clip.mp4")])
    phases = _phases(contact=32, video="clip.mp4")
    contact = ContactEvent(
        CONTACT_TYPE_TRACKED,
        32,
        32 / 30.0,
        0.9,
        kinematic_frame_index=28,
        kinematic_timestamp=28 / 30.0,
    )
    report = validate_phase_contact(ann_set, {"clip.mp4": (phases, contact)})
    out = tmp_path / "phase_contact_validation.json"
    export_phase_contact_validation_report(report, out)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert (
        data["phase_contact_validation_report_version"]
        == PHASE_CONTACT_VALIDATION_REPORT_VERSION
    )
    assert data["video_count"] == 1
    assert data["overall_contact"]["mae_frames"] == pytest.approx(4.0)
    assert "by_contact_type" in data
    assert "by_fps" in data
    assert "by_analysis_confidence" in data

    paths = render_worst_timing_timelines(report, tmp_path / "dbg", max_videos=1)
    assert paths and paths[0].is_file()
    assert report.debug_image_paths


def test_load_example_phase_contact_annotations() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "pose_validation"
        / "example_phase_contact_annotations.json"
    )
    loaded = load_phase_contact_annotation_set(path)
    assert len(loaded.videos) == 1
    assert "CONTACT" in loaded.videos[0].boundaries
    assert loaded.videos[0].boundaries["CONTACT"].frame_index == 28


def test_timestamp_derived_from_fps_when_omitted() -> None:
    ann = VideoPhaseContactAnnotation(
        video="v.mp4",
        fps=25.0,
        boundaries={"CONTACT": BoundaryAnnotation(frame_index=25)},
    )
    assert ann.resolved_timestamp("CONTACT") == pytest.approx(1.0)
