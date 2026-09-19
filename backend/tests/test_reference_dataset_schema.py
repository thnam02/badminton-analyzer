"""Schema validation tests for the reference-data protocol and dataset metadata."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.schemas.reference_dataset import (
    ACCEPTANCE_MIN_POSE_COVERAGE,
    CAMERA_VIEW_REAR_45,
    CAMERA_VIEW_SIDE_45,
    PROTOCOL_MIN_FPS,
    PROTOCOL_MIN_HEIGHT,
    PROTOCOL_MIN_WIDTH,
    PROTOCOL_STROKE_TYPE,
    REFERENCE_DATA_PROTOCOL_VERSION,
    REFERENCE_DATASET_SCHEMA_VERSION,
    AnalysisArtifactRefs,
    CameraSetup,
    Player,
    RecordingProtocolSpec,
    RecordingSession,
    ReferenceDatasetManifest,
    ReferenceLabel,
    StrokeAcceptanceSignals,
    StrokeSample,
    default_recording_protocol,
    evaluate_stroke_acceptance,
    load_reference_dataset_manifest,
    normalize_skill_group,
)


def test_default_protocol_matches_collection_targets() -> None:
    spec = default_recording_protocol()
    assert spec.protocol_version == REFERENCE_DATA_PROTOCOL_VERSION
    assert spec.stroke_type == PROTOCOL_STROKE_TYPE
    assert set(spec.allowed_camera_views) == {
        CAMERA_VIEW_REAR_45,
        CAMERA_VIEW_SIDE_45,
    }
    assert spec.min_fps == PROTOCOL_MIN_FPS
    assert spec.min_width == PROTOCOL_MIN_WIDTH
    assert spec.min_height == PROTOCOL_MIN_HEIGHT
    assert spec.players_per_skill_group_min == 5
    assert spec.players_per_skill_group_max == 10
    assert spec.valid_smashes_per_player_min == 15
    assert spec.valid_smashes_per_player_max == 30
    payload = spec.to_dict()
    assert payload["collection_targets"]["valid_smashes_per_player"]["min"] == 15


def test_player_normalizes_skill_and_handedness() -> None:
    player = Player(
        anonymized_player_id="p1",
        skill_group="Advanced",
        handedness="right",
    )
    assert player.skill_group == "advanced_expert"
    assert player.handedness == "RIGHT"
    assert normalize_skill_group("expert") == "advanced_expert"
    assert normalize_skill_group("advanced/expert") == "advanced_expert"


def test_invalid_vocab_rejected() -> None:
    with pytest.raises(ValueError, match="skill_group"):
        Player(anonymized_player_id="p1", skill_group="pro", handedness="RIGHT")
    with pytest.raises(ValueError, match="handedness"):
        Player(anonymized_player_id="p1", skill_group="beginner", handedness="BOTH")
    with pytest.raises(ValueError, match="camera_view"):
        CameraSetup(camera_view="front", fps=60, width=1920, height=1080)
    with pytest.raises(ValueError, match="take_number"):
        StrokeSample(
            stroke_id="s1",
            session_id="sess",
            anonymized_player_id="p1",
            stroke_type=PROTOCOL_STROKE_TYPE,
            take_number=0,
            camera_view=CAMERA_VIEW_SIDE_45,
            fps=60,
            width=1920,
            height=1080,
            handedness="RIGHT",
            skill_group="beginner",
        )


def test_camera_setup_meets_minimum_protocol() -> None:
    ok = CameraSetup(
        camera_view=CAMERA_VIEW_REAR_45,
        fps=30,
        width=1280,
        height=720,
    )
    assert ok.meets_minimum_protocol() is True
    low = CameraSetup(
        camera_view=CAMERA_VIEW_SIDE_45,
        fps=24,
        width=1280,
        height=720,
    )
    assert low.meets_minimum_protocol() is False


def test_stroke_acceptance_pass_and_fail() -> None:
    good = evaluate_stroke_acceptance(
        StrokeAcceptanceSignals(
            full_body_visible=True,
            major_occlusion=False,
            camera_stable=True,
            pose_coverage=0.9,
            contact_available_or_estimable=True,
            fps=60,
            width=1920,
            height=1080,
            camera_view=CAMERA_VIEW_SIDE_45,
        )
    )
    assert good.accepted is True
    assert good.failures == []

    bad = evaluate_stroke_acceptance(
        StrokeAcceptanceSignals(
            full_body_visible=False,
            major_occlusion=True,
            camera_stable=False,
            pose_coverage=ACCEPTANCE_MIN_POSE_COVERAGE - 0.1,
            contact_available_or_estimable=False,
            fps=25,
            width=640,
            height=360,
            camera_view="front",
        )
    )
    assert bad.accepted is False
    assert "full_body_not_visible" in bad.failures
    assert "major_occlusion" in bad.failures
    assert "unstable_camera" in bad.failures
    assert "pose_coverage_below_threshold" in bad.failures
    assert "contact_event_unavailable" in bad.failures
    assert "fps_below_minimum" in bad.failures
    assert "resolution_below_minimum" in bad.failures
    assert "invalid_camera_view" in bad.failures


def test_manifest_round_trip_and_required_fields(tmp_path: Path) -> None:
    player = Player(
        anonymized_player_id="p_anon_001",
        skill_group="intermediate",
        handedness="RIGHT",
    )
    camera = CameraSetup(
        camera_view=CAMERA_VIEW_SIDE_45,
        fps=60,
        width=1920,
        height=1080,
        camera_height_m=1.4,
        distance_to_player_m=6.0,
    )
    session = RecordingSession(
        session_id="sess_1",
        anonymized_player_id=player.anonymized_player_id,
        camera_setup=camera,
        planned_smash_count=25,
        recorded_at="2026-09-01T10:00:00+10:00",
    )
    stroke = StrokeSample(
        stroke_id="stroke_1",
        session_id=session.session_id,
        anonymized_player_id=player.anonymized_player_id,
        stroke_type=PROTOCOL_STROKE_TYPE,
        take_number=3,
        camera_view=CAMERA_VIEW_SIDE_45,
        fps=60,
        width=1920,
        height=1080,
        handedness="RIGHT",
        skill_group="intermediate",
        acceptance_status="accepted",
        artifact_refs=AnalysisArtifactRefs(
            analysis_id="abc123",
            dataset_export_path="outputs/abc123_dataset.json",
        ),
    )
    label = ReferenceLabel(
        label_id="label_1",
        stroke_id=stroke.stroke_id,
        labeler_id="coach_a",
        issue_labels=["LOW_KNEE_CONTRIBUTION"],
        is_valid_reference_stroke=True,
    )
    manifest = ReferenceDatasetManifest(
        players=[player],
        sessions=[session],
        strokes=[stroke],
        labels=[label],
    )
    assert manifest.schema_version == REFERENCE_DATASET_SCHEMA_VERSION
    path = tmp_path / "manifest.json"
    manifest.save_json(path)
    loaded = load_reference_dataset_manifest(path)
    assert loaded.players[0].anonymized_player_id == "p_anon_001"
    assert loaded.sessions[0].camera_setup.camera_view == CAMERA_VIEW_SIDE_45
    assert loaded.strokes[0].take_number == 3
    assert loaded.strokes[0].artifact_refs.analysis_id == "abc123"
    assert loaded.labels[0].issue_labels == ["LOW_KNEE_CONTRIBUTION"]
    assert "reference_ranges" not in json.loads(path.read_text(encoding="utf-8"))


def test_load_example_reference_dataset() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "reference_dataset"
        / "example_reference_dataset.json"
    )
    loaded = load_reference_dataset_manifest(path)
    assert loaded.protocol_version == REFERENCE_DATA_PROTOCOL_VERSION
    assert loaded.protocol.stroke_type == PROTOCOL_STROKE_TYPE
    assert len(loaded.players) == 2
    assert len(loaded.sessions) == 1
    assert len(loaded.strokes) == 2
    accepted = [s for s in loaded.strokes if s.acceptance_status == "accepted"]
    rejected = [s for s in loaded.strokes if s.acceptance_status == "rejected"]
    assert len(accepted) == 1
    assert "major_occlusion" in rejected[0].acceptance_failures
    assert loaded.labels[0].is_valid_reference_stroke is True
    # Ensure protocol targets are documented in the embedded snapshot.
    targets = loaded.protocol.to_dict()["collection_targets"]
    assert targets["players_per_skill_group"]["min"] == 5
    assert targets["players_per_skill_group"]["max"] == 10


def test_recording_protocol_spec_is_frozen_defaults() -> None:
    spec = RecordingProtocolSpec()
    assert spec.preferred_fps == 60.0
    assert spec.repeated_smashes_per_session_preferred == 25
