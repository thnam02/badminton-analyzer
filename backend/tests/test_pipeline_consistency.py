"""End-to-end consistency: one FinalAnalysisState + one AnalysisSnapshot.

Covers tracked vs kinematic, track failures, coaching isolation, stale
artifacts, and invalid final-state rejection — without changing biomechanics,
thresholds, OpenAI prompts, trackers, or WHAM.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from app.processing.phases import detect_smash_phases
from app.schemas.contact import CONTACT_TYPE_KINEMATIC, CONTACT_TYPE_TRACKED
from app.schemas.coaching import CoachingStatus
from app.schemas.final_analysis import (
    FinalAnalysisState,
    FinalAnalysisStateError,
    build_final_analysis_state,
)
from app.schemas.phases import PhaseSegment, PhaseSequence, SmashPhase
from app.schemas.provenance import ProvenanceError, validate_artifact_provenance
from app.schemas.racket import RacketTrajectory
from app.services.dataset_exporter import DatasetExporter
from app.services.pose_service import FinalizedAnalysis, PoseService
from app.services.video_service import _artifact_base_stem
from tests.test_analyze_orchestration import _kinematics_with_tracked_shift
from tests.test_contact_resolver import _shuttle_traj
from tests.test_final_analysis_state import _valid_state


# ---------------------------------------------------------------------------
# Shared assertions
# ---------------------------------------------------------------------------


_FINAL_ARTIFACT_ATTRS: tuple[str, ...] = (
    "phases_json_path",
    "metrics_json_path",
    "technique_json_path",
    "keyframes_json_path",
    "evidence_json_path",
    "coaching_json_path",
    "contact_json_path",
    "overlay_meta_json_path",
    "quality_json_path",
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_one_canonical_final(
    result: FinalizedAnalysis,
    *,
    expected_contact_frame: int,
    expected_contact_type: str | None = None,
    kinematic_frame: int | None = None,
) -> None:
    """Exactly one FinalAnalysisState + one AnalysisSnapshot; all finals agree."""
    state = result.final_state
    snapshot = result.snapshot

    assert isinstance(state, FinalAnalysisState)
    assert snapshot is not None
    assert snapshot.analysis_id == _artifact_base_stem(result.output_path)
    assert snapshot.snapshot_id.startswith("snap_")
    assert len(snapshot.fingerprint) == 64

    contact_f = int(state.contact.frame_index)
    assert contact_f == expected_contact_frame
    assert state.phases.estimated_contact_frame_index == contact_f
    assert state.phases.phase_at(contact_f) is SmashPhase.ESTIMATED_CONTACT
    if expected_contact_type is not None:
        assert state.contact.contact_type == expected_contact_type
    if kinematic_frame is not None:
        assert state.contact.kinematic_frame_index == kinematic_frame
        if state.intermediate is not None:
            assert (
                state.intermediate.kinematic_contact.frame_index == kinematic_frame
            )

    # In-memory derived objects share provenance + contact.
    for obj in (
        result.stroke_metrics,
        result.technique_evaluation,
        result.keyframe_set,
        result.evidence_package,
        result.coaching_report,
    ):
        assert obj.analysis_id == snapshot.analysis_id
        assert obj.snapshot_id == snapshot.snapshot_id
        assert obj.fingerprint == snapshot.fingerprint

    assert result.stroke_metrics.estimated_contact_frame_index == contact_f
    assert result.evidence_package.contact.frame_index == contact_f
    assert result.evidence_package.metrics["estimated_contact_frame_index"] == contact_f
    assert any(kf.frame_index == contact_f for kf in result.keyframe_set.keyframes)

    # On-disk final artifacts (plus snapshot + overlay meta).
    assert result.snapshot_json_path.is_file()
    snap_disk = _load(result.snapshot_json_path)
    assert snap_disk["snapshot_id"] == snapshot.snapshot_id
    assert snap_disk["fingerprint"] == snapshot.fingerprint
    assert snap_disk["analysis_id"] == snapshot.analysis_id

    for attr in _FINAL_ARTIFACT_ATTRS:
        path = getattr(result, attr)
        assert path.is_file(), f"missing {attr}"
        payload = _load(path)
        assert payload["analysis_id"] == snapshot.analysis_id, attr
        assert payload["snapshot_id"] == snapshot.snapshot_id, attr
        assert payload["fingerprint"] == snapshot.fingerprint, attr
        assert payload.get("artifact_role") == "final", attr

    contact_disk = _load(result.contact_json_path)
    phases_disk = _load(result.phases_json_path)
    metrics_disk = _load(result.metrics_json_path)
    evidence_disk = _load(result.evidence_json_path)
    overlay_disk = _load(result.overlay_meta_json_path)
    assert contact_disk["frame_index"] == contact_f
    assert phases_disk["estimated_contact_frame_index"] == contact_f
    assert metrics_disk["estimated_contact_frame_index"] == contact_f
    assert evidence_disk["contact"]["frame_index"] == contact_f
    assert overlay_disk["contact_frame_index"] == contact_f

    # Intermediate raw pose is stamped but clearly not final.
    raw = _load(result.raw_json_path)
    assert raw["snapshot_id"] == snapshot.snapshot_id
    assert raw["artifact_role"] == "intermediate"

    # Dataset export from the same active snapshot.
    dataset_path, _template, export = DatasetExporter().export_from_final(
        state,
        metrics=result.stroke_metrics,
        technique=result.technique_evaluation,
        keyframes=result.keyframe_set,
        snapshot=snapshot,
        phases_json_path=result.phases_json_path,
        metrics_json_path=result.metrics_json_path,
        contact_json_path=result.contact_json_path,
        technique_json_path=result.technique_json_path,
        keyframes_json_path=result.keyframes_json_path,
        evidence_json_path=result.evidence_json_path,
    )
    assert dataset_path.is_file()
    assert export.snapshot_id == snapshot.snapshot_id
    assert export.fingerprint == snapshot.fingerprint
    assert export.analysis_id == snapshot.analysis_id
    assert export.contact_event["frame_index"] == contact_f
    dataset_disk = _load(dataset_path)
    assert dataset_disk["snapshot_id"] == snapshot.snapshot_id
    assert dataset_disk["fingerprint"] == snapshot.fingerprint
    assert dataset_disk["contact_event"]["frame_index"] == contact_f


def _finalize(
    kin,
    *,
    shuttle=None,
    racket=None,
    monkeypatch: pytest.MonkeyPatch,
) -> FinalizedAnalysis:
    monkeypatch.setattr("app.services.pose_service.settings.mesh_enabled", False)
    return PoseService().finalize_analysis(
        kin,
        shuttle=shuttle,
        racket=racket,
        mesh_overlay=False,
    )


def _failed_racket(n: int = 40) -> RacketTrajectory:
    """Racket trajectory with no usable frames (tracking failure)."""
    return RacketTrajectory(
        video="smash.mp4",
        backend="failed",
        fps=20.0,
        width=1280,
        height=720,
        hitting_hand="RIGHT",
        frames=[],
        missing_frame_indices=list(range(n)),
    )


# ---------------------------------------------------------------------------
# (1) Tracked contact differs from kinematic
# ---------------------------------------------------------------------------


def test_e2e_tracked_contact_differs_from_kinematic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kinematic, tracked = 28, 31
    kin, shuttle, racket = _kinematics_with_tracked_shift(
        tmp_path, kinematic=kinematic, tracked=tracked
    )
    result = _finalize(kin, shuttle=shuttle, racket=racket, monkeypatch=monkeypatch)

    assert result.final_state.contact.contact_type == CONTACT_TYPE_TRACKED
    assert result.final_state.contact.frame_index != kinematic
    assert (
        result.final_state.intermediate is not None
        and result.final_state.intermediate.initial_phases.estimated_contact_frame_index
        == kinematic
    )
    _assert_one_canonical_final(
        result,
        expected_contact_frame=tracked,
        expected_contact_type=CONTACT_TYPE_TRACKED,
        kinematic_frame=kinematic,
    )


# ---------------------------------------------------------------------------
# (2) Shuttle unavailable → kinematic fallback
# ---------------------------------------------------------------------------


def test_e2e_shuttle_unavailable_falls_back_to_kinematic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kinematic = 28
    kin, _shuttle, _racket = _kinematics_with_tracked_shift(tmp_path, kinematic=kinematic)
    # Restore kinematic smash pose so resolver/metrics stay on kinematic peak.
    from tests.test_phases import _synthetic_smash

    pose, angles, motion, _ = _synthetic_smash(n=40, contact=kinematic)
    kin.smoothed_sequence = pose
    kin.angle_sequence = angles
    kin.motion_sequence = motion
    kin.initial_phases = detect_smash_phases(pose, angles, motion)
    from app.services.pose_service import _kinematic_contact_event

    kin.kinematic_contact = _kinematic_contact_event(kin.initial_phases)

    result = _finalize(kin, shuttle=None, racket=None, monkeypatch=monkeypatch)
    assert result.final_state.contact.contact_type == CONTACT_TYPE_KINEMATIC
    _assert_one_canonical_final(
        result,
        expected_contact_frame=kinematic,
        expected_contact_type=CONTACT_TYPE_KINEMATIC,
        kinematic_frame=kinematic,
    )


# ---------------------------------------------------------------------------
# (3) Racket fails while shuttle succeeds — no partial corruption
# ---------------------------------------------------------------------------


def test_e2e_racket_fails_shuttle_succeeds_keeps_single_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kinematic, tracked = 28, 31
    kin, shuttle, _good_racket = _kinematics_with_tracked_shift(
        tmp_path, kinematic=kinematic, tracked=tracked
    )
    tip = (0.61, 0.41)
    shuttle = _shuttle_traj(contact=tracked, n=40, tip_at_contact=tip)
    bad_racket = _failed_racket(n=40)

    result = _finalize(
        kin, shuttle=shuttle, racket=bad_racket, monkeypatch=monkeypatch
    )
    contact_f = result.final_state.contact.frame_index
    # Resolver may track via shuttle signals or fall back; either way one snapshot.
    assert contact_f in {kinematic, tracked}
    _assert_one_canonical_final(
        result,
        expected_contact_frame=contact_f,
        kinematic_frame=kinematic,
    )
    # Failure isolation: intermediate kinematic still intact; no dual contacts.
    assert result.final_state.intermediate is not None
    assert (
        result.final_state.intermediate.kinematic_contact.frame_index == kinematic
    )
    assert result.final_state.contact.frame_index == contact_f


# ---------------------------------------------------------------------------
# (4) Coaching disabled / fails — finalized analysis intact
# ---------------------------------------------------------------------------


def test_e2e_coaching_disabled_still_consistent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kinematic, tracked = 28, 31
    kin, shuttle, racket = _kinematics_with_tracked_shift(
        tmp_path, kinematic=kinematic, tracked=tracked
    )
    monkeypatch.setattr(
        "app.ai.coaching.settings.openai_coaching_enabled", False
    )
    result = _finalize(kin, shuttle=shuttle, racket=racket, monkeypatch=monkeypatch)

    assert result.coaching_report.status == CoachingStatus.SKIPPED.value
    _assert_one_canonical_final(
        result,
        expected_contact_frame=tracked,
        expected_contact_type=CONTACT_TYPE_TRACKED,
        kinematic_frame=kinematic,
    )
    coaching_disk = _load(result.coaching_json_path)
    assert coaching_disk["status"] == CoachingStatus.SKIPPED.value
    assert coaching_disk["snapshot_id"] == result.snapshot.snapshot_id


def test_e2e_coaching_generation_failure_falls_back_without_corrupting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kinematic, tracked = 28, 31
    kin, shuttle, racket = _kinematics_with_tracked_shift(
        tmp_path, kinematic=kinematic, tracked=tracked
    )
    monkeypatch.setattr("app.ai.coaching.settings.openai_coaching_enabled", True)
    monkeypatch.setattr("app.ai.coaching.settings.openai_api_key", "test-key")

    def _boom(*_a, **_k):  # noqa: ANN001
        raise RuntimeError("simulated OpenAI outage")

    monkeypatch.setattr("app.ai.coaching._call_responses_api", _boom)
    result = _finalize(kin, shuttle=shuttle, racket=racket, monkeypatch=monkeypatch)

    assert result.coaching_report.status == CoachingStatus.FALLBACK.value
    # Evidence / metrics / contact still form one canonical final.
    _assert_one_canonical_final(
        result,
        expected_contact_frame=tracked,
        expected_contact_type=CONTACT_TYPE_TRACKED,
        kinematic_frame=kinematic,
    )
    assert result.evidence_package.snapshot_id == result.snapshot.snapshot_id
    assert result.coaching_report.fingerprint == result.snapshot.fingerprint


# ---------------------------------------------------------------------------
# (5) Stale artifact from older snapshot — reject / regenerate
# ---------------------------------------------------------------------------


def test_e2e_stale_snapshot_artifacts_rejected_and_regenerated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kinematic, tracked = 28, 31
    kin, shuttle, racket = _kinematics_with_tracked_shift(
        tmp_path, kinematic=kinematic, tracked=tracked
    )

    first = _finalize(kin, shuttle=None, racket=None, monkeypatch=monkeypatch)
    assert first.final_state.contact.frame_index == kinematic
    stale_metrics = _load(first.metrics_json_path)
    stale_snapshot_id = first.snapshot.snapshot_id
    stale_fingerprint = first.snapshot.fingerprint
    # Preserve a copy of the stale blob (simulates leftover on disk).
    stale_copy = tmp_path / "stale_metrics.json"
    stale_copy.write_text(json.dumps(stale_metrics), encoding="utf-8")

    second = _finalize(kin, shuttle=shuttle, racket=racket, monkeypatch=monkeypatch)
    assert second.final_state.contact.frame_index == tracked
    assert second.snapshot.snapshot_id != stale_snapshot_id
    assert second.snapshot.fingerprint != stale_fingerprint

    # Regenerated on-disk metrics belong only to the new snapshot.
    fresh = _load(second.metrics_json_path)
    assert fresh["snapshot_id"] == second.snapshot.snapshot_id
    assert fresh["estimated_contact_frame_index"] == tracked

    # Stale blob is detected, not silently accepted.
    with pytest.raises(ProvenanceError, match="Stale artifact"):
        validate_artifact_provenance(stale_metrics, second.snapshot)
    with pytest.raises(ProvenanceError, match="Stale artifact"):
        validate_artifact_provenance(_load(stale_copy), second.snapshot)

    # Downstream export rejects mixing stale metrics with the new snapshot.
    with pytest.raises(ProvenanceError, match="Stale artifact"):
        DatasetExporter().export_from_final(
            second.final_state,
            metrics=first.stroke_metrics,
            technique=second.technique_evaluation,
            keyframes=second.keyframe_set,
            snapshot=second.snapshot,
        )

    _assert_one_canonical_final(
        second,
        expected_contact_frame=tracked,
        expected_contact_type=CONTACT_TYPE_TRACKED,
        kinematic_frame=kinematic,
    )


# ---------------------------------------------------------------------------
# (6) Invalid phase / contact ordering — fail before downstream writes
# ---------------------------------------------------------------------------


def test_e2e_invalid_phase_contact_mismatch_fails_before_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kinematic, tracked = 28, 31
    kin, shuttle, racket = _kinematics_with_tracked_shift(
        tmp_path, kinematic=kinematic, tracked=tracked
    )
    monkeypatch.setattr("app.services.pose_service.settings.mesh_enabled", False)

    # Ignore forced contact so phases stay on kinematic while resolver tracks.
    real_detect = detect_smash_phases

    def _ignore_forced(pose, angles, motion, forced_contact_frame_index=None):
        del forced_contact_frame_index
        return real_detect(pose, angles, motion)

    monkeypatch.setattr(
        "app.processing.strokes.smash.analyzer.detect_smash_phases", _ignore_forced
    )

    output = kin.output_path
    phases_path = output.with_name(f"{output.stem}_phases.json")
    metrics_path = output.with_name(f"{output.stem}_stroke_metrics.json")
    evidence_path = output.with_name(f"{output.stem}_evidence.json")
    contact_path = output.with_name(
        f"{_artifact_base_stem(output)}_contact.json"
    )

    with pytest.raises(FinalAnalysisStateError, match="disagrees with final phase"):
        PoseService().finalize_analysis(
            kin, shuttle=shuttle, racket=racket, mesh_overlay=False
        )

    # No downstream final artifacts written after the failed final-state build.
    assert not phases_path.exists()
    assert not metrics_path.exists()
    assert not evidence_path.exists()
    assert not contact_path.exists()


def test_e2e_non_monotonic_phases_rejected_before_downstream(
    tmp_path: Path,
) -> None:
    state = _valid_state(tmp_path, contact_frame=28)
    bad_segments = [
        PhaseSegment(
            phase=SmashPhase.ACCELERATION,
            start_frame_index=20,
            end_frame_index=27,
            start_timestamp=1.0,
            end_timestamp=1.35,
            confidence=1.0,
        ),
        PhaseSegment(
            phase=SmashPhase.PREPARATION,
            start_frame_index=0,
            end_frame_index=10,
            start_timestamp=0.0,
            end_timestamp=0.5,
            confidence=1.0,
        ),
    ]
    bad_phases = PhaseSequence(
        video=state.phases.video,
        segments=bad_segments,
        frame_phases=dict(state.phases.frame_phases),
        estimated_contact_frame_index=state.phases.estimated_contact_frame_index,
        estimated_contact_timestamp=state.phases.estimated_contact_timestamp,
        confidence=state.phases.confidence,
    )
    with pytest.raises(FinalAnalysisStateError, match="not monotonic"):
        build_final_analysis_state(
            smoothed_pose=state.smoothed_pose,
            angles=state.angles,
            motion=state.motion,
            video_quality=state.video_quality,
            phases=bad_phases,
            contact=state.contact,
            video_fps=state.video_fps,
            video_width=state.video_width,
            video_height=state.video_height,
            input_path=state.input_path,
            output_path=state.output_path,
        )
