"""Artifact provenance / AnalysisSnapshot consistency tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.ai.coaching import generate_coaching_report_from_final
from app.processing.keyframes import extract_keyframes_from_final
from app.processing.stroke_metrics import compute_stroke_metrics_from_final
from app.processing.technique import evaluate_technique_from_final
from app.schemas.analysis_snapshot import build_analysis_snapshot
from app.schemas.provenance import (
    ARTIFACT_ROLE_INTERMEDIATE,
    ProvenanceError,
    STROKE_METRICS_ARTIFACT_SCHEMA_VERSION,
    apply_provenance,
    validate_artifact_provenance,
)
from app.services.dataset_exporter import DatasetExporter
from app.services.evidence_packager import EvidencePackager
from app.services.pose_service import PoseService
from app.services.video_service import _artifact_base_stem
from tests.test_analyze_orchestration import _kinematics_with_tracked_shift
from tests.test_final_analysis_state import _valid_state


def _stamp_downstream(state, snapshot):
    metrics = compute_stroke_metrics_from_final(state)
    apply_provenance(
        metrics, snapshot, artifact_schema_version=STROKE_METRICS_ARTIFACT_SCHEMA_VERSION
    )
    technique = evaluate_technique_from_final(state, metrics)
    apply_provenance(
        technique, snapshot, artifact_schema_version="1.0.0"
    )
    keyframes = extract_keyframes_from_final(state, state.output_path.parent / "kf")
    apply_provenance(keyframes, snapshot, artifact_schema_version="1.0.0")
    return metrics, technique, keyframes


def test_fingerprint_ignores_paths_and_is_deterministic(tmp_path: Path) -> None:
    state_a = _valid_state(tmp_path / "a", contact_frame=28)
    state_b = _valid_state(tmp_path / "b", contact_frame=28)
    # Different temp paths must not change fingerprint.
    id_a = _artifact_base_stem(state_a.output_path)
    id_b = _artifact_base_stem(state_b.output_path)
    snap_a = build_analysis_snapshot(state_a, analysis_id=id_a)
    snap_b = build_analysis_snapshot(state_b, analysis_id=id_b)
    assert snap_a.fingerprint == snap_b.fingerprint
    assert snap_a.snapshot_id == snap_b.snapshot_id
    assert "input_path" not in snap_a.canonical
    assert "output_path" not in snap_a.canonical
    assert "created_at" not in snap_a.canonical


def test_contact_change_produces_new_snapshot_id(tmp_path: Path) -> None:
    state_28 = _valid_state(tmp_path / "c28", contact_frame=28)
    state_31 = _valid_state(tmp_path / "c31", contact_frame=31)
    snap_28 = build_analysis_snapshot(
        state_28, analysis_id=_artifact_base_stem(state_28.output_path)
    )
    snap_31 = build_analysis_snapshot(
        state_31, analysis_id=_artifact_base_stem(state_31.output_path)
    )
    assert snap_28.fingerprint != snap_31.fingerprint
    assert snap_28.snapshot_id != snap_31.snapshot_id


def test_stale_artifact_rejected_against_new_snapshot(tmp_path: Path) -> None:
    state_old = _valid_state(tmp_path / "old", contact_frame=28)
    state_new = _valid_state(tmp_path / "new", contact_frame=31)
    snap_old = build_analysis_snapshot(
        state_old, analysis_id=_artifact_base_stem(state_old.output_path)
    )
    snap_new = build_analysis_snapshot(
        state_new, analysis_id=_artifact_base_stem(state_new.output_path)
    )
    metrics_old, technique_old, keyframes_old = _stamp_downstream(state_old, snap_old)

    with pytest.raises(ProvenanceError, match="Stale artifact"):
        EvidencePackager().package_from_final(
            state_new,
            metrics=metrics_old,
            technique=technique_old,
            keyframes=keyframes_old,
            snapshot=snap_new,
        )

    with pytest.raises(ProvenanceError, match="Stale artifact"):
        DatasetExporter().export_from_final(
            state_new,
            metrics=metrics_old,
            technique=technique_old,
            keyframes=keyframes_old,
            snapshot=snap_new,
        )

    validate_artifact_provenance(metrics_old.to_dict(), snap_old)
    with pytest.raises(ProvenanceError, match="Stale artifact"):
        validate_artifact_provenance(metrics_old.to_dict(), snap_new)


def test_coaching_rejects_stale_evidence(tmp_path: Path) -> None:
    state_old = _valid_state(tmp_path / "old", contact_frame=28)
    state_new = _valid_state(tmp_path / "new", contact_frame=31)
    snap_old = build_analysis_snapshot(
        state_old, analysis_id=_artifact_base_stem(state_old.output_path)
    )
    snap_new = build_analysis_snapshot(
        state_new, analysis_id=_artifact_base_stem(state_new.output_path)
    )
    metrics, technique, keyframes = _stamp_downstream(state_old, snap_old)
    evidence = EvidencePackager().package_from_final(
        state_old,
        metrics=metrics,
        technique=technique,
        keyframes=keyframes,
        snapshot=snap_old,
    )
    with pytest.raises(ProvenanceError, match="Stale artifact"):
        generate_coaching_report_from_final(
            state_new, evidence, snapshot=snap_new
        )


def test_finalize_re_resolution_regenerates_snapshot_and_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Kinematic-only then tracked contact: new fingerprint; old artifacts rejected."""
    monkeypatch.setattr("app.services.pose_service.settings.mesh_enabled", False)
    kin, shuttle, racket = _kinematics_with_tracked_shift(tmp_path)

    first = PoseService().finalize_analysis(
        kin, shuttle=None, racket=None, mesh_overlay=False
    )
    first_snap = first.snapshot
    first_metrics_disk = json.loads(first.metrics_json_path.read_text(encoding="utf-8"))
    first_contact = first.final_state.contact.frame_index
    assert first_contact == kin.kinematic_contact.frame_index
    assert first_metrics_disk["snapshot_id"] == first_snap.snapshot_id
    assert first_metrics_disk["fingerprint"] == first_snap.fingerprint

    # Raw pose marked intermediate; final phases marked final.
    raw_disk = json.loads(first.raw_json_path.read_text(encoding="utf-8"))
    phases_disk = json.loads(first.phases_json_path.read_text(encoding="utf-8"))
    assert raw_disk["artifact_role"] == ARTIFACT_ROLE_INTERMEDIATE
    assert phases_disk["artifact_role"] == "final"
    assert phases_disk["snapshot_id"] == first_snap.snapshot_id

    second = PoseService().finalize_analysis(
        kin, shuttle=shuttle, racket=racket, mesh_overlay=False
    )
    second_snap = second.snapshot
    assert second.final_state.contact.frame_index != first_contact
    assert second_snap.fingerprint != first_snap.fingerprint
    assert second_snap.snapshot_id != first_snap.snapshot_id

    # Regenerated on-disk artifacts match the new snapshot only.
    second_metrics = json.loads(second.metrics_json_path.read_text(encoding="utf-8"))
    second_evidence = json.loads(second.evidence_json_path.read_text(encoding="utf-8"))
    second_coaching = json.loads(second.coaching_json_path.read_text(encoding="utf-8"))
    second_overlay = json.loads(
        second.overlay_meta_json_path.read_text(encoding="utf-8")
    )
    for payload in (
        second_metrics,
        second_evidence,
        second_coaching,
        second_overlay,
        json.loads(second.phases_json_path.read_text(encoding="utf-8")),
        json.loads(second.contact_json_path.read_text(encoding="utf-8")),
    ):
        assert payload["analysis_id"] == second_snap.analysis_id
        assert payload["snapshot_id"] == second_snap.snapshot_id
        assert payload["fingerprint"] == second_snap.fingerprint

    # Old artifact blob is not accepted for the new snapshot.
    with pytest.raises(ProvenanceError, match="Stale artifact"):
        validate_artifact_provenance(first_metrics_disk, second_snap)

    # Dataset export tied to second snapshot rejects first metrics.
    with pytest.raises(ProvenanceError, match="Stale artifact"):
        DatasetExporter().export_from_final(
            second.final_state,
            metrics=first.stroke_metrics,
            technique=second.technique_evaluation,
            keyframes=second.keyframe_set,
            snapshot=second_snap,
        )

    dataset_path, _, export = DatasetExporter().export_from_final(
        second.final_state,
        metrics=second.stroke_metrics,
        technique=second.technique_evaluation,
        keyframes=second.keyframe_set,
        snapshot=second_snap,
    )
    assert export.snapshot_id == second_snap.snapshot_id
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    assert payload["snapshot_id"] == second_snap.snapshot_id
    assert payload["fingerprint"] == second_snap.fingerprint
    assert second.snapshot_json_path.is_file()
    assert _artifact_base_stem(second.output_path) == second_snap.analysis_id
