"""Integration: resolved final contact is the only contact used downstream."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.cv.overlay import AnnotationRenderer
from app.processing.phases import detect_smash_phases
from app.processing.video_quality import assess_video_quality
from app.schemas.contact import CONTACT_TYPE_KINEMATIC, CONTACT_TYPE_TRACKED
from app.schemas.final_analysis import FinalAnalysisState
from app.schemas.phases import SmashPhase
from app.services.dataset_exporter import DatasetExporter
from app.services.pose_service import PoseKinematics, PoseService, _kinematic_contact_event
from tests.test_contact_resolver import _pose_near_racket, _racket_traj, _shuttle_traj
from tests.test_phases import _synthetic_smash


def _write_blank_video(path: Path, n: int = 40, fps: float = 20.0) -> Path:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (64, 48),
    )
    assert writer.isOpened()
    for i in range(n):
        writer.write(np.full((48, 64, 3), i % 255, dtype=np.uint8))
    writer.release()
    return path


def _kinematics_with_tracked_shift(
    tmp_path: Path,
    *,
    kinematic: int = 28,
    tracked: int = 31,
    n: int = 40,
) -> tuple[PoseKinematics, object, object]:
    """Kinematic peak at ``kinematic``; shuttle/racket resolve to ``tracked``."""
    pose, angles, motion, _ = _synthetic_smash(n=n, contact=kinematic)
    assert (
        detect_smash_phases(pose, angles, motion).estimated_contact_frame_index
        == kinematic
    )

    video = _write_blank_video(tmp_path / "clip.mp4", n=n)
    output = tmp_path / "run_pose.mp4"
    quality = assess_video_quality(
        pose,
        smoothed_pose=pose,
        fps=20.0,
        width=64,
        height=48,
    )
    initial_phases = detect_smash_phases(pose, angles, motion)
    tip = (0.61, 0.41)
    # Pose near racket tip so resolver can score proximity at tracked frame.
    pose_for_resolver = _pose_near_racket(n, tracked, tip)
    kin = PoseKinematics(
        input_path=video,
        output_path=output,
        raw_sequence=pose,
        smoothed_sequence=pose_for_resolver,
        quality_report=quality,
        angle_sequence=angles,
        motion_sequence=motion,
        initial_phases=initial_phases,
        kinematic_contact=_kinematic_contact_event(initial_phases),
        video_fps=20.0,
        video_width=64,
        video_height=48,
    )
    shuttle = _shuttle_traj(contact=tracked, n=n, tip_at_contact=tip)
    racket = _racket_traj(contact=tracked, n=n, tip_at_contact=tip)
    return kin, shuttle, racket


def test_finalize_uses_resolved_contact_for_all_downstream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kinematic, tracked = 28, 31
    kin, shuttle, racket = _kinematics_with_tracked_shift(
        tmp_path, kinematic=kinematic, tracked=tracked
    )
    assert kin.kinematic_contact.frame_index == kinematic

    overlay_phases: list[SmashPhase | None] = []
    orig_render = AnnotationRenderer.render

    def _capture_render(self, frame, **kwargs):  # noqa: ANN001
        phase = kwargs.get("phase")
        frame_index = kwargs.get("frame_index")
        if frame_index == tracked and phase is not None:
            overlay_phases.append(phase)
        return orig_render(self, frame, **kwargs)

    monkeypatch.setattr(AnnotationRenderer, "render", _capture_render)
    monkeypatch.setattr("app.services.pose_service.settings.mesh_enabled", False)

    result = PoseService().finalize_analysis(
        kin,
        shuttle=shuttle,
        racket=racket,
        mesh_overlay=False,
    )

    final_state = result.final_state
    assert isinstance(final_state, FinalAnalysisState)
    contact_event = final_state.contact
    phase_sequence = final_state.phases
    stroke_metrics = result.stroke_metrics
    technique_evaluation = result.technique_evaluation
    keyframe_set = result.keyframe_set
    evidence_package = result.evidence_package
    coaching_report = result.coaching_report

    assert contact_event.contact_type == CONTACT_TYPE_TRACKED
    assert contact_event.frame_index == tracked
    assert contact_event.kinematic_frame_index == kinematic

    # Final phase re-snap locked into FinalAnalysisState
    assert phase_sequence.estimated_contact_frame_index == tracked
    assert phase_sequence.phase_at(tracked) is SmashPhase.ESTIMATED_CONTACT
    assert final_state.contact_frame_index == tracked
    assert final_state.phase_contact_frame_index == tracked

    # Stroke metrics / technique use resolved contact frame measurements
    assert stroke_metrics.estimated_contact_frame_index == tracked
    angle_tracked = next(
        f for f in kin.angle_sequence.frames if f.frame_index == tracked
    )
    assert stroke_metrics.contact_elbow_angle_deg == pytest.approx(
        angle_tracked.right_elbow
    )
    assert technique_evaluation.video == stroke_metrics.video

    # Keyframes anchored on final contact (and neighbors)
    assert any(kf.frame_index == tracked for kf in keyframe_set.keyframes)
    neighbor_idxs = {kf.frame_index for kf in keyframe_set.keyframes}
    assert (tracked - 2) in neighbor_idxs or (tracked + 2) in neighbor_idxs

    # Evidence + coaching input
    assert evidence_package.contact.frame_index == tracked
    assert evidence_package.contact.contact_type == CONTACT_TYPE_TRACKED
    assert evidence_package.contact.kinematic_frame_index == kinematic
    assert evidence_package.metrics["estimated_contact_frame_index"] == tracked
    assert coaching_report.evidence_version == evidence_package.evidence_version

    # Overlay render used final phase labels at the resolved contact frame
    assert overlay_phases
    assert all(p is SmashPhase.ESTIMATED_CONTACT for p in overlay_phases)

    # On-disk artifacts agree (single write; no stale kinematic rewrite)
    contact_disk = json.loads(result.contact_json_path.read_text(encoding="utf-8"))
    phases_disk = json.loads(result.phases_json_path.read_text(encoding="utf-8"))
    metrics_disk = json.loads(result.metrics_json_path.read_text(encoding="utf-8"))
    evidence_disk = json.loads(result.evidence_json_path.read_text(encoding="utf-8"))
    keyframes_disk = json.loads(result.keyframes_json_path.read_text(encoding="utf-8"))
    assert contact_disk["frame_index"] == tracked
    assert phases_disk["estimated_contact_frame_index"] == tracked
    assert metrics_disk["estimated_contact_frame_index"] == tracked
    assert evidence_disk["contact"]["frame_index"] == tracked
    assert any(kf["frame_index"] == tracked for kf in keyframes_disk["keyframes"])
    assert result.technique_json_path.is_file()
    assert result.coaching_json_path.is_file()
    assert result.output_path.is_file()

    # Dataset export from the same FinalAnalysisState
    dataset_path, _template, export = DatasetExporter().export_from_final(
        final_state,
        metrics=stroke_metrics,
        technique=technique_evaluation,
        keyframes=keyframe_set,
        snapshot=result.snapshot,
        phases_json_path=result.phases_json_path,
        metrics_json_path=result.metrics_json_path,
        contact_json_path=result.contact_json_path,
        technique_json_path=result.technique_json_path,
        keyframes_json_path=result.keyframes_json_path,
        evidence_json_path=result.evidence_json_path,
    )
    assert dataset_path.is_file()
    assert export.contact_event["frame_index"] == tracked
    assert export.contact_event["contact_type"] == CONTACT_TYPE_TRACKED
    assert export.pose_metrics["estimated_contact_frame_index"] == tracked
    assert export.phases["estimated_contact_frame_index"] == tracked
    assert export.snapshot_id == result.snapshot.snapshot_id
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    assert payload["contact_event"]["frame_index"] == tracked
    assert payload["pose_metrics"]["estimated_contact_frame_index"] == tracked
    assert payload["phases"]["estimated_contact_frame_index"] == tracked
    assert payload["snapshot_id"] == result.snapshot.snapshot_id

    # All final artifacts share the same provenance envelope
    for path in (
        result.phases_json_path,
        result.metrics_json_path,
        result.contact_json_path,
        result.evidence_json_path,
        result.coaching_json_path,
        result.overlay_meta_json_path,
    ):
        disk = json.loads(path.read_text(encoding="utf-8"))
        assert disk["snapshot_id"] == result.snapshot.snapshot_id
        assert disk["fingerprint"] == result.snapshot.fingerprint
        assert disk["analysis_id"] == result.snapshot.analysis_id


def test_finalize_without_tracks_keeps_kinematic_contact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kin, _shuttle, _racket = _kinematics_with_tracked_shift(tmp_path)
    # Restore smoothed pose to the kinematic smash profile for this case.
    pose, angles, motion, _ = _synthetic_smash(n=40, contact=28)
    kin.smoothed_sequence = pose
    kin.angle_sequence = angles
    kin.motion_sequence = motion
    kin.initial_phases = detect_smash_phases(pose, angles, motion)
    kin.kinematic_contact = _kinematic_contact_event(kin.initial_phases)
    kinematic = kin.kinematic_contact.frame_index
    monkeypatch.setattr("app.services.pose_service.settings.mesh_enabled", False)

    result = PoseService().finalize_analysis(
        kin, shuttle=None, racket=None, mesh_overlay=False
    )
    contact_event = result.final_state.contact
    phase_sequence = result.final_state.phases
    stroke_metrics = result.stroke_metrics
    evidence_package = result.evidence_package

    assert contact_event.contact_type == CONTACT_TYPE_KINEMATIC
    assert contact_event.frame_index == kinematic
    assert phase_sequence.estimated_contact_frame_index == kinematic
    assert stroke_metrics.estimated_contact_frame_index == kinematic
    assert evidence_package.contact.frame_index == kinematic


def test_apply_resolved_contact_removed() -> None:
    assert not hasattr(PoseService, "apply_resolved_contact")
