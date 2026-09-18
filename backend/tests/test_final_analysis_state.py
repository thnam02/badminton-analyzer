"""FinalAnalysisState invariants and downstream consumer consistency."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from app.ai.coaching import generate_coaching_report_from_final
from app.processing.keyframes import extract_keyframes_from_final
from app.processing.phases import detect_smash_phases
from app.processing.stroke_metrics import compute_stroke_metrics_from_final
from app.processing.technique import evaluate_technique_from_final
from app.processing.video_quality import assess_video_quality
from app.schemas.contact import CONTACT_TYPE_KINEMATIC, CONTACT_TYPE_TRACKED, ContactEvent
from app.schemas.final_analysis import (
    FinalAnalysisState,
    FinalAnalysisStateError,
    IntermediateAnalysisSnapshot,
    build_final_analysis_state,
    validate_final_analysis_state,
)
from app.schemas.phases import PhaseSegment, PhaseSequence, SmashPhase
from app.services.dataset_exporter import DatasetExporter
from app.services.evidence_packager import EvidencePackager
from app.services.pose_service import PoseService
from tests.test_analyze_orchestration import (
    _kinematics_with_tracked_shift,
    _write_blank_video,
)
from tests.test_phases import _synthetic_smash


def _valid_state(
    tmp_path: Path,
    *,
    contact_frame: int = 28,
    contact_type: str = CONTACT_TYPE_KINEMATIC,
) -> FinalAnalysisState:
    pose, angles, motion, _ = _synthetic_smash(n=40, contact=contact_frame)
    phases = detect_smash_phases(
        pose,
        angles,
        motion,
        forced_contact_frame_index=contact_frame,
    )
    quality = assess_video_quality(
        pose, smoothed_pose=pose, fps=20.0, width=64, height=48
    )
    video = _write_blank_video(tmp_path / "clip.mp4", n=40)
    contact = ContactEvent(
        contact_type=contact_type,
        frame_index=contact_frame,
        timestamp=contact_frame * 0.05,
        confidence=0.9,
        kinematic_frame_index=contact_frame,
        kinematic_timestamp=contact_frame * 0.05,
    )
    return build_final_analysis_state(
        smoothed_pose=pose,
        angles=angles,
        motion=motion,
        video_quality=quality,
        phases=phases,
        contact=contact,
        video_fps=20.0,
        video_width=64,
        video_height=48,
        input_path=video,
        output_path=tmp_path / "out_pose.mp4",
        intermediate=IntermediateAnalysisSnapshot(
            raw_pose=pose,
            initial_phases=phases,
            kinematic_contact=contact,
        ),
    )


def test_build_final_analysis_state_is_frozen(tmp_path: Path) -> None:
    state = _valid_state(tmp_path)
    with pytest.raises(Exception):
        state.contact = ContactEvent(  # type: ignore[misc]
            contact_type=CONTACT_TYPE_TRACKED,
            frame_index=31,
            timestamp=1.55,
            confidence=1.0,
        )


def test_contact_outside_frame_range_rejected(tmp_path: Path) -> None:
    pose, angles, motion, _ = _synthetic_smash(n=40, contact=28)
    phases = detect_smash_phases(
        pose, angles, motion, forced_contact_frame_index=28
    )
    quality = assess_video_quality(
        pose, smoothed_pose=pose, fps=20.0, width=64, height=48
    )
    video = _write_blank_video(tmp_path / "clip.mp4", n=40)
    bad_contact = ContactEvent(
        contact_type=CONTACT_TYPE_TRACKED,
        frame_index=999,
        timestamp=49.95,
        confidence=1.0,
        kinematic_frame_index=28,
        kinematic_timestamp=1.4,
    )
    with pytest.raises(FinalAnalysisStateError, match="outside analyzed frame range"):
        build_final_analysis_state(
            smoothed_pose=pose,
            angles=angles,
            motion=motion,
            video_quality=quality,
            phases=phases,
            contact=bad_contact,
            video_fps=20.0,
            video_width=64,
            video_height=48,
            input_path=video,
            output_path=tmp_path / "out.mp4",
        )


def test_phase_contact_mismatch_rejected(tmp_path: Path) -> None:
    pose, angles, motion, _ = _synthetic_smash(n=40, contact=28)
    phases = detect_smash_phases(
        pose, angles, motion, forced_contact_frame_index=28
    )
    quality = assess_video_quality(
        pose, smoothed_pose=pose, fps=20.0, width=64, height=48
    )
    video = _write_blank_video(tmp_path / "clip.mp4", n=40)
    # Contact claims 31 but phases still anchored at 28.
    bad_contact = ContactEvent(
        contact_type=CONTACT_TYPE_TRACKED,
        frame_index=31,
        timestamp=1.55,
        confidence=1.0,
        kinematic_frame_index=28,
        kinematic_timestamp=1.4,
    )
    with pytest.raises(FinalAnalysisStateError, match="disagrees with final phase"):
        build_final_analysis_state(
            smoothed_pose=pose,
            angles=angles,
            motion=motion,
            video_quality=quality,
            phases=phases,
            contact=bad_contact,
            video_fps=20.0,
            video_width=64,
            video_height=48,
            input_path=video,
            output_path=tmp_path / "out.mp4",
        )


def test_non_monotonic_phases_rejected(tmp_path: Path) -> None:
    state = _valid_state(tmp_path)
    bad_segments = [
        PhaseSegment(
            phase=SmashPhase.PREPARATION,
            start_frame_index=10,
            end_frame_index=20,
            start_timestamp=0.5,
            end_timestamp=1.0,
            confidence=1.0,
        ),
        PhaseSegment(
            phase=SmashPhase.BACKSWING,
            start_frame_index=5,
            end_frame_index=8,
            start_timestamp=0.25,
            end_timestamp=0.4,
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
        validate_final_analysis_state(
            FinalAnalysisState(
                smoothed_pose=state.smoothed_pose,
                angles=state.angles,
                motion=state.motion,
                video_quality=state.video_quality,
                phases=bad_phases,
                contact=state.contact,
                video=state.video,
                video_fps=state.video_fps,
                video_width=state.video_width,
                video_height=state.video_height,
                input_path=state.input_path,
                output_path=state.output_path,
                intermediate=state.intermediate,
            )
        )


def test_downstream_consumers_share_final_contact_and_phases(
    tmp_path: Path,
) -> None:
    state = _valid_state(tmp_path, contact_frame=28)
    contact_f = state.contact.frame_index

    metrics = compute_stroke_metrics_from_final(state)
    technique = evaluate_technique_from_final(state, metrics)
    keyframes = extract_keyframes_from_final(
        state, tmp_path / "keyframes", include_contact_neighbors=True
    )
    evidence = EvidencePackager().package_from_final(
        state, metrics=metrics, technique=technique, keyframes=keyframes
    )
    coaching = generate_coaching_report_from_final(state, evidence)
    dataset_path, _template, export = DatasetExporter().export_from_final(
        state,
        metrics=metrics,
        technique=technique,
        keyframes=keyframes,
    )

    assert metrics.estimated_contact_frame_index == contact_f
    assert evidence.contact.frame_index == contact_f
    assert evidence.contact.contact_type == state.contact.contact_type
    assert evidence.metrics["estimated_contact_frame_index"] == contact_f
    assert any(kf.frame_index == contact_f for kf in keyframes.keyframes)
    assert state.phases.estimated_contact_frame_index == contact_f
    assert state.phases.phase_at(contact_f) is SmashPhase.ESTIMATED_CONTACT
    assert coaching.evidence_version == evidence.evidence_version
    assert export.contact_event["frame_index"] == contact_f
    assert export.phases["estimated_contact_frame_index"] == contact_f
    assert export.pose_metrics["estimated_contact_frame_index"] == contact_f
    assert dataset_path.is_file()
    assert technique.video == metrics.video


def test_finalize_exposes_canonical_final_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kinematic, tracked = 28, 31
    kin, shuttle, racket = _kinematics_with_tracked_shift(
        tmp_path, kinematic=kinematic, tracked=tracked
    )
    monkeypatch.setattr("app.services.pose_service.settings.mesh_enabled", False)

    result = PoseService().finalize_analysis(
        kin, shuttle=shuttle, racket=racket, mesh_overlay=False
    )
    state = result.final_state

    assert isinstance(state, FinalAnalysisState)
    assert state.contact.frame_index == tracked
    assert state.phases.estimated_contact_frame_index == tracked
    assert result.stroke_metrics.estimated_contact_frame_index == tracked
    assert result.evidence_package.contact.frame_index == tracked
    assert result.keyframe_set.keyframes
    assert any(kf.frame_index == tracked for kf in result.keyframe_set.keyframes)
    # Intermediate (pre-resolution) remains available for debug, distinct from final.
    assert state.intermediate is not None
    assert state.intermediate.kinematic_contact.frame_index == kinematic
    assert state.intermediate.initial_phases.estimated_contact_frame_index == kinematic
    assert state.contact.frame_index != state.intermediate.kinematic_contact.frame_index


def test_stale_metrics_rejected_by_technique_from_final(tmp_path: Path) -> None:
    state = _valid_state(tmp_path, contact_frame=28)
    metrics = compute_stroke_metrics_from_final(state)
    stale = replace(metrics, estimated_contact_frame_index=31)
    with pytest.raises(ValueError, match="must match FinalAnalysisState"):
        evaluate_technique_from_final(state, stale)


def test_placeholder_contact_skips_timeline_invariant(tmp_path: Path) -> None:
    pose, angles, motion, _ = _synthetic_smash(n=40, contact=28)
    phases = detect_smash_phases(pose, angles, motion)
    # Clear phase contact to simulate no smash detected.
    empty_phases = PhaseSequence(
        video=phases.video,
        segments=list(phases.segments),
        frame_phases=dict(phases.frame_phases),
        estimated_contact_frame_index=None,
        estimated_contact_timestamp=None,
        confidence=0.0,
        notes=phases.notes,
    )
    quality = assess_video_quality(
        pose, smoothed_pose=pose, fps=20.0, width=64, height=48
    )
    video = _write_blank_video(tmp_path / "clip.mp4", n=40)
    placeholder = ContactEvent(
        contact_type=CONTACT_TYPE_KINEMATIC,
        frame_index=0,
        timestamp=0.0,
        confidence=0.0,
        notes="No kinematic contact available.",
    )
    state = build_final_analysis_state(
        smoothed_pose=pose,
        angles=angles,
        motion=motion,
        video_quality=quality,
        phases=empty_phases,
        contact=placeholder,
        video_fps=20.0,
        video_width=64,
        video_height=48,
        input_path=video,
        output_path=tmp_path / "out.mp4",
    )
    assert state.contact.frame_index == 0
    assert state.phases.estimated_contact_frame_index is None
