"""Multi-stroke registry, clear analyzer, and smash/clear isolation tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from app.ai.coaching import generate_coaching_report_from_final
from app.processing.phases import detect_smash_phases
from app.processing.reference_profile_selector import (
    MATCH_NONE,
    ReferenceProfileSelector,
)
from app.processing.reference_profiles import (
    build_provisional_smash_right_side,
    default_reference_profiles,
)
from app.processing.strokes import get_stroke_analyzer, list_supported_stroke_types
from app.processing.strokes.clear.analyzer import ForehandClearAnalyzer
from app.processing.strokes.clear.evaluator import evaluate_clear_technique
from app.processing.strokes.clear.metrics import compute_clear_metrics
from app.processing.strokes.clear.phases import detect_clear_phases
from app.processing.strokes.clear.reference_profiles import (
    default_clear_reference_profiles,
)
from app.processing.strokes.smash.analyzer import SmashAnalyzer
from app.processing.technique import evaluate_technique
from app.processing.video_quality import assess_video_quality
from app.schemas.analysis_snapshot import build_analysis_snapshot
from app.schemas.clear_metrics import ForehandClearMetrics
from app.schemas.contact import CONTACT_TYPE_KINEMATIC, ContactEvent
from app.schemas.final_analysis import (
    IntermediateAnalysisSnapshot,
    build_final_analysis_state,
)
from app.schemas.keyframes import KeyframeSet
from app.schemas.phases import SmashPhase
from app.schemas.provenance import apply_provenance
from app.schemas.stroke_types import (
    StrokeType,
    UnsupportedStrokeError,
    normalize_stroke_type,
)
from app.schemas.technique_calibration import IssueStatus
from app.services.dataset_exporter import DatasetExporter
from app.services.evidence_packager import EvidencePackager
from tests.test_analyze_orchestration import _write_blank_video
from tests.test_phases import _synthetic_smash
from tests.test_technique import _build_pipeline


def _clear_state(tmp_path: Path, *, contact_frame: int = 28):
    pose, angles, motion, _ = _synthetic_smash(n=40, contact=contact_frame)
    phases = detect_clear_phases(
        pose, angles, motion, forced_contact_frame_index=contact_frame
    )
    quality = assess_video_quality(
        pose, smoothed_pose=pose, fps=20.0, width=64, height=48
    )
    contact = ContactEvent(
        contact_type=CONTACT_TYPE_KINEMATIC,
        frame_index=contact_frame,
        timestamp=contact_frame * 0.05,
        confidence=0.9,
        kinematic_frame_index=contact_frame,
        kinematic_timestamp=contact_frame * 0.05,
    )
    video = _write_blank_video(tmp_path / "clear.mp4", n=40)
    state = build_final_analysis_state(
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
    return state


def _empty_keyframes(video: str) -> KeyframeSet:
    return KeyframeSet(video=video, output_dir="", keyframes=[])


def test_registry_selects_smash_analyzer():
    analyzer = get_stroke_analyzer(StrokeType.FOREHAND_SMASH)
    assert isinstance(analyzer, SmashAnalyzer)
    assert analyzer.stroke_type == StrokeType.FOREHAND_SMASH


def test_registry_selects_clear_analyzer():
    analyzer = get_stroke_analyzer("FOREHAND_CLEAR")
    assert isinstance(analyzer, ForehandClearAnalyzer)
    assert analyzer.stroke_type == StrokeType.FOREHAND_CLEAR


def test_registry_aliases():
    assert normalize_stroke_type("SMASH") == StrokeType.FOREHAND_SMASH
    assert normalize_stroke_type("CLEAR") == StrokeType.FOREHAND_CLEAR


def test_unsupported_stroke_fails_clearly():
    with pytest.raises(UnsupportedStrokeError, match="DROP"):
        get_stroke_analyzer("DROP")
    with pytest.raises(UnsupportedStrokeError):
        normalize_stroke_type("SERVE")


def test_supported_stroke_types_list():
    supported = list_supported_stroke_types()
    assert "FOREHAND_SMASH" in supported
    assert "FOREHAND_CLEAR" in supported


def test_clear_never_loads_smash_reference_profile():
    smash = build_provisional_smash_right_side()
    clear_catalog = default_clear_reference_profiles()
    mixed = [smash, *clear_catalog]

    clear_sel = ReferenceProfileSelector(mixed).select(
        stroke_type="FOREHAND_CLEAR",
        handedness="RIGHT",
        camera_view="SIDE",
    )
    assert clear_sel.has_valid_profile
    assert clear_sel.profile is not None
    assert clear_sel.profile.stroke_type == "FOREHAND_CLEAR"
    assert clear_sel.profile.profile_id != smash.profile_id

    smash_sel = ReferenceProfileSelector(mixed).select(
        stroke_type="FOREHAND_SMASH",
        handedness="RIGHT",
        camera_view="SIDE",
    )
    assert smash_sel.has_valid_profile
    assert smash_sel.profile is not None
    assert smash_sel.profile.stroke_type in {"SMASH", "FOREHAND_SMASH"}
    assert "clear" not in smash_sel.profile.profile_id.lower()


def test_clear_selector_does_not_fall_back_to_smash_only_catalog():
    smash_only = default_reference_profiles()
    selection = ReferenceProfileSelector(smash_only).select(
        stroke_type="FOREHAND_CLEAR",
        handedness="RIGHT",
    )
    assert selection.match_level == MATCH_NONE
    assert selection.profile is None


def test_smash_behavior_unchanged_via_analyzer():
    pose, angles, motion, _phases, direct_metrics = _build_pipeline()
    analyzer = SmashAnalyzer()
    smash_phases = analyzer.detect_phases(pose, angles, motion)
    legacy_phases = detect_smash_phases(pose, angles, motion)
    assert smash_phases.estimated_contact_frame_index == (
        legacy_phases.estimated_contact_frame_index
    )
    assert [s.phase for s in smash_phases.segments] == [
        s.phase for s in legacy_phases.segments
    ]
    direct_eval = evaluate_technique(direct_metrics, stroke_type="SMASH")
    assert direct_eval.rule_version


def test_clear_phase_ordering_valid():
    pose, angles, motion, _ = _synthetic_smash(n=40, contact=28)
    phases = detect_clear_phases(pose, angles, motion)
    assert phases.estimated_contact_frame_index == 28
    order = [s.phase for s in phases.segments]
    # Contiguous non-decreasing progression through the shared vocabulary.
    allowed = {
        SmashPhase.PREPARATION,
        SmashPhase.BACKSWING,
        SmashPhase.ACCELERATION,
        SmashPhase.ESTIMATED_CONTACT,
        SmashPhase.FOLLOW_THROUGH,
        SmashPhase.RECOVERY,
    }
    assert set(order).issubset(allowed)
    for earlier, later in zip(phases.segments, phases.segments[1:]):
        assert earlier.end_frame_index + 1 == later.start_frame_index or (
            earlier.end_frame_index == later.start_frame_index
        )
        assert earlier.start_frame_index <= earlier.end_frame_index
    # Recovery, when present, is last.
    if SmashPhase.RECOVERY in order:
        assert order[-1] == SmashPhase.RECOVERY
        assert SmashPhase.FOLLOW_THROUGH in order


def test_clear_metrics_derive_from_finalized_contact_phases(tmp_path: Path):
    state = _clear_state(tmp_path)
    analyzer = ForehandClearAnalyzer()
    metrics = analyzer.extract_metrics(state)
    assert isinstance(metrics, ForehandClearMetrics)
    assert metrics.stroke_type == "FOREHAND_CLEAR"
    assert metrics.estimated_contact_frame_index == state.contact.frame_index
    assert metrics.contact_elbow_angle_deg is not None
    assert metrics.peak_wrist_speed is not None
    assert metrics.preparation_elbow_angle_deg is not None


def test_low_confidence_clear_metrics_produce_insufficient_evidence():
    pose, angles, motion, _ = _synthetic_smash(n=40, contact=28)
    phases = detect_clear_phases(pose, angles, motion)
    low_phases = replace(phases, confidence=0.05)
    metrics = compute_clear_metrics(pose, angles, motion, low_phases)
    metrics = replace(
        metrics,
        contact_confidence=0.05,
        phase_confidence=0.05,
        contact_elbow_angle_deg=120.0,
    )
    evaluation = evaluate_clear_technique(
        metrics,
        profiles=[],
        quality_confidence=0.05,
        pose_confidence=0.05,
    )
    assert evaluation.issues
    assert all(
        i.status == IssueStatus.INSUFFICIENT_EVIDENCE.value or i.uncertain
        for i in evaluation.issues
    )


def test_evidence_package_preserves_stroke_type(tmp_path: Path):
    state = _clear_state(tmp_path)
    analyzer = ForehandClearAnalyzer()
    metrics = analyzer.extract_metrics(state)
    technique = analyzer.evaluate(state, metrics, quality_confidence=0.9)
    snapshot = build_analysis_snapshot(state, analysis_id="clear_test")
    apply_provenance(metrics, snapshot, artifact_schema_version="v1")
    apply_provenance(technique, snapshot, artifact_schema_version="v1")
    keyframes = _empty_keyframes(state.smoothed_pose.video)
    apply_provenance(keyframes, snapshot, artifact_schema_version="v1")
    package = EvidencePackager().package_from_final(
        state,
        metrics=metrics,
        technique=technique,
        keyframes=keyframes,
        snapshot=snapshot,
        stroke_type=StrokeType.FOREHAND_CLEAR.value,
    )
    assert package.stroke_type == "FOREHAND_CLEAR"
    assert package.metrics.get("stroke_type") == "FOREHAND_CLEAR"


def test_coaching_receives_clear_evidence(tmp_path: Path):
    state = _clear_state(tmp_path)
    analyzer = ForehandClearAnalyzer()
    metrics = analyzer.extract_metrics(state)
    technique = analyzer.evaluate(state, metrics, quality_confidence=0.9)
    snapshot = build_analysis_snapshot(state, analysis_id="clear_coach")
    apply_provenance(metrics, snapshot, artifact_schema_version="v1")
    apply_provenance(technique, snapshot, artifact_schema_version="v1")
    keyframes = _empty_keyframes(state.smoothed_pose.video)
    apply_provenance(keyframes, snapshot, artifact_schema_version="v1")
    package = EvidencePackager().package_from_final(
        state,
        metrics=metrics,
        technique=technique,
        keyframes=keyframes,
        snapshot=snapshot,
        stroke_type="FOREHAND_CLEAR",
    )
    report = generate_coaching_report_from_final(state, package, snapshot=snapshot)
    assert package.stroke_type == "FOREHAND_CLEAR"
    assert report is not None


def test_dataset_export_records_clear_stroke_type(tmp_path: Path):
    state = _clear_state(tmp_path)
    analyzer = ForehandClearAnalyzer()
    metrics = analyzer.extract_metrics(state)
    technique = analyzer.evaluate(state, metrics, quality_confidence=0.9)
    from app.services.video_service import _artifact_base_stem

    analysis_id = _artifact_base_stem(state.output_path)
    snapshot = build_analysis_snapshot(state, analysis_id=analysis_id)
    apply_provenance(state.phases, snapshot, artifact_schema_version="v1")
    apply_provenance(state.contact, snapshot, artifact_schema_version="v1")
    apply_provenance(state.video_quality, snapshot, artifact_schema_version="v1")
    apply_provenance(metrics, snapshot, artifact_schema_version="v1")
    apply_provenance(technique, snapshot, artifact_schema_version="v1")
    keyframes = _empty_keyframes(state.smoothed_pose.video)
    apply_provenance(keyframes, snapshot, artifact_schema_version="v1")
    export_path, _, export = DatasetExporter().export_from_final(
        state,
        metrics=metrics,
        technique=technique,
        keyframes=keyframes,
        snapshot=snapshot,
        stroke_type="FOREHAND_CLEAR",
    )
    assert export.stroke_type == "FOREHAND_CLEAR"
    assert export_path.exists()


def test_clear_evaluator_rejects_smash_profile():
    pose, angles, motion, _ = _synthetic_smash()
    phases = detect_clear_phases(pose, angles, motion)
    metrics = compute_clear_metrics(pose, angles, motion, phases)
    smash = build_provisional_smash_right_side()
    with pytest.raises(ValueError, match="non-clear profile"):
        evaluate_clear_technique(metrics, profile=smash)
