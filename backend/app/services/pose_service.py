"""Orchestrate frame-by-frame pose estimation, skeleton overlay, and JSON export."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.ai.coaching import generate_coaching_report_from_final
from app.config import settings
from app.cv.mmpose_estimator import MMPoseEstimator
from app.cv.overlay import AnnotationRenderer
from app.processing.angles import compute_angle_sequence
from app.processing.contact_resolver import resolve_contact
from app.processing.keyframes import extract_keyframes_from_final
from app.processing.motion import compute_motion_derivatives
from app.processing.phases import detect_smash_phases
from app.processing.stroke_metrics import compute_stroke_metrics_from_final
from app.processing.technique import evaluate_technique_from_final
from app.processing.technique_config import reference_profile_from_settings
from app.processing.temporal import preprocess_pose_sequence
from app.processing.video_quality import assess_video_quality
from app.schemas.angles import AngleSequence
from app.schemas.coaching import CoachingReport
from app.schemas.contact import CONTACT_TYPE_KINEMATIC, ContactEvent
from app.schemas.evidence import EvidencePackage
from app.schemas.final_analysis import (
    FinalAnalysisState,
    IntermediateAnalysisSnapshot,
    build_final_analysis_state,
)
from app.schemas.keyframes import KeyframeSet
from app.schemas.motion import MotionSequence
from app.schemas.phases import PhaseSequence
from app.schemas.pose import PoseFrame, PoseSequence
from app.schemas.racket import RacketTrajectory
from app.schemas.shuttle import ShuttleTrajectory
from app.schemas.stroke_metrics import StrokeMetrics
from app.schemas.technique import TechniqueEvaluation
from app.schemas.video_quality import VideoQualityReport
from app.services.evidence_packager import evidence_packager
from app.services.video_service import (
    angles_json_path_for,
    coaching_json_path_for,
    contact_json_path_for,
    evidence_json_path_for,
    iter_video_frames,
    keyframes_dir_for,
    keyframes_json_path_for,
    motion_json_path_for,
    phases_json_path_for,
    pose_json_path_for,
    probe_video_metadata,
    process_video_frames,
    smoothed_pose_json_path_for,
    stroke_metrics_json_path_for,
    technique_json_path_for,
    video_quality_json_path_for,
)


@dataclass(slots=True)
class PoseKinematics:
    """Intermediate pose measurements up through the kinematic contact candidate.

    Debug / pre-resolution only. Downstream coaching, overlay, metrics, technique,
    evidence, and export must use ``FinalAnalysisState`` after finalize.
    """

    input_path: Path
    output_path: Path
    raw_sequence: PoseSequence
    smoothed_sequence: PoseSequence
    quality_report: VideoQualityReport
    angle_sequence: AngleSequence
    motion_sequence: MotionSequence
    initial_phases: PhaseSequence
    kinematic_contact: ContactEvent
    video_fps: float
    video_width: int
    video_height: int


@dataclass(slots=True)
class FinalizedAnalysis:
    """Artifacts produced from a single immutable ``FinalAnalysisState``."""

    final_state: FinalAnalysisState
    stroke_metrics: StrokeMetrics
    technique_evaluation: TechniqueEvaluation
    keyframe_set: KeyframeSet
    evidence_package: EvidencePackage
    coaching_report: CoachingReport
    output_path: Path
    raw_json_path: Path
    smoothed_json_path: Path
    angles_json_path: Path
    motion_json_path: Path
    phases_json_path: Path
    metrics_json_path: Path
    technique_json_path: Path
    quality_json_path: Path
    keyframes_json_path: Path
    evidence_json_path: Path
    coaching_json_path: Path
    contact_json_path: Path
    mesh_video_path: Path | None
    mesh_json_path: Path | None
    mesh_status: dict | None


class PoseService:
    def __init__(self) -> None:
        self._estimator: MMPoseEstimator | None = None

    @property
    def estimator(self) -> MMPoseEstimator:
        if self._estimator is None:
            self._estimator = MMPoseEstimator()
        return self._estimator

    def compute_pose_kinematics(
        self,
        input_path: Path,
        output_path: Path,
    ) -> PoseKinematics:
        """RTMPose → temporal → quality → angles/motion → initial phases → kinematic contact."""
        estimator = self.estimator
        raw_sequence = PoseSequence(video=output_path.name)
        video_fps, video_width, video_height = probe_video_metadata(input_path)

        def collect_frame(frame, frame_index: int, fps: float) -> None:
            keypoints = estimator.predict(frame)
            raw_sequence.append(
                PoseFrame(
                    frame_index=frame_index,
                    timestamp=frame_index / fps if fps > 0 else 0.0,
                    keypoints=keypoints,
                )
            )

        iter_video_frames(input_path, collect_frame)

        smoothed_sequence = preprocess_pose_sequence(
            raw_sequence,
            confidence_threshold=settings.pose_confidence_threshold,
            max_gap=settings.pose_interp_max_gap,
            savgol_window=settings.pose_savgol_window,
            savgol_polyorder=settings.pose_savgol_polyorder,
        )
        quality_report = assess_video_quality(
            raw_sequence,
            smoothed_pose=smoothed_sequence,
            fps=video_fps,
            width=video_width,
            height=video_height,
            confidence_threshold=settings.pose_confidence_threshold,
        )
        angle_sequence = compute_angle_sequence(
            smoothed_sequence,
            confidence_threshold=settings.pose_confidence_threshold,
        )
        motion_sequence = compute_motion_derivatives(
            smoothed_sequence,
            angle_sequence,
            confidence_threshold=settings.pose_confidence_threshold,
        )
        initial_phases = detect_smash_phases(
            smoothed_sequence,
            angle_sequence,
            motion_sequence,
        )
        kinematic_contact = _kinematic_contact_event(initial_phases)
        return PoseKinematics(
            input_path=input_path,
            output_path=output_path,
            raw_sequence=raw_sequence,
            smoothed_sequence=smoothed_sequence,
            quality_report=quality_report,
            angle_sequence=angle_sequence,
            motion_sequence=motion_sequence,
            initial_phases=initial_phases,
            kinematic_contact=kinematic_contact,
            video_fps=video_fps,
            video_width=video_width,
            video_height=video_height,
        )

    def finalize_analysis(
        self,
        kinematics: PoseKinematics,
        *,
        shuttle: ShuttleTrajectory | None = None,
        racket: RacketTrajectory | None = None,
        mesh_overlay: bool | None = None,
    ) -> FinalizedAnalysis:
        """Resolve final contact once, lock FinalAnalysisState, then run downstream.

        Order: ContactResolver → final contact → phase re-snap → FinalAnalysisState →
        stroke metrics → technique → keyframes → evidence → coaching → annotated
        video → (async mesh). After ``FinalAnalysisState`` is built, contact and
        phase boundaries are not replaced.
        """
        input_path = kinematics.input_path
        output_path = kinematics.output_path

        contact_event = _resolve_final_contact(
            initial_phases=kinematics.initial_phases,
            kinematic_contact=kinematics.kinematic_contact,
            pose=kinematics.smoothed_sequence,
            motion=kinematics.motion_sequence,
            shuttle=shuttle,
            racket=racket,
        )
        if kinematics.kinematic_contact.kinematic_frame_index is not None:
            final_phases = detect_smash_phases(
                kinematics.smoothed_sequence,
                kinematics.angle_sequence,
                kinematics.motion_sequence,
                forced_contact_frame_index=contact_event.frame_index,
            )
        else:
            final_phases = kinematics.initial_phases

        intermediate = IntermediateAnalysisSnapshot(
            raw_pose=kinematics.raw_sequence,
            initial_phases=kinematics.initial_phases,
            kinematic_contact=kinematics.kinematic_contact,
        )
        final_state = build_final_analysis_state(
            smoothed_pose=kinematics.smoothed_sequence,
            angles=kinematics.angle_sequence,
            motion=kinematics.motion_sequence,
            video_quality=kinematics.quality_report,
            phases=final_phases,
            contact=contact_event,
            video_fps=kinematics.video_fps,
            video_width=kinematics.video_width,
            video_height=kinematics.video_height,
            input_path=input_path,
            output_path=output_path,
            intermediate=intermediate,
        )

        # All downstream stages read contact/phases only via final_state.
        stroke_metrics = compute_stroke_metrics_from_final(final_state)
        technique_evaluation = evaluate_technique_from_final(
            final_state,
            stroke_metrics,
            profile=reference_profile_from_settings(
                stroke_type="SMASH",
                handedness=None,
            ),
        )
        keyframe_set = extract_keyframes_from_final(
            final_state,
            keyframes_dir_for(output_path),
            include_contact_neighbors=True,
        )
        evidence_package = evidence_packager.package_from_final(
            final_state,
            metrics=stroke_metrics,
            technique=technique_evaluation,
            keyframes=keyframe_set,
            handedness=None,
        )
        coaching_report = generate_coaching_report_from_final(
            final_state,
            evidence_package,
        )

        _render_annotated_video_from_final(final_state)

        mesh_video_path: Path | None = None
        mesh_json_path: Path | None = None
        mesh_status: dict | None = None
        run_mesh = settings.mesh_enabled if mesh_overlay is None else mesh_overlay
        if run_mesh:
            # WHAM on CPU can take many minutes; run async so /analyze does not
            # hold the HTTP connection open (browser "Failed to fetch").
            from app.services.mesh_jobs import start_mesh_job
            from app.services.video_service import mesh_json_path_for, mesh_video_path_for

            mesh_video_path = mesh_video_path_for(output_path)
            mesh_json_path = mesh_json_path_for(output_path)
            mesh_status = start_mesh_job(
                video_path=input_path,
                pose_output=output_path,
                pose_sequence=final_state.smoothed_pose,
            )

        raw_json_path = pose_json_path_for(output_path)
        smoothed_json_path = smoothed_pose_json_path_for(output_path)
        angles_json_path = angles_json_path_for(output_path)
        motion_json_path = motion_json_path_for(output_path)
        phases_json_path = phases_json_path_for(output_path)
        metrics_json_path = stroke_metrics_json_path_for(output_path)
        technique_json_path = technique_json_path_for(output_path)
        quality_json_path = video_quality_json_path_for(output_path)
        keyframes_json_path = keyframes_json_path_for(output_path)
        evidence_json_path = evidence_json_path_for(output_path)
        coaching_json_path = coaching_json_path_for(output_path)
        contact_json_path = contact_json_path_for(output_path)

        raw_pose = (
            final_state.intermediate.raw_pose
            if final_state.intermediate is not None
            else kinematics.raw_sequence
        )
        raw_pose.save_json(raw_json_path)
        final_state.smoothed_pose.save_json(smoothed_json_path)
        final_state.angles.save_json(angles_json_path)
        final_state.motion.save_json(motion_json_path)
        final_state.phases.save_json(phases_json_path)
        stroke_metrics.save_json(metrics_json_path)
        technique_evaluation.save_json(technique_json_path)
        final_state.video_quality.save_json(quality_json_path)
        keyframe_set.save_json(keyframes_json_path)
        evidence_package.save_json(evidence_json_path)
        coaching_report.save_json(coaching_json_path)
        final_state.contact.save_json(contact_json_path)

        return FinalizedAnalysis(
            final_state=final_state,
            stroke_metrics=stroke_metrics,
            technique_evaluation=technique_evaluation,
            keyframe_set=keyframe_set,
            evidence_package=evidence_package,
            coaching_report=coaching_report,
            output_path=output_path,
            raw_json_path=raw_json_path,
            smoothed_json_path=smoothed_json_path,
            angles_json_path=angles_json_path,
            motion_json_path=motion_json_path,
            phases_json_path=phases_json_path,
            metrics_json_path=metrics_json_path,
            technique_json_path=technique_json_path,
            quality_json_path=quality_json_path,
            keyframes_json_path=keyframes_json_path,
            evidence_json_path=evidence_json_path,
            coaching_json_path=coaching_json_path,
            contact_json_path=contact_json_path,
            mesh_video_path=mesh_video_path,
            mesh_json_path=mesh_json_path,
            mesh_status=mesh_status,
        )

    def analyze_video(
        self,
        input_path: Path,
        output_path: Path,
        *,
        muscle_overlay: bool | None = None,
        mesh_overlay: bool | None = None,
        shuttle: ShuttleTrajectory | None = None,
        racket: RacketTrajectory | None = None,
    ) -> FinalizedAnalysis:
        """Full pose pipeline with optional precomputed shuttle/racket trajectories.

        DensePose muscle overlay is disabled. WHAM mesh (when enabled) stays an
        independent async debug job after the annotated video is written.
        """
        del muscle_overlay  # retired path — ignored
        kinematics = self.compute_pose_kinematics(input_path, output_path)
        return self.finalize_analysis(
            kinematics,
            shuttle=shuttle,
            racket=racket,
            mesh_overlay=mesh_overlay,
        )


def _render_annotated_video_from_final(state: FinalAnalysisState) -> None:
    """Overlay render consumes only final-state pose / angles / motion / phases."""
    pose_by_index = {f.frame_index: f for f in state.smoothed_pose.frames}
    angle_by_index = {f.frame_index: f for f in state.angles.frames}
    motion_by_index = {f.frame_index: f for f in state.motion.frames}
    renderer = AnnotationRenderer(muscle_overlay=False)

    def render_frame(frame, frame_index: int, fps: float):
        del fps
        return renderer.render_from_final(
            frame,
            state,
            frame_index=frame_index,
            pose_by_index=pose_by_index,
            angle_by_index=angle_by_index,
            motion_by_index=motion_by_index,
        )

    process_video_frames(state.input_path, state.output_path, render_frame)


def _resolve_final_contact(
    *,
    initial_phases: PhaseSequence,
    kinematic_contact: ContactEvent,
    pose: PoseSequence,
    motion: MotionSequence,
    shuttle: ShuttleTrajectory | None,
    racket: RacketTrajectory | None,
) -> ContactEvent:
    """Single ContactResolver call that every downstream artifact must use."""
    kin_frame = kinematic_contact.kinematic_frame_index
    kin_ts = kinematic_contact.kinematic_timestamp
    if kin_frame is None or kin_ts is None:
        return ContactEvent(
            contact_type=CONTACT_TYPE_KINEMATIC,
            frame_index=0,
            timestamp=0.0,
            confidence=0.0,
            notes="No kinematic contact available.",
        )
    return resolve_contact(
        kinematic_frame_index=kin_frame,
        kinematic_timestamp=kin_ts,
        kinematic_confidence=float(initial_phases.confidence),
        pose=pose,
        motion=motion,
        shuttle=shuttle,
        racket=racket,
        hitting_hand=racket.hitting_hand if racket is not None else None,
    )


def _kinematic_contact_event(phases: PhaseSequence) -> ContactEvent:
    frame = phases.estimated_contact_frame_index
    ts = phases.estimated_contact_timestamp
    if frame is None or ts is None:
        return ContactEvent(
            contact_type=CONTACT_TYPE_KINEMATIC,
            frame_index=0,
            timestamp=0.0,
            confidence=0.0,
            notes="No kinematic contact frame detected.",
        )
    return ContactEvent(
        contact_type=CONTACT_TYPE_KINEMATIC,
        frame_index=int(frame),
        timestamp=float(ts),
        confidence=float(phases.confidence),
        kinematic_frame_index=int(frame),
        kinematic_timestamp=float(ts),
        notes=(
            "KINEMATIC_ESTIMATE from peak right-wrist speed "
            "(pending ContactResolver with optional shuttle/racket)."
        ),
    )


pose_service = PoseService()
