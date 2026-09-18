"""Orchestrate frame-by-frame pose estimation, skeleton overlay, and JSON export."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.ai.coaching import generate_coaching_report
from app.config import settings
from app.cv.mmpose_estimator import MMPoseEstimator
from app.cv.overlay import AnnotationRenderer
from app.processing.angles import compute_angle_sequence
from app.processing.contact_resolver import resolve_contact
from app.processing.keyframes import extract_keyframes
from app.processing.motion import compute_motion_derivatives
from app.processing.phases import detect_smash_phases
from app.processing.stroke_metrics import compute_stroke_metrics
from app.processing.technique import evaluate_technique
from app.processing.technique_config import reference_profile_from_settings
from app.processing.temporal import preprocess_pose_sequence
from app.processing.video_quality import assess_video_quality
from app.schemas.angles import AngleSequence
from app.schemas.contact import CONTACT_TYPE_KINEMATIC, ContactEvent
from app.schemas.motion import MotionSequence
from app.schemas.phases import PhaseSequence
from app.schemas.pose import PoseFrame, PoseSequence
from app.schemas.racket import RacketTrajectory
from app.schemas.shuttle import ShuttleTrajectory
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
    """Pose measurements up through the initial kinematic contact candidate.

    No evidence, coaching, overlay, metrics, or technique yet — those wait until
    ContactResolver produces the single final contact / phase state.
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
    ) -> tuple:
        """Resolve final contact once, then build every downstream artifact from it.

        Order: ContactResolver → final contact → phase re-snap → stroke metrics →
        technique → keyframes → evidence → coaching → annotated video → (async mesh).
        """
        input_path = kinematics.input_path
        output_path = kinematics.output_path
        pose = kinematics.smoothed_sequence
        angles = kinematics.angle_sequence
        motion = kinematics.motion_sequence
        quality_report = kinematics.quality_report
        initial_phases = kinematics.initial_phases

        contact_event = _resolve_final_contact(
            initial_phases=initial_phases,
            kinematic_contact=kinematics.kinematic_contact,
            pose=pose,
            motion=motion,
            shuttle=shuttle,
            racket=racket,
        )
        if kinematics.kinematic_contact.kinematic_frame_index is not None:
            phase_sequence = detect_smash_phases(
                pose,
                angles,
                motion,
                forced_contact_frame_index=contact_event.frame_index,
            )
        else:
            phase_sequence = initial_phases

        stroke_metrics = compute_stroke_metrics(
            pose,
            angles,
            motion,
            phase_sequence,
            contact=contact_event,
        )
        technique_evaluation = evaluate_technique(
            stroke_metrics,
            profile=reference_profile_from_settings(
                stroke_type="SMASH",
                handedness=None,
            ),
        )
        keyframe_set = extract_keyframes(
            input_path,
            phase_sequence,
            pose,
            keyframes_dir_for(output_path),
            include_contact_neighbors=True,
        )
        evidence_package = evidence_packager.package(
            video_quality=quality_report,
            phases=phase_sequence,
            metrics=stroke_metrics,
            technique=technique_evaluation,
            keyframes=keyframe_set,
            handedness=None,
            contact=contact_event,
        )
        coaching_report = generate_coaching_report(evidence_package)

        pose_by_index = {f.frame_index: f for f in pose.frames}
        angle_by_index = {f.frame_index: f for f in angles.frames}
        motion_by_index = {f.frame_index: f for f in motion.frames}

        renderer = AnnotationRenderer(muscle_overlay=False)

        def render_frame(frame, frame_index: int, fps: float):
            del fps
            return renderer.render(
                frame,
                pose_frame=pose_by_index.get(frame_index),
                angle_frame=angle_by_index.get(frame_index),
                motion_frame=motion_by_index.get(frame_index),
                phase=phase_sequence.phase_at(frame_index),
                frame_index=frame_index,
            )

        process_video_frames(input_path, output_path, render_frame)

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
                pose_sequence=pose,
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

        kinematics.raw_sequence.save_json(raw_json_path)
        pose.save_json(smoothed_json_path)
        angles.save_json(angles_json_path)
        motion.save_json(motion_json_path)
        phase_sequence.save_json(phases_json_path)
        stroke_metrics.save_json(metrics_json_path)
        technique_evaluation.save_json(technique_json_path)
        quality_report.save_json(quality_json_path)
        keyframe_set.save_json(keyframes_json_path)
        evidence_package.save_json(evidence_json_path)
        coaching_report.save_json(coaching_json_path)
        contact_event.save_json(contact_json_path)

        return (
            output_path,
            raw_json_path,
            smoothed_json_path,
            angles_json_path,
            motion_json_path,
            phases_json_path,
            metrics_json_path,
            technique_json_path,
            quality_json_path,
            keyframes_json_path,
            evidence_json_path,
            coaching_json_path,
            contact_json_path,
            mesh_video_path,
            mesh_json_path,
            mesh_status,
            kinematics.raw_sequence,
            pose,
            angles,
            motion,
            phase_sequence,
            stroke_metrics,
            technique_evaluation,
            quality_report,
            keyframe_set,
            evidence_package,
            coaching_report,
            contact_event,
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
    ) -> tuple:
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
