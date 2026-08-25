"""Orchestrate frame-by-frame pose estimation, skeleton overlay, and JSON export."""

from __future__ import annotations

from pathlib import Path

from app.config import settings
from app.cv.densepose.debug import DensePoseDebugSession
from app.cv.mmpose_estimator import MMPoseEstimator
from app.cv.overlay import AnnotationRenderer
from app.processing.angles import compute_angle_sequence
from app.processing.motion import compute_motion_derivatives
from app.processing.phases import detect_smash_phases
from app.processing.stroke_metrics import compute_stroke_metrics
from app.processing.technique import evaluate_technique
from app.processing.technique_config import technique_rule_config_from_settings
from app.processing.temporal import preprocess_pose_sequence
from app.schemas.angles import AngleSequence
from app.schemas.motion import MotionSequence
from app.schemas.phases import PhaseSequence
from app.schemas.pose import PoseFrame, PoseSequence
from app.schemas.stroke_metrics import StrokeMetrics
from app.schemas.technique import TechniqueEvaluation
from app.services.video_service import (
    angles_json_path_for,
    iter_video_frames,
    motion_json_path_for,
    phases_json_path_for,
    pose_json_path_for,
    process_video_frames,
    smoothed_pose_json_path_for,
    stroke_metrics_json_path_for,
    technique_json_path_for,
)


class PoseService:
    def __init__(self) -> None:
        self._estimator: MMPoseEstimator | None = None

    @property
    def estimator(self) -> MMPoseEstimator:
        if self._estimator is None:
            self._estimator = MMPoseEstimator()
        return self._estimator

    def analyze_video(
        self,
        input_path: Path,
        output_path: Path,
        *,
        muscle_overlay: bool | None = None,
    ) -> tuple[
        Path,
        Path,
        Path,
        Path,
        Path,
        Path,
        Path,
        PoseSequence,
        PoseSequence,
        AngleSequence,
        MotionSequence,
        PhaseSequence,
        StrokeMetrics,
        TechniqueEvaluation,
    ]:
        """Process video; return artifact paths plus analysis sequences."""
        estimator = self.estimator
        raw_sequence = PoseSequence(video=output_path.name)

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
        angle_sequence = compute_angle_sequence(
            smoothed_sequence,
            confidence_threshold=settings.pose_confidence_threshold,
        )
        motion_sequence = compute_motion_derivatives(
            smoothed_sequence,
            angle_sequence,
            confidence_threshold=settings.pose_confidence_threshold,
        )
        phase_sequence = detect_smash_phases(
            smoothed_sequence,
            angle_sequence,
            motion_sequence,
        )
        stroke_metrics = compute_stroke_metrics(
            smoothed_sequence,
            angle_sequence,
            motion_sequence,
            phase_sequence,
        )
        technique_evaluation = evaluate_technique(
            stroke_metrics,
            technique_rule_config_from_settings(),
        )

        pose_by_index = {f.frame_index: f for f in smoothed_sequence.frames}
        angle_by_index = {f.frame_index: f for f in angle_sequence.frames}
        motion_by_index = {f.frame_index: f for f in motion_sequence.frames}

        show_muscles = (
            settings.overlay_muscle_enabled
            if muscle_overlay is None
            else muscle_overlay
        )
        debug_session: DensePoseDebugSession | None = None
        if show_muscles and settings.densepose_debug:
            debug_dir = output_path.with_name(f"{output_path.stem}_densepose_debug")
            debug_session = DensePoseDebugSession(
                output_dir=debug_dir,
                max_frames=settings.densepose_debug_frames,
            )

        renderer = AnnotationRenderer(
            muscle_overlay=show_muscles,
            debug_session=debug_session,
        )
        if show_muscles:
            renderer.ensure_muscle_overlay_ready()

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

        raw_json_path = pose_json_path_for(output_path)
        smoothed_json_path = smoothed_pose_json_path_for(output_path)
        angles_json_path = angles_json_path_for(output_path)
        motion_json_path = motion_json_path_for(output_path)
        phases_json_path = phases_json_path_for(output_path)
        metrics_json_path = stroke_metrics_json_path_for(output_path)
        technique_json_path = technique_json_path_for(output_path)
        raw_sequence.save_json(raw_json_path)
        smoothed_sequence.save_json(smoothed_json_path)
        angle_sequence.save_json(angles_json_path)
        motion_sequence.save_json(motion_json_path)
        phase_sequence.save_json(phases_json_path)
        stroke_metrics.save_json(metrics_json_path)
        technique_evaluation.save_json(technique_json_path)
        return (
            output_path,
            raw_json_path,
            smoothed_json_path,
            angles_json_path,
            motion_json_path,
            phases_json_path,
            metrics_json_path,
            technique_json_path,
            raw_sequence,
            smoothed_sequence,
            angle_sequence,
            motion_sequence,
            phase_sequence,
            stroke_metrics,
            technique_evaluation,
        )


pose_service = PoseService()
