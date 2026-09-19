"""Smash stroke analyzer — wraps existing smash phases/metrics/evaluator."""

from __future__ import annotations

from app.processing.phases import detect_smash_phases
from app.processing.stroke_metrics import compute_stroke_metrics_from_final
from app.processing.technique import evaluate_technique_from_final
from app.processing.technique_config import reference_profile_from_settings
from app.schemas.angles import AngleSequence
from app.schemas.final_analysis import FinalAnalysisState
from app.schemas.motion import MotionSequence
from app.schemas.phases import PhaseSequence
from app.schemas.pose import PoseSequence
from app.schemas.stroke_metrics import StrokeMetrics
from app.schemas.stroke_types import StrokeType
from app.schemas.technique import TechniqueEvaluation


class SmashAnalyzer:
    """FOREHAND_SMASH strategy — preserves existing smash behavior."""

    stroke_type = StrokeType.FOREHAND_SMASH

    def detect_phases(
        self,
        pose: PoseSequence,
        angles: AngleSequence,
        motion: MotionSequence,
        *,
        forced_contact_frame_index: int | None = None,
    ) -> PhaseSequence:
        return detect_smash_phases(
            pose,
            angles,
            motion,
            forced_contact_frame_index=forced_contact_frame_index,
        )

    def extract_metrics(self, state: FinalAnalysisState) -> StrokeMetrics:
        return compute_stroke_metrics_from_final(state)

    def evaluate(
        self,
        state: FinalAnalysisState,
        metrics: StrokeMetrics,
        *,
        quality_confidence: float | None = None,
        handedness: str | None = None,
        camera_view: str | None = None,
        skill_level: str | None = None,
    ) -> TechniqueEvaluation:
        q = quality_confidence
        if q is None:
            q = float(state.video_quality.analysis_confidence)
        return evaluate_technique_from_final(
            state,
            metrics,
            profile=reference_profile_from_settings(
                stroke_type="SMASH",
                handedness=handedness,
            ),
            stroke_type="SMASH",
            handedness=handedness,
            camera_view=camera_view,
            skill_level=skill_level,
            quality_confidence=q,
        )
