"""Forehand clear stroke analyzer."""

from __future__ import annotations

from app.processing.strokes.clear.evaluator import evaluate_clear_from_final
from app.processing.strokes.clear.metrics import compute_clear_metrics_from_final
from app.processing.strokes.clear.phases import detect_clear_phases
from app.schemas.angles import AngleSequence
from app.schemas.clear_metrics import ForehandClearMetrics
from app.schemas.final_analysis import FinalAnalysisState
from app.schemas.motion import MotionSequence
from app.schemas.phases import PhaseSequence
from app.schemas.pose import PoseSequence
from app.schemas.stroke_types import StrokeType
from app.schemas.technique import TechniqueEvaluation


class ForehandClearAnalyzer:
    """FOREHAND_CLEAR strategy — clear-specific phases/metrics/evaluator."""

    stroke_type = StrokeType.FOREHAND_CLEAR

    def detect_phases(
        self,
        pose: PoseSequence,
        angles: AngleSequence,
        motion: MotionSequence,
        *,
        forced_contact_frame_index: int | None = None,
    ) -> PhaseSequence:
        return detect_clear_phases(
            pose,
            angles,
            motion,
            forced_contact_frame_index=forced_contact_frame_index,
        )

    def extract_metrics(self, state: FinalAnalysisState) -> ForehandClearMetrics:
        return compute_clear_metrics_from_final(state)

    def evaluate(
        self,
        state: FinalAnalysisState,
        metrics: ForehandClearMetrics,
        *,
        quality_confidence: float | None = None,
        handedness: str | None = None,
        camera_view: str | None = None,
        skill_level: str | None = None,
    ) -> TechniqueEvaluation:
        return evaluate_clear_from_final(
            state,
            metrics,
            quality_confidence=quality_confidence,
            handedness=handedness,
            camera_view=camera_view,
            skill_level=skill_level,
        )
