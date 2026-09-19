"""Stroke analyzer protocol and shared result container."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from app.schemas.angles import AngleSequence
from app.schemas.contact import ContactEvent
from app.schemas.final_analysis import FinalAnalysisState
from app.schemas.motion import MotionSequence
from app.schemas.phases import PhaseSequence
from app.schemas.pose import PoseSequence
from app.schemas.stroke_types import StrokeType
from app.schemas.technique import TechniqueEvaluation


@dataclass(slots=True)
class StrokeAnalysisBundle:
    """Stroke-specific outputs after FinalAnalysisState is locked."""

    stroke_type: StrokeType
    phases: PhaseSequence
    metrics: Any
    technique: TechniqueEvaluation
    metrics_schema: str = ""


class StrokeAnalyzer(Protocol):
    """Stroke-specific phase / metrics / evaluation strategy."""

    stroke_type: StrokeType

    def detect_phases(
        self,
        pose: PoseSequence,
        angles: AngleSequence,
        motion: MotionSequence,
        *,
        forced_contact_frame_index: int | None = None,
    ) -> PhaseSequence: ...

    def extract_metrics(
        self,
        state: FinalAnalysisState,
    ) -> Any: ...

    def evaluate(
        self,
        state: FinalAnalysisState,
        metrics: Any,
        *,
        quality_confidence: float | None = None,
        handedness: str | None = None,
        camera_view: str | None = None,
        skill_level: str | None = None,
    ) -> TechniqueEvaluation: ...
