"""Stroke analyzer registry / factory."""

from __future__ import annotations

from app.processing.strokes.base import StrokeAnalyzer
from app.schemas.stroke_types import (
    StrokeType,
    UnsupportedStrokeError,
    normalize_stroke_type,
)


def get_stroke_analyzer(stroke_type: str | StrokeType | None) -> StrokeAnalyzer:
    """Return the analyzer for a supported stroke type."""
    st = normalize_stroke_type(stroke_type)
    if st == StrokeType.FOREHAND_SMASH:
        from app.processing.strokes.smash.analyzer import SmashAnalyzer

        return SmashAnalyzer()
    if st == StrokeType.FOREHAND_CLEAR:
        from app.processing.strokes.clear.analyzer import ForehandClearAnalyzer

        return ForehandClearAnalyzer()
    raise UnsupportedStrokeError(f"No analyzer registered for {st.value}")


def list_supported_stroke_types() -> list[str]:
    return [StrokeType.FOREHAND_SMASH.value, StrokeType.FOREHAND_CLEAR.value]
