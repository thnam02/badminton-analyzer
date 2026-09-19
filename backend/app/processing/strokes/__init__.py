"""Package for stroke-specific analyzers."""

from app.processing.strokes.registry import get_stroke_analyzer, list_supported_stroke_types

__all__ = ["get_stroke_analyzer", "list_supported_stroke_types"]
