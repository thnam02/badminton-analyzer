"""Factory for research AQA models (mock only in this milestone)."""

from __future__ import annotations

from app.research.aqa.mock_model import MockActionQualityModel
from app.research.aqa.protocol import ActionQualityModel


def get_aqa_model(backend: str | None = None) -> ActionQualityModel:
    """Return a research AQA model.

    Production analyze/coaching paths must not call this until evaluation gates
    in ``docs/aqa-research.md`` are met. Default backend is ``mock``.
    """
    name = (backend or "mock").strip().lower()
    if name in ("mock", "mock_aqa", "none"):
        return MockActionQualityModel()
    raise ValueError(
        f"Unknown AQA backend '{name}'. Only 'mock' is available; "
        "learned backends are not implemented yet."
    )
