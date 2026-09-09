"""ActionQualityModel protocol — research interface only."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.research.aqa.schemas import AQASample, ActionQualityPrediction


@runtime_checkable
class ActionQualityModel(Protocol):
    """Learned badminton Action Quality Assessment model interface.

    Implementations must not be wired into production coaching feedback until
    evaluation gates in docs/aqa-research.md are satisfied. Deterministic
    biomechanics stay available via ``sample.deterministic_evidence``.
    """

    model_id: str

    def is_available(self) -> bool:
        """Return True when weights / runtime deps are ready."""

    def predict(self, sample: AQASample) -> ActionQualityPrediction:
        """Predict quality dimensions + issue probabilities with uncertainty."""
