"""Reference profiles for percentile-based technique comparison."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class MetricReference:
    """Statistical reference band for one stroke metric.

    ``lower_percentile`` / ``upper_percentile`` are the *values* at the configured
    percentile ranks (default p10 / p90), not the ranks themselves.
    """

    metric_id: str
    unit: str
    median: float | None = None
    lower_percentile: float | None = None
    upper_percentile: float | None = None
    lower_percentile_rank: float = 10.0
    upper_percentile_rank: float = 90.0
    sample_count: int = 0
    provenance: str = "provisional_sample"
    confidence: float = 0.4
    provisional: bool = True
    # higher_is_better | lower_is_better | in_range
    direction: str = "in_range"
    # Optional distribution moments from the C2 builder (robust comparisons).
    mean: float | None = None
    std: float | None = None
    p25: float | None = None
    p75: float | None = None
    iqr: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "unit": self.unit,
            "median": self.median,
            "mean": self.mean,
            "std": self.std,
            "lower_percentile": self.lower_percentile,
            "upper_percentile": self.upper_percentile,
            "lower_percentile_rank": self.lower_percentile_rank,
            "upper_percentile_rank": self.upper_percentile_rank,
            "p25": self.p25,
            "p75": self.p75,
            "iqr": self.iqr,
            "sample_count": self.sample_count,
            "provenance": self.provenance,
            "confidence": self.confidence,
            "provisional": self.provisional,
            "direction": self.direction,
        }


@dataclass(frozen=True, slots=True)
class ReferenceEvidence:
    """Evidence snippet attached to a TechniqueIssue (fully explainable)."""

    metric_id: str
    median: float | None
    lower_percentile: float | None
    upper_percentile: float | None
    sample_count: int
    provenance: str
    confidence: float
    provisional: bool
    direction: str
    mean: float | None = None
    std: float | None = None
    iqr: float | None = None
    lower_percentile_rank: float = 10.0
    upper_percentile_rank: float = 90.0
    deviation: float | None = None
    percentile_position: float | None = None
    robust_z: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "median": self.median,
            "mean": self.mean,
            "std": self.std,
            "iqr": self.iqr,
            "lower_percentile": self.lower_percentile,
            "upper_percentile": self.upper_percentile,
            "lower_percentile_rank": self.lower_percentile_rank,
            "upper_percentile_rank": self.upper_percentile_rank,
            "sample_count": self.sample_count,
            "provenance": self.provenance,
            "confidence": self.confidence,
            "provisional": self.provisional,
            "direction": self.direction,
            "deviation": self.deviation,
            "percentile_position": self.percentile_position,
            "robust_z": self.robust_z,
        }


@dataclass(frozen=True, slots=True)
class ReferenceProfile:
    """Keyed reference band set for a stroke / handedness / camera / skill context."""

    profile_id: str
    stroke_type: str
    handedness: str | None = None
    camera_view: str | None = None
    skill_level: str | None = None
    metrics: dict[str, MetricReference] = field(default_factory=dict)
    provisional: bool = True
    profile_version: str = ""
    source: str = "provisional_catalog"
    notes: str = (
        "Provisional reference values from configuration/sample data — "
        "not scientifically validated."
    )

    def get_metric(self, metric_id: str) -> MetricReference | None:
        return self.metrics.get(metric_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "stroke_type": self.stroke_type,
            "handedness": self.handedness,
            "camera_view": self.camera_view,
            "skill_level": self.skill_level,
            "provisional": self.provisional,
            "profile_version": self.profile_version,
            "source": self.source,
            "notes": self.notes,
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
        }
