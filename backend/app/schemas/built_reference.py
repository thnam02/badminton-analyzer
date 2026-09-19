"""Versioned reference-profile artifacts produced by the offline builder.

These schemas store full empirical distributions for technique metrics.
They do **not** change production technique rules; ``ReferenceProfile`` /
``MetricReference`` remain the runtime comparison shape until a later wiring step.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.schemas.reference import MetricReference, ReferenceProfile

REFERENCE_PROFILE_BUILD_VERSION = "1.0.0"


@dataclass(frozen=True, slots=True)
class MetricDistribution:
    """Robust univariate summary for one technique metric within a profile group."""

    metric_id: str
    unit: str
    direction: str
    sample_count: int
    observed_count: int
    missing_rate: float
    median: float | None = None
    mean: float | None = None
    std: float | None = None
    p10: float | None = None
    p25: float | None = None
    p75: float | None = None
    p90: float | None = None
    iqr: float | None = None
    quality_summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "unit": self.unit,
            "direction": self.direction,
            "sample_count": self.sample_count,
            "observed_count": self.observed_count,
            "missing_rate": self.missing_rate,
            "median": self.median,
            "mean": self.mean,
            "std": self.std,
            "p10": self.p10,
            "p25": self.p25,
            "p75": self.p75,
            "p90": self.p90,
            "iqr": self.iqr,
            "quality_summary": dict(self.quality_summary),
        }

    def to_metric_reference(
        self,
        *,
        provenance: str = "built_from_reference_dataset",
        provisional: bool = True,
        confidence: float | None = None,
    ) -> MetricReference:
        """Project to the runtime MetricReference band (P10–P90)."""
        conf = confidence
        if conf is None:
            raw = self.quality_summary.get("mean_analysis_confidence")
            conf = float(raw) if raw is not None else 0.5
        return MetricReference(
            metric_id=self.metric_id,
            unit=self.unit,
            median=self.median,
            mean=self.mean,
            std=self.std,
            p25=self.p25,
            p75=self.p75,
            iqr=self.iqr,
            lower_percentile=self.p10,
            upper_percentile=self.p90,
            lower_percentile_rank=10.0,
            upper_percentile_rank=90.0,
            sample_count=self.observed_count,
            provenance=provenance,
            confidence=float(conf),
            provisional=provisional,
            direction=self.direction,
        )


@dataclass(slots=True)
class BuiltReferenceProfile:
    """One grouped empirical reference profile (stroke / hand / view / skill)."""

    profile_id: str
    stroke_type: str
    handedness: str
    camera_view: str
    skill_level: str
    build_version: str = REFERENCE_PROFILE_BUILD_VERSION
    stroke_count: int = 0
    included_stroke_ids: list[str] = field(default_factory=list)
    excluded_stroke_ids: list[str] = field(default_factory=list)
    metrics: dict[str, MetricDistribution] = field(default_factory=dict)
    quality_summary: dict[str, Any] = field(default_factory=dict)
    notes: str = (
        "Built from accepted StrokeSample rows + finalized analysis metrics. "
        "Not yet wired into production technique rules."
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference_profile_build_version": self.build_version,
            "profile_id": self.profile_id,
            "stroke_type": self.stroke_type,
            "handedness": self.handedness,
            "camera_view": self.camera_view,
            "skill_level": self.skill_level,
            "stroke_count": self.stroke_count,
            "included_stroke_ids": list(self.included_stroke_ids),
            "excluded_stroke_ids": list(self.excluded_stroke_ids),
            "metrics": {k: v.to_dict() for k, v in sorted(self.metrics.items())},
            "quality_summary": dict(self.quality_summary),
            "notes": self.notes,
        }

    def to_reference_profile(self, *, provisional: bool = True) -> ReferenceProfile:
        """Compatibility projection for technique-rule wiring."""
        return ReferenceProfile(
            profile_id=self.profile_id,
            stroke_type=self.stroke_type,
            handedness=self.handedness,
            camera_view=self.camera_view,
            skill_level=self.skill_level,
            metrics={
                mid: dist.to_metric_reference(provisional=provisional)
                for mid, dist in self.metrics.items()
            },
            provisional=provisional,
            profile_version=self.build_version,
            source="built_reference",
            notes=self.notes,
        )

    def save_json(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path


@dataclass(slots=True)
class BuiltReferenceProfileSet:
    """Versioned export of all built profiles from one dataset build."""

    build_version: str = REFERENCE_PROFILE_BUILD_VERSION
    protocol_version: str = ""
    dataset_schema_version: str = ""
    profiles: list[BuiltReferenceProfile] = field(default_factory=list)
    total_input_strokes: int = 0
    total_included_strokes: int = 0
    total_excluded_strokes: int = 0
    exclusion_reasons: dict[str, int] = field(default_factory=dict)
    notes: str = (
        "Deterministic reference-profile build output. "
        "Does not modify production technique thresholds."
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference_profile_build_version": self.build_version,
            "protocol_version": self.protocol_version,
            "dataset_schema_version": self.dataset_schema_version,
            "profile_count": len(self.profiles),
            "total_input_strokes": self.total_input_strokes,
            "total_included_strokes": self.total_included_strokes,
            "total_excluded_strokes": self.total_excluded_strokes,
            "exclusion_reasons": dict(self.exclusion_reasons),
            "profiles": [p.to_dict() for p in self.profiles],
            "notes": self.notes,
        }

    def save_json(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path
