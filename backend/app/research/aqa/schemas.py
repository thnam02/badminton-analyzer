"""Typed inputs/outputs for research Action Quality Assessment models."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

AQA_PREDICTION_VERSION = "0.1.0-research"

# Align with coach annotation facets (schemas.annotation).
QUALITY_DIMENSION_NAMES: tuple[str, ...] = (
    "preparation_quality",
    "kinetic_chain_timing",
    "contact_quality",
    "follow_through_quality",
)


@dataclass(slots=True)
class UncertaintyEstimate:
    """Simple uncertainty container (aleatoric/epistemic split optional later)."""

    std: float = 0.0
    confidence: float = 0.0  # 1 - normalized uncertainty, [0, 1]
    method: str = "unspecified"

    def to_dict(self) -> dict[str, Any]:
        return {
            "std": self.std,
            "confidence": self.confidence,
            "method": self.method,
        }


@dataclass(slots=True)
class QualityDimensionPrediction:
    name: str
    score_mean: float  # mapped to ~[1, 4] coach scale when available
    rating_probabilities: dict[str, float] = field(default_factory=dict)
    uncertainty: UncertaintyEstimate = field(default_factory=UncertaintyEstimate)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "score_mean": self.score_mean,
            "rating_probabilities": dict(self.rating_probabilities),
            "uncertainty": self.uncertainty.to_dict(),
        }


@dataclass(slots=True)
class IssueProbability:
    code: str
    probability: float
    uncertainty: UncertaintyEstimate = field(default_factory=UncertaintyEstimate)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "probability": self.probability,
            "uncertainty": self.uncertainty.to_dict(),
            "notes": self.notes,
        }


@dataclass(slots=True)
class PhaseNormalizedMotionFeatures:
    """Phase-aligned motion descriptors for a stroke.

    Values are research placeholders — loaders may leave arrays empty until a
    feature extractor is implemented. Keys are smash phase names.
    """

    phases: dict[str, list[float]] = field(default_factory=dict)
    contact_frame_index: int | None = None
    feature_names: list[str] = field(default_factory=list)
    notes: str = (
        "Phase-normalized motion features for AQA research; "
        "not used by the production technique evaluator."
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "phases": {k: list(v) for k, v in self.phases.items()},
            "contact_frame_index": self.contact_frame_index,
            "feature_names": list(self.feature_names),
            "notes": self.notes,
        }


@dataclass(slots=True)
class RGBKeyframeFeatures:
    """Optional RGB / keyframe embeddings (paths or vectors)."""

    keyframe_paths: list[str] = field(default_factory=list)
    embedding_dim: int | None = None
    embeddings: list[list[float]] = field(default_factory=list)
    backend: str = "none"
    notes: str = "RGB/keyframe features are optional; empty until a vision encoder is added."

    def to_dict(self) -> dict[str, Any]:
        return {
            "keyframe_paths": list(self.keyframe_paths),
            "embedding_dim": self.embedding_dim,
            "embeddings": [list(e) for e in self.embeddings],
            "backend": self.backend,
            "notes": self.notes,
        }


@dataclass(slots=True)
class ReferenceProfileContext:
    """Reference-profile bands available as model context (not hard labels)."""

    profile_id: str | None = None
    stroke_type: str | None = None
    handedness: str | None = None
    camera_view: str | None = None
    metric_bands: dict[str, dict[str, Any]] = field(default_factory=dict)
    provisional: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "stroke_type": self.stroke_type,
            "handedness": self.handedness,
            "camera_view": self.camera_view,
            "metric_bands": self.metric_bands,
            "provisional": self.provisional,
        }


@dataclass(slots=True)
class DeterministicEvidenceBundle:
    """Explainable biomechanics retained alongside any learned prediction."""

    pose_metrics: dict[str, Any] = field(default_factory=dict)
    phases: dict[str, Any] = field(default_factory=dict)
    contact_event: dict[str, Any] = field(default_factory=dict)
    technique_issues: list[dict[str, Any]] = field(default_factory=list)
    video_quality: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "pose_metrics": self.pose_metrics,
            "phases": self.phases,
            "contact_event": self.contact_event,
            "technique_issues": list(self.technique_issues),
            "video_quality": self.video_quality,
        }


@dataclass(slots=True)
class AQASample:
    """One stroke sample for research AQA models / dataset iteration."""

    analysis_id: str
    stroke_type: str = "SMASH"
    pose_frames: list[dict[str, Any]] = field(default_factory=list)
    phase_normalized_motion: PhaseNormalizedMotionFeatures = field(
        default_factory=PhaseNormalizedMotionFeatures
    )
    rgb_keyframe_features: RGBKeyframeFeatures = field(
        default_factory=RGBKeyframeFeatures
    )
    reference_profile: ReferenceProfileContext = field(
        default_factory=ReferenceProfileContext
    )
    coach_labels: list[dict[str, Any]] = field(default_factory=list)
    deterministic_evidence: DeterministicEvidenceBundle = field(
        default_factory=DeterministicEvidenceBundle
    )
    artifact_refs: dict[str, str | None] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis_id": self.analysis_id,
            "stroke_type": self.stroke_type,
            "pose_frame_count": len(self.pose_frames),
            "pose_frames": self.pose_frames,
            "phase_normalized_motion": self.phase_normalized_motion.to_dict(),
            "rgb_keyframe_features": self.rgb_keyframe_features.to_dict(),
            "reference_profile": self.reference_profile.to_dict(),
            "coach_labels": list(self.coach_labels),
            "deterministic_evidence": self.deterministic_evidence.to_dict(),
            "artifact_refs": dict(self.artifact_refs),
        }


@dataclass(slots=True)
class ActionQualityPrediction:
    """Learned (or mock) AQA output — research-only until evaluation gates pass."""

    prediction_version: str = AQA_PREDICTION_VERSION
    model_id: str = "mock"
    analysis_id: str = ""
    quality_dimensions: list[QualityDimensionPrediction] = field(default_factory=list)
    issue_probabilities: list[IssueProbability] = field(default_factory=list)
    overall_quality_mean: float | None = None
    overall_uncertainty: UncertaintyEstimate = field(
        default_factory=UncertaintyEstimate
    )
    research_only: bool = True
    affects_production_feedback: bool = False
    deterministic_evidence: DeterministicEvidenceBundle | None = None
    notes: str = (
        "Research AQA prediction. Deterministic biomechanics remain the "
        "production explainable evidence until evaluation gates are met."
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "prediction_version": self.prediction_version,
            "model_id": self.model_id,
            "analysis_id": self.analysis_id,
            "quality_dimensions": [q.to_dict() for q in self.quality_dimensions],
            "issue_probabilities": [i.to_dict() for i in self.issue_probabilities],
            "overall_quality_mean": self.overall_quality_mean,
            "overall_uncertainty": self.overall_uncertainty.to_dict(),
            "research_only": self.research_only,
            "affects_production_feedback": self.affects_production_feedback,
            "deterministic_evidence": (
                self.deterministic_evidence.to_dict()
                if self.deterministic_evidence is not None
                else None
            ),
            "notes": self.notes,
        }

    def save_json(self, path: Path) -> Path:
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path
