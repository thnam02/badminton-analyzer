"""Assemble an EvidencePackage from already-computed analysis artifacts.

Performs no biomechanics, pose estimation, or other CV calculations.
"""

from __future__ import annotations

from app.schemas.evidence import (
    CONTACT_TYPE_ESTIMATED,
    EVIDENCE_VERSION,
    STROKE_TYPE_SMASH,
    ContactEvidence,
    EvidencePackage,
)
from app.schemas.keyframes import KeyframeSet
from app.schemas.phases import PhaseSequence, SmashPhase
from app.schemas.stroke_metrics import StrokeMetrics
from app.schemas.technique import TechniqueEvaluation, TechniqueIssue
from app.schemas.video_quality import VideoQualityReport

# Maps technique issue codes → StrokeMetrics field that supplies measured_value.
# Timing / follow-through may use a secondary field when the primary is absent;
# ``issue_source_metric`` resolves that from the issue unit.
ISSUE_PRIMARY_METRIC: dict[str, str] = {
    "INSUFFICIENT_ELBOW_EXTENSION": "contact_elbow_angle_deg",
    "LOW_KNEE_CONTRIBUTION": "knee_contribution_deg",
    "PREPARATION_KNEE_OUT_OF_RANGE": "preparation_knee_angle_deg",
    "POOR_ARM_ACCELERATION_TIMING": "peak_elbow_omega_offset_frames",
    "LOW_CONTACT_POSTURE": "contact_wrist_y_normalized",
    "WEAK_FOLLOW_THROUGH": "follow_through_speed_ratio",
}


def issue_source_metric(issue: TechniqueIssue) -> str:
    """Return the StrokeMetrics attribute that backs ``issue.measured_value``."""
    if issue.code == "POOR_ARM_ACCELERATION_TIMING":
        if issue.unit == "frames":
            return "peak_elbow_omega_offset_frames"
        return "acceleration_phase_fraction"
    if issue.code == "WEAK_FOLLOW_THROUGH":
        if issue.unit == "speed_ratio":
            return "follow_through_speed_ratio"
        return "follow_through_frame_count"
    return ISSUE_PRIMARY_METRIC.get(issue.code, "")


class EvidencePackager:
    """Coaching-facing evidence assembler (pure data merge)."""

    def package(
        self,
        *,
        video_quality: VideoQualityReport,
        phases: PhaseSequence,
        metrics: StrokeMetrics,
        technique: TechniqueEvaluation,
        keyframes: KeyframeSet,
        stroke_type: str = STROKE_TYPE_SMASH,
        handedness: str | None = None,
        evidence_version: str = EVIDENCE_VERSION,
    ) -> EvidencePackage:
        video = (
            metrics.video
            or phases.video
            or technique.video
            or video_quality.video
            or keyframes.video
        )
        contact_conf = _contact_confidence(phases)
        analysis_confidence = _analysis_confidence(
            video_quality.analysis_confidence,
            phases.confidence,
            technique.confidence,
            video_quality.usable,
        )
        return EvidencePackage(
            evidence_version=evidence_version,
            video=video,
            stroke_type=stroke_type,
            handedness=handedness,
            analysis_confidence=analysis_confidence,
            video_quality=video_quality.to_dict(),
            phase_boundaries=[seg.to_dict() for seg in phases.segments],
            phase_confidence=float(phases.confidence),
            contact=ContactEvidence(
                contact_type=CONTACT_TYPE_ESTIMATED,
                confidence=contact_conf,
                frame_index=phases.estimated_contact_frame_index,
                timestamp=phases.estimated_contact_timestamp,
            ),
            metrics=metrics.to_dict(),
            technique_issues=[issue.to_dict() for issue in technique.issues],
            technique_confidence=float(technique.confidence),
            keyframes=[kf.to_dict() for kf in keyframes.keyframes],
            keyframes_output_dir=keyframes.output_dir or None,
        )


def package_evidence(
    *,
    video_quality: VideoQualityReport,
    phases: PhaseSequence,
    metrics: StrokeMetrics,
    technique: TechniqueEvaluation,
    keyframes: KeyframeSet,
    stroke_type: str = STROKE_TYPE_SMASH,
    handedness: str | None = None,
) -> EvidencePackage:
    """Module-level convenience wrapper around ``EvidencePackager``."""
    return EvidencePackager().package(
        video_quality=video_quality,
        phases=phases,
        metrics=metrics,
        technique=technique,
        keyframes=keyframes,
        stroke_type=stroke_type,
        handedness=handedness,
    )


def _contact_confidence(phases: PhaseSequence) -> float:
    for segment in phases.segments:
        if segment.phase is SmashPhase.ESTIMATED_CONTACT:
            return float(max(0.0, min(1.0, segment.confidence)))
    return float(max(0.0, min(1.0, phases.confidence)))


def _analysis_confidence(
    quality_confidence: float,
    phase_confidence: float,
    technique_confidence: float,
    usable: bool,
) -> float:
    if not usable:
        return 0.0
    parts = [
        float(quality_confidence),
        float(phase_confidence),
        float(technique_confidence),
    ]
    return float(max(0.0, min(1.0, sum(parts) / len(parts))))


evidence_packager = EvidencePackager()
