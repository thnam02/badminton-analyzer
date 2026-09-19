"""Legacy hard-coded technique evaluator (kept for C5 A/B validation).

Uses settings thresholds only — never reference distributions.
Do not use as the production default when a validated profile exists.
"""

from __future__ import annotations

from app.config import settings
from app.processing.reference_profiles import (
    METRIC_CONTACT_ELBOW,
    METRIC_CONTACT_WRIST_Y,
    METRIC_ELBOW_PEAK_TIMING,
    METRIC_FOLLOW_THROUGH_RETENTION,
    METRIC_KNEE_CONTRIBUTION,
)
from app.processing.technique_config import TechniqueSeverityConfig
from app.schemas.phases import SmashPhase
from app.schemas.stroke_metrics import StrokeMetrics
from app.schemas.technique import (
    TECHNIQUE_RULE_VERSION_LEGACY,
    IssueSeverity,
    ReferenceRange,
    TechniqueEvaluation,
    TechniqueIssue,
)
from app.schemas.technique_calibration import (
    SEVERITY_CALIBRATION_VERSION,
    IssueStatus,
)


def evaluate_technique_legacy(
    metrics: StrokeMetrics,
    config: TechniqueSeverityConfig | None = None,
    *,
    quality_confidence: float | None = None,
) -> TechniqueEvaluation:
    """Legacy hard-coded rules — preserved for coach-label comparison (C5)."""
    cfg = config or TechniqueSeverityConfig()
    phase_conf = float(metrics.phase_confidence)
    video_conf = float(quality_confidence) if quality_confidence is not None else phase_conf
    measurement_conf = phase_conf if metrics.estimated_contact_frame_index is not None else 0.0
    combined = min(measurement_conf, video_conf) if video_conf else measurement_conf

    issues: list[TechniqueIssue] = []

    def _add(
        *,
        code: str,
        phase: SmashPhase,
        measured: float | None,
        lo: float | None,
        hi: float | None,
        unit: str,
        metric_name: str,
        higher_is_better: bool | None,
        description: str,
    ) -> None:
        if measured is None:
            return
        outside = False
        if higher_is_better is True and lo is not None:
            outside = measured < lo
        elif higher_is_better is False and hi is not None:
            outside = measured > hi
        else:
            if lo is not None and measured < lo:
                outside = True
            if hi is not None and measured > hi:
                outside = True
        if not outside:
            return
        ref = ReferenceRange(min=lo, max=hi)
        # Simple legacy severity from relative violation.
        severity = IssueSeverity.LOW
        if higher_is_better is True and lo is not None:
            frac = (lo - measured) / max(abs(lo), 1e-6)
            if frac >= cfg.high_violation_fraction:
                severity = IssueSeverity.HIGH
            elif frac >= cfg.medium_violation_fraction:
                severity = IssueSeverity.MEDIUM
        elif higher_is_better is False and hi is not None:
            frac = (measured - hi) / max(abs(hi), 1e-6)
            if frac >= cfg.high_violation_fraction:
                severity = IssueSeverity.HIGH
            elif frac >= cfg.medium_violation_fraction:
                severity = IssueSeverity.MEDIUM
        status = {
            IssueSeverity.LOW: IssueStatus.MINOR,
            IssueSeverity.MEDIUM: IssueStatus.MODERATE,
            IssueSeverity.HIGH: IssueStatus.MAJOR,
        }[severity]
        issues.append(
            TechniqueIssue(
                code=code,
                phase=phase,
                severity=severity,
                confidence=float(max(0.2, min(0.9, combined))),
                measured_value=float(measured),
                reference_range=ref,
                unit=unit,
                description=description + " [legacy hard-coded rules]",
                reference_profile_id="legacy_hardcoded_v1",
                metric_name=metric_name,
                measurement_confidence=measurement_conf,
                phase_confidence=phase_conf,
                video_quality_confidence=video_conf,
                reference_confidence=0.0,
                combined_confidence=combined,
                rule_version=TECHNIQUE_RULE_VERSION_LEGACY,
                severity_calibration_version=SEVERITY_CALIBRATION_VERSION,
                decision_mode="legacy_hardcoded",
                status=status.value,
                status_reason="Legacy threshold crossing",
            )
        )

    _add(
        code="INSUFFICIENT_ELBOW_EXTENSION",
        phase=SmashPhase.ESTIMATED_CONTACT,
        measured=metrics.contact_elbow_angle_deg,
        lo=float(settings.technique_min_contact_elbow_angle_deg),
        hi=None,
        unit="deg",
        metric_name=METRIC_CONTACT_ELBOW,
        higher_is_better=True,
        description="Right elbow is not sufficiently extended at estimated contact.",
    )
    _add(
        code="LOW_KNEE_CONTRIBUTION",
        phase=SmashPhase.ACCELERATION,
        measured=metrics.knee_contribution_deg,
        lo=float(settings.technique_min_knee_contribution_deg),
        hi=None,
        unit="deg",
        metric_name=METRIC_KNEE_CONTRIBUTION,
        higher_is_better=True,
        description="Limited knee extension from preparation to contact.",
    )
    _add(
        code="POOR_ARM_ACCELERATION_TIMING",
        phase=SmashPhase.ACCELERATION,
        measured=(
            float(metrics.peak_elbow_omega_offset_frames)
            if metrics.peak_elbow_omega_offset_frames is not None
            else None
        ),
        lo=float(settings.technique_min_peak_elbow_omega_lead_frames),
        hi=float(settings.technique_max_peak_elbow_omega_lead_frames),
        unit="frames",
        metric_name=METRIC_ELBOW_PEAK_TIMING,
        higher_is_better=None,
        description="Peak elbow angular velocity is poorly timed relative to contact.",
    )
    _add(
        code="LOW_CONTACT_POSTURE",
        phase=SmashPhase.ESTIMATED_CONTACT,
        measured=metrics.contact_wrist_y_normalized,
        lo=None,
        hi=float(settings.technique_max_contact_wrist_y_normalized),
        unit="normalized_y",
        metric_name=METRIC_CONTACT_WRIST_Y,
        higher_is_better=False,
        description="Contact point appears too low (wrist y above threshold).",
    )
    _add(
        code="WEAK_FOLLOW_THROUGH",
        phase=SmashPhase.FOLLOW_THROUGH,
        measured=metrics.follow_through_speed_ratio,
        lo=float(settings.technique_min_follow_through_speed_ratio),
        hi=None,
        unit="speed_ratio",
        metric_name=METRIC_FOLLOW_THROUGH_RETENTION,
        higher_is_better=True,
        description="Follow-through lacks sustained arm speed after contact.",
    )

    return TechniqueEvaluation(
        video=metrics.video,
        issues=issues,
        confidence=combined,
        reference_profile_id="legacy_hardcoded_v1",
        reference_profile_version=TECHNIQUE_RULE_VERSION_LEGACY,
        profile_match_level="legacy",
        decision_mode="legacy_hardcoded",
        rule_version=TECHNIQUE_RULE_VERSION_LEGACY,
    )
