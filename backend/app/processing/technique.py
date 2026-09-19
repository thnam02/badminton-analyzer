"""Technique evaluation against selected ReferenceProfile distributions.

Uses C2/versioned profile percentiles (and optional robust z) as the primary
decision basis. Hard-coded settings thresholds are used only when
``ReferenceProfileSelector`` returns ``match_level=none`` — never mixed into
reference-based decisions.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from app.config import settings
from app.processing.reference_profiles import (
    METRIC_ACCEL_FRACTION,
    METRIC_CONTACT_ELBOW,
    METRIC_CONTACT_WRIST_Y,
    METRIC_ELBOW_PEAK_TIMING,
    METRIC_FOLLOW_THROUGH_FRAMES,
    METRIC_FOLLOW_THROUGH_RETENTION,
    METRIC_KNEE_CONTRIBUTION,
    METRIC_PREP_KNEE,
)
from app.processing.reference_profile_selector import (
    MATCH_NONE,
    ReferenceProfileSelector,
)
from app.processing.technique_config import TechniqueSeverityConfig
from app.schemas.final_analysis import FinalAnalysisState
from app.schemas.phases import SmashPhase
from app.schemas.reference import MetricReference, ReferenceEvidence, ReferenceProfile
from app.schemas.stroke_metrics import StrokeMetrics
from app.schemas.technique import (
    TECHNIQUE_RULE_VERSION_FALLBACK,
    TECHNIQUE_RULE_VERSION_REFERENCE,
    IssueSeverity,
    ReferenceRange,
    TechniqueEvaluation,
    TechniqueIssue,
)


@dataclass(frozen=True, slots=True)
class TechniqueDecisionConfig:
    """Confidence gates and distribution-comparison knobs."""

    suppress_below_confidence: float = 0.25
    uncertain_below_confidence: float = 0.45
    use_robust_z: bool = True
    robust_z_threshold: float = 1.5
    # Gate below lower percentile rank region (P10 by default on profiles).
    low_percentile_margin: float = 0.0


def evaluate_technique_from_final(
    state: FinalAnalysisState,
    metrics: StrokeMetrics,
    config: TechniqueSeverityConfig | None = None,
    *,
    profile: ReferenceProfile | None = None,
    profiles: Sequence[ReferenceProfile] | None = None,
    stroke_type: str = "SMASH",
    handedness: str | None = None,
    camera_view: str | None = None,
    skill_level: str | None = None,
    quality_confidence: float | None = None,
    decision_config: TechniqueDecisionConfig | None = None,
) -> TechniqueEvaluation:
    """Evaluate technique using metrics derived from ``FinalAnalysisState``."""
    if metrics.estimated_contact_frame_index != state.contact.frame_index:
        raise ValueError(
            "StrokeMetrics contact frame must match FinalAnalysisState.contact; "
            f"got metrics={metrics.estimated_contact_frame_index}, "
            f"state={state.contact.frame_index}."
        )
    q_conf = quality_confidence
    if q_conf is None:
        q_conf = float(state.video_quality.analysis_confidence)
    return evaluate_technique(
        metrics,
        config,
        profile=profile,
        profiles=profiles,
        stroke_type=stroke_type,
        handedness=handedness,
        camera_view=camera_view,
        skill_level=skill_level,
        quality_confidence=q_conf,
        decision_config=decision_config,
    )


def evaluate_technique(
    metrics: StrokeMetrics,
    config: TechniqueSeverityConfig | None = None,
    *,
    profile: ReferenceProfile | None = None,
    profiles: Sequence[ReferenceProfile] | None = None,
    stroke_type: str = "SMASH",
    handedness: str | None = None,
    camera_view: str | None = None,
    skill_level: str | None = None,
    quality_confidence: float | None = None,
    decision_config: TechniqueDecisionConfig | None = None,
) -> TechniqueEvaluation:
    """Compare StrokeMetrics to a selected reference distribution; no LLM."""
    cfg = config or TechniqueSeverityConfig()
    decision = decision_config or TechniqueDecisionConfig()

    if profile is not None:
        selection_profile = profile
        match_level = "explicit"
        decision_mode = "reference_distribution"
        rule_version = TECHNIQUE_RULE_VERSION_REFERENCE
    else:
        selection = ReferenceProfileSelector(profiles).select(
            stroke_type=stroke_type,
            handedness=handedness,
            camera_view=camera_view,
            skill_level=skill_level,
        )
        if selection.has_valid_profile and selection.profile is not None:
            selection_profile = selection.profile
            match_level = selection.match_level
            decision_mode = "reference_distribution"
            rule_version = TECHNIQUE_RULE_VERSION_REFERENCE
        else:
            selection_profile = None
            match_level = MATCH_NONE
            decision_mode = "provisional_fallback"
            rule_version = TECHNIQUE_RULE_VERSION_FALLBACK

    measurement_confidence = _measurement_confidence(
        metrics, quality_confidence=quality_confidence
    )

    if decision_mode == "reference_distribution" and selection_profile is not None:
        issues = _evaluate_against_profile(
            metrics,
            selection_profile,
            cfg,
            decision,
            measurement_confidence=measurement_confidence,
            rule_version=rule_version,
        )
        profile_id = selection_profile.profile_id
    else:
        issues = _evaluate_provisional_fallback(
            metrics,
            cfg,
            decision,
            measurement_confidence=measurement_confidence,
        )
        profile_id = "provisional_fallback_hardcoded_v1"

    return TechniqueEvaluation(
        video=metrics.video,
        issues=issues,
        confidence=_evaluation_confidence(metrics, issues, measurement_confidence),
        reference_profile_id=profile_id,
        profile_match_level=match_level,
        decision_mode=decision_mode,
        rule_version=rule_version,
    )


def _measurement_confidence(
    metrics: StrokeMetrics,
    *,
    quality_confidence: float | None,
) -> float:
    parts = [float(metrics.phase_confidence)]
    if quality_confidence is not None:
        parts.append(float(max(0.0, min(1.0, quality_confidence))))
    if metrics.estimated_contact_frame_index is None:
        return 0.0
    return float(sum(parts) / len(parts))


def _evaluation_confidence(
    metrics: StrokeMetrics,
    issues: list[TechniqueIssue],
    measurement_confidence: float,
) -> float:
    if metrics.estimated_contact_frame_index is None:
        return 0.0
    if not issues:
        return float(max(0.2, min(0.98, measurement_confidence)))
    avg_issue_conf = sum(i.confidence for i in issues) / len(issues)
    return float(
        max(0.15, min(0.98, 0.55 * measurement_confidence + 0.45 * avg_issue_conf))
    )


# ---------------------------------------------------------------------------
# Reference-distribution path
# ---------------------------------------------------------------------------


def _evaluate_against_profile(
    m: StrokeMetrics,
    profile: ReferenceProfile,
    cfg: TechniqueSeverityConfig,
    decision: TechniqueDecisionConfig,
    *,
    measurement_confidence: float,
    rule_version: str,
) -> list[TechniqueIssue]:
    issues: list[TechniqueIssue] = []
    checks = (
        (
            "INSUFFICIENT_ELBOW_EXTENSION",
            SmashPhase.ESTIMATED_CONTACT,
            m.contact_elbow_angle_deg,
            METRIC_CONTACT_ELBOW,
            "Right elbow is not sufficiently extended at estimated contact.",
        ),
        (
            "LOW_KNEE_CONTRIBUTION",
            SmashPhase.ACCELERATION,
            m.knee_contribution_deg,
            METRIC_KNEE_CONTRIBUTION,
            "Limited knee extension from preparation to contact.",
        ),
        (
            "PREPARATION_KNEE_OUT_OF_RANGE",
            SmashPhase.PREPARATION,
            m.preparation_knee_angle_deg,
            METRIC_PREP_KNEE,
            "Preparation knee angle sits outside the reference percentile band.",
        ),
        (
            "LOW_CONTACT_POSTURE",
            SmashPhase.ESTIMATED_CONTACT,
            m.contact_wrist_y_normalized,
            METRIC_CONTACT_WRIST_Y,
            "Contact point appears too low (wrist y above reference).",
        ),
    )
    for code, phase, measured, metric_id, description in checks:
        if measured is None:
            continue
        metric = profile.get_metric(metric_id)
        if metric is None:
            continue
        issue = _maybe_issue_from_distribution(
            code=code,
            phase=phase,
            measured=float(measured),
            metric=metric,
            profile=profile,
            cfg=cfg,
            decision=decision,
            measurement_confidence=measurement_confidence,
            rule_version=rule_version,
            description=description,
        )
        if issue is not None:
            issues.append(issue)

    # Timing: compare elbow-to-wrist peak delay against reference distribution.
    timing_issue = _check_acceleration_timing_distribution(
        m, profile, cfg, decision, measurement_confidence, rule_version
    )
    if timing_issue is not None:
        issues.append(timing_issue)

    follow_issue = _check_follow_through_distribution(
        m, profile, cfg, decision, measurement_confidence, rule_version
    )
    if follow_issue is not None:
        issues.append(follow_issue)

    return issues


def _check_acceleration_timing_distribution(
    m: StrokeMetrics,
    profile: ReferenceProfile,
    cfg: TechniqueSeverityConfig,
    decision: TechniqueDecisionConfig,
    measurement_confidence: float,
    rule_version: str,
) -> TechniqueIssue | None:
    offset = m.peak_elbow_omega_offset_frames
    frac = m.acceleration_phase_fraction
    timing_metric = profile.get_metric(METRIC_ELBOW_PEAK_TIMING)
    frac_metric = profile.get_metric(METRIC_ACCEL_FRACTION)

    candidates: list[tuple[float, MetricReference, str, str]] = []
    if offset is not None and timing_metric is not None:
        candidates.append(
            (
                float(offset),
                timing_metric,
                "Peak elbow angular velocity is poorly timed relative to estimated contact.",
                "frames",
            )
        )
    if frac is not None and frac_metric is not None:
        candidates.append(
            (
                float(frac),
                frac_metric,
                "Acceleration phase is too short relative to preparation-to-contact.",
                "ratio",
            )
        )

    for measured, metric, desc, unit in candidates:
        if not _is_outside_distribution(measured, metric, decision):
            continue
        issue = _maybe_issue_from_distribution(
            code="POOR_ARM_ACCELERATION_TIMING",
            phase=SmashPhase.ACCELERATION,
            measured=measured,
            metric=metric,
            profile=profile,
            cfg=cfg,
            decision=decision,
            measurement_confidence=measurement_confidence,
            rule_version=rule_version,
            description=desc,
        )
        if issue is not None:
            issue.unit = unit
            return issue
    return None


def _check_follow_through_distribution(
    m: StrokeMetrics,
    profile: ReferenceProfile,
    cfg: TechniqueSeverityConfig,
    decision: TechniqueDecisionConfig,
    measurement_confidence: float,
    rule_version: str,
) -> TechniqueIssue | None:
    ratio = m.follow_through_speed_ratio
    frames = m.follow_through_frame_count
    ratio_metric = profile.get_metric(METRIC_FOLLOW_THROUGH_RETENTION)
    frames_metric = profile.get_metric(METRIC_FOLLOW_THROUGH_FRAMES)

    candidates: list[tuple[float, MetricReference, str]] = []
    if ratio is not None and ratio_metric is not None:
        candidates.append((float(ratio), ratio_metric, "speed_ratio"))
    if frames is not None and frames_metric is not None:
        candidates.append((float(frames), frames_metric, "frames"))

    for measured, metric, unit in candidates:
        if not _is_outside_distribution(measured, metric, decision):
            continue
        issue = _maybe_issue_from_distribution(
            code="WEAK_FOLLOW_THROUGH",
            phase=SmashPhase.FOLLOW_THROUGH,
            measured=measured,
            metric=metric,
            profile=profile,
            cfg=cfg,
            decision=decision,
            measurement_confidence=measurement_confidence,
            rule_version=rule_version,
            description=(
                "Follow-through lacks sustained arm speed after estimated contact."
            ),
        )
        if issue is not None:
            issue.unit = unit
            return issue
    return None


def _is_outside_distribution(
    measured: float,
    metric: MetricReference,
    decision: TechniqueDecisionConfig,
) -> bool:
    """Percentile-band gate, with optional robust-z adverse detection."""
    lo = metric.lower_percentile
    hi = metric.upper_percentile
    direction = metric.direction
    outside = False
    if direction == "higher_is_better":
        outside = lo is not None and measured < (lo - decision.low_percentile_margin)
    elif direction == "lower_is_better":
        outside = hi is not None and measured > (hi + decision.low_percentile_margin)
    else:
        if lo is not None and measured < (lo - decision.low_percentile_margin):
            outside = True
        if hi is not None and measured > (hi + decision.low_percentile_margin):
            outside = True
    if outside:
        return True

    if not decision.use_robust_z:
        return False
    z = _robust_z(measured, metric)
    if z is None:
        return False
    if direction == "higher_is_better":
        return z <= -decision.robust_z_threshold
    if direction == "lower_is_better":
        return z >= decision.robust_z_threshold
    return abs(z) >= decision.robust_z_threshold


def _robust_z(measured: float, metric: MetricReference) -> float | None:
    center = metric.median if metric.median is not None else metric.mean
    if center is None:
        return None
    scale = None
    if metric.std is not None and metric.std > 1e-9:
        scale = float(metric.std)
    elif metric.iqr is not None and metric.iqr > 1e-9:
        scale = float(metric.iqr) / 1.349
    elif (
        metric.lower_percentile is not None
        and metric.upper_percentile is not None
        and metric.upper_percentile != metric.lower_percentile
    ):
        scale = abs(metric.upper_percentile - metric.lower_percentile) / 2.56
    if scale is None or scale <= 1e-9:
        return None
    return float((measured - center) / scale)


def _estimate_percentile_position(measured: float, metric: MetricReference) -> float | None:
    """Piecewise-linear CDF estimate through P10 / median / P90."""
    p10 = metric.lower_percentile
    p50 = metric.median
    p90 = metric.upper_percentile
    if p10 is None or p50 is None or p90 is None:
        z = _robust_z(measured, metric)
        if z is None:
            return None
        return float(max(0.0, min(100.0, 100.0 / (1.0 + math.exp(-1.7 * z)))))

    if measured <= p10:
        return float(max(0.0, min(10.0, 10.0 * measured / max(abs(p10), 1e-6))))
    if measured <= p50:
        return float(10.0 + 40.0 * (measured - p10) / max(p50 - p10, 1e-6))
    if measured <= p90:
        return float(50.0 + 40.0 * (measured - p50) / max(p90 - p50, 1e-6))
    return float(min(100.0, 90.0 + 10.0 * (measured - p90) / max(p90 - p50, 1e-6)))


def _deviation_from_median(measured: float, metric: MetricReference) -> float | None:
    if metric.median is None:
        return None
    return float(measured - metric.median)


def _maybe_issue_from_distribution(
    *,
    code: str,
    phase: SmashPhase,
    measured: float,
    metric: MetricReference,
    profile: ReferenceProfile,
    cfg: TechniqueSeverityConfig,
    decision: TechniqueDecisionConfig,
    measurement_confidence: float,
    rule_version: str,
    description: str,
) -> TechniqueIssue | None:
    if measurement_confidence < decision.suppress_below_confidence:
        return None
    if not _is_outside_distribution(measured, metric, decision):
        return None

    uncertain = measurement_confidence < decision.uncertain_below_confidence
    ref = _range_from_metric(metric)
    higher = _higher_is_better_flag(metric)
    severity = _severity(measured, ref, cfg, higher_is_better=higher)
    if uncertain:
        severity = IssueSeverity.LOW

    deviation = _deviation_from_median(measured, metric)
    percentile_position = _estimate_percentile_position(measured, metric)
    robust_z = _robust_z(measured, metric)
    rule_conf = _rule_confidence(measured, ref, higher)
    confidence = float(
        max(
            0.1,
            min(
                0.98,
                0.50 * measurement_confidence
                + 0.30 * rule_conf
                + 0.20 * float(metric.confidence),
            ),
        )
    )
    if uncertain:
        confidence = min(confidence, 0.45)

    return TechniqueIssue(
        code=code,
        phase=phase,
        severity=severity,
        confidence=confidence,
        measured_value=measured,
        reference_range=ref,
        unit=metric.unit,
        description=description
        + (" (uncertain — low measurement confidence)" if uncertain else ""),
        reference_profile_id=profile.profile_id,
        reference_evidence=_evidence_from_metric(
            metric,
            deviation=deviation,
            percentile_position=percentile_position,
            robust_z=robust_z,
        ),
        metric_name=metric.metric_id,
        reference_median=metric.median,
        deviation=deviation,
        percentile_position=percentile_position,
        measurement_confidence=measurement_confidence,
        rule_version=rule_version,
        decision_mode="reference_distribution",
        uncertain=uncertain,
    )


# ---------------------------------------------------------------------------
# Provisional hard-coded fallback (only when no profile selected)
# ---------------------------------------------------------------------------


def _evaluate_provisional_fallback(
    m: StrokeMetrics,
    cfg: TechniqueSeverityConfig,
    decision: TechniqueDecisionConfig,
    *,
    measurement_confidence: float,
) -> list[TechniqueIssue]:
    """Clearly marked hard-coded thresholds — never mixed with profile bands."""
    if measurement_confidence < decision.suppress_below_confidence:
        return []

    uncertain = measurement_confidence < decision.uncertain_below_confidence
    issues: list[TechniqueIssue] = []

    def _add(
        *,
        code: str,
        phase: SmashPhase,
        measured: float | None,
        threshold_min: float | None,
        threshold_max: float | None,
        unit: str,
        metric_name: str,
        higher_is_better: bool | None,
        description: str,
    ) -> None:
        if measured is None:
            return
        outside = False
        if higher_is_better is True and threshold_min is not None:
            outside = measured < threshold_min
        elif higher_is_better is False and threshold_max is not None:
            outside = measured > threshold_max
        else:
            if threshold_min is not None and measured < threshold_min:
                outside = True
            if threshold_max is not None and measured > threshold_max:
                outside = True
        if not outside:
            return
        ref = ReferenceRange(min=threshold_min, max=threshold_max)
        severity = _severity(measured, ref, cfg, higher_is_better=higher_is_better)
        if uncertain:
            severity = IssueSeverity.LOW
        deviation = (
            measured - (threshold_min if higher_is_better is True else (threshold_max or 0.0))
        )
        conf = float(max(0.1, min(0.85, 0.7 * measurement_confidence + 0.2)))
        if uncertain:
            conf = min(conf, 0.45)
        issues.append(
            TechniqueIssue(
                code=code,
                phase=phase,
                severity=severity,
                confidence=conf,
                measured_value=float(measured),
                reference_range=ref,
                unit=unit,
                description=(
                    description
                    + " [provisional fallback thresholds — no reference profile]"
                    + (" (uncertain)" if uncertain else "")
                ),
                reference_profile_id="provisional_fallback_hardcoded_v1",
                reference_evidence=None,
                metric_name=metric_name,
                reference_median=None,
                deviation=float(deviation),
                percentile_position=None,
                measurement_confidence=measurement_confidence,
                rule_version=TECHNIQUE_RULE_VERSION_FALLBACK,
                decision_mode="provisional_fallback",
                uncertain=uncertain,
            )
        )

    _add(
        code="INSUFFICIENT_ELBOW_EXTENSION",
        phase=SmashPhase.ESTIMATED_CONTACT,
        measured=m.contact_elbow_angle_deg,
        threshold_min=float(settings.technique_min_contact_elbow_angle_deg),
        threshold_max=None,
        unit="deg",
        metric_name=METRIC_CONTACT_ELBOW,
        higher_is_better=True,
        description="Right elbow is not sufficiently extended at estimated contact.",
    )
    _add(
        code="LOW_KNEE_CONTRIBUTION",
        phase=SmashPhase.ACCELERATION,
        measured=m.knee_contribution_deg,
        threshold_min=float(settings.technique_min_knee_contribution_deg),
        threshold_max=None,
        unit="deg",
        metric_name=METRIC_KNEE_CONTRIBUTION,
        higher_is_better=True,
        description="Limited knee extension from preparation to contact.",
    )
    _add(
        code="POOR_ARM_ACCELERATION_TIMING",
        phase=SmashPhase.ACCELERATION,
        measured=(
            float(m.peak_elbow_omega_offset_frames)
            if m.peak_elbow_omega_offset_frames is not None
            else None
        ),
        threshold_min=float(settings.technique_min_peak_elbow_omega_lead_frames),
        threshold_max=float(settings.technique_max_peak_elbow_omega_lead_frames),
        unit="frames",
        metric_name=METRIC_ELBOW_PEAK_TIMING,
        higher_is_better=None,
        description="Peak elbow angular velocity is poorly timed relative to contact.",
    )
    _add(
        code="LOW_CONTACT_POSTURE",
        phase=SmashPhase.ESTIMATED_CONTACT,
        measured=m.contact_wrist_y_normalized,
        threshold_min=None,
        threshold_max=float(settings.technique_max_contact_wrist_y_normalized),
        unit="normalized_y",
        metric_name=METRIC_CONTACT_WRIST_Y,
        higher_is_better=False,
        description="Contact point appears too low (wrist y above threshold).",
    )
    _add(
        code="WEAK_FOLLOW_THROUGH",
        phase=SmashPhase.FOLLOW_THROUGH,
        measured=m.follow_through_speed_ratio,
        threshold_min=float(settings.technique_min_follow_through_speed_ratio),
        threshold_max=None,
        unit="speed_ratio",
        metric_name=METRIC_FOLLOW_THROUGH_RETENTION,
        higher_is_better=True,
        description="Follow-through lacks sustained arm speed after contact.",
    )
    return issues


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _range_from_metric(metric: MetricReference) -> ReferenceRange:
    return ReferenceRange(min=metric.lower_percentile, max=metric.upper_percentile)


def _higher_is_better_flag(metric: MetricReference) -> bool | None:
    if metric.direction == "higher_is_better":
        return True
    if metric.direction == "lower_is_better":
        return False
    return None


def _evidence_from_metric(
    metric: MetricReference,
    *,
    deviation: float | None = None,
    percentile_position: float | None = None,
    robust_z: float | None = None,
) -> ReferenceEvidence:
    return ReferenceEvidence(
        metric_id=metric.metric_id,
        median=metric.median,
        mean=metric.mean,
        std=metric.std,
        iqr=metric.iqr,
        lower_percentile=metric.lower_percentile,
        upper_percentile=metric.upper_percentile,
        lower_percentile_rank=metric.lower_percentile_rank,
        upper_percentile_rank=metric.upper_percentile_rank,
        sample_count=metric.sample_count,
        provenance=metric.provenance,
        confidence=metric.confidence,
        provisional=metric.provisional,
        direction=metric.direction,
        deviation=deviation,
        percentile_position=percentile_position,
        robust_z=robust_z,
    )


def _severity(
    measured: float,
    ref: ReferenceRange,
    cfg: TechniqueSeverityConfig,
    *,
    higher_is_better: bool | None,
) -> IssueSeverity:
    violation = _violation_fraction(measured, ref, higher_is_better=higher_is_better)
    if violation >= cfg.high_violation_fraction:
        return IssueSeverity.HIGH
    if violation >= cfg.medium_violation_fraction:
        return IssueSeverity.MEDIUM
    return IssueSeverity.LOW


def _violation_fraction(
    measured: float,
    ref: ReferenceRange,
    *,
    higher_is_better: bool | None,
) -> float:
    if higher_is_better is True and ref.min is not None and measured < ref.min:
        span = max(abs(ref.min), 1e-6)
        return (ref.min - measured) / span
    if higher_is_better is False and ref.max is not None and measured > ref.max:
        span = max(abs(ref.max), 1e-6)
        return (measured - ref.max) / span
    if higher_is_better is None:
        if ref.min is not None and measured < ref.min:
            span = max(
                abs(ref.max - ref.min) if ref.max is not None else abs(ref.min),
                1e-6,
            )
            return (ref.min - measured) / span
        if ref.max is not None and measured > ref.max:
            span = max(
                abs(ref.max - ref.min) if ref.min is not None else abs(ref.max),
                1e-6,
            )
            return (measured - ref.max) / span
    return 0.0


def _rule_confidence(
    measured: float,
    ref: ReferenceRange,
    higher_is_better: bool | None,
) -> float:
    frac = _violation_fraction(measured, ref, higher_is_better=higher_is_better)
    if not math.isfinite(frac):
        return 0.5
    return float(max(0.35, min(0.95, 0.5 + frac)))
