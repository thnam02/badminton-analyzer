"""V1 rule-based smash technique evaluation against a ReferenceProfile."""

from __future__ import annotations

import math

from app.processing.reference_profiles import (
    METRIC_ACCEL_FRACTION,
    METRIC_CONTACT_ELBOW,
    METRIC_CONTACT_WRIST_Y,
    METRIC_ELBOW_PEAK_TIMING,
    METRIC_FOLLOW_THROUGH_FRAMES,
    METRIC_FOLLOW_THROUGH_RETENTION,
    METRIC_KNEE_CONTRIBUTION,
    METRIC_PREP_KNEE,
    select_reference_profile,
)
from app.processing.technique_config import TechniqueSeverityConfig
from app.schemas.phases import SmashPhase
from app.schemas.reference import MetricReference, ReferenceEvidence, ReferenceProfile
from app.schemas.stroke_metrics import StrokeMetrics
from app.schemas.technique import (
    IssueSeverity,
    ReferenceRange,
    TechniqueEvaluation,
    TechniqueIssue,
)


def evaluate_technique(
    metrics: StrokeMetrics,
    config: TechniqueSeverityConfig | None = None,
    *,
    profile: ReferenceProfile | None = None,
    stroke_type: str = "SMASH",
    handedness: str | None = None,
    camera_view: str | None = None,
) -> TechniqueEvaluation:
    """Compare StrokeMetrics to a selected reference profile; no LLM / scoring."""
    cfg = config or TechniqueSeverityConfig()
    active = profile or select_reference_profile(
        stroke_type=stroke_type,
        handedness=handedness,
        camera_view=camera_view,
    )
    issues: list[TechniqueIssue] = []

    rules = (
        _check_elbow_extension,
        _check_knee_contribution,
        _check_preparation_knee,
        _check_acceleration_timing,
        _check_contact_posture,
        _check_follow_through,
    )
    for rule in rules:
        issue = rule(metrics, active, cfg)
        if issue is not None:
            issues.append(issue)

    return TechniqueEvaluation(
        video=metrics.video,
        issues=issues,
        confidence=_evaluation_confidence(metrics, issues),
        reference_profile_id=active.profile_id,
    )


def _evaluation_confidence(
    metrics: StrokeMetrics,
    issues: list[TechniqueIssue],
) -> float:
    base = metrics.phase_confidence
    if metrics.estimated_contact_frame_index is None:
        return 0.0
    if not issues:
        return float(max(0.2, min(0.98, base)))
    avg_issue_conf = sum(i.confidence for i in issues) / len(issues)
    return float(max(0.15, min(0.98, 0.6 * base + 0.4 * avg_issue_conf)))


def _check_elbow_extension(
    m: StrokeMetrics,
    profile: ReferenceProfile,
    cfg: TechniqueSeverityConfig,
) -> TechniqueIssue | None:
    if m.contact_elbow_angle_deg is None:
        return None
    metric = profile.get_metric(METRIC_CONTACT_ELBOW)
    if metric is None:
        return None
    if not _is_outside_band(m.contact_elbow_angle_deg, metric):
        return None
    return _make_issue(
        code="INSUFFICIENT_ELBOW_EXTENSION",
        phase=SmashPhase.ESTIMATED_CONTACT,
        measured=m.contact_elbow_angle_deg,
        metric=metric,
        profile=profile,
        cfg=cfg,
        phase_confidence=m.phase_confidence,
        description="Right elbow is not sufficiently extended at estimated contact.",
    )


def _check_knee_contribution(
    m: StrokeMetrics,
    profile: ReferenceProfile,
    cfg: TechniqueSeverityConfig,
) -> TechniqueIssue | None:
    if m.knee_contribution_deg is None:
        return None
    metric = profile.get_metric(METRIC_KNEE_CONTRIBUTION)
    if metric is None:
        return None
    if not _is_outside_band(m.knee_contribution_deg, metric):
        return None
    return _make_issue(
        code="LOW_KNEE_CONTRIBUTION",
        phase=SmashPhase.ACCELERATION,
        measured=m.knee_contribution_deg,
        metric=metric,
        profile=profile,
        cfg=cfg,
        phase_confidence=m.phase_confidence,
        description="Limited knee extension from preparation to contact.",
    )


def _check_preparation_knee(
    m: StrokeMetrics,
    profile: ReferenceProfile,
    cfg: TechniqueSeverityConfig,
) -> TechniqueIssue | None:
    """Optional prep-knee band check (percentile in_range)."""
    if m.preparation_knee_angle_deg is None:
        return None
    metric = profile.get_metric(METRIC_PREP_KNEE)
    if metric is None:
        return None
    if not _is_outside_band(m.preparation_knee_angle_deg, metric):
        return None
    return _make_issue(
        code="PREPARATION_KNEE_OUT_OF_RANGE",
        phase=SmashPhase.PREPARATION,
        measured=m.preparation_knee_angle_deg,
        metric=metric,
        profile=profile,
        cfg=cfg,
        phase_confidence=m.phase_confidence,
        description="Preparation knee angle sits outside the reference percentile band.",
    )


def _check_acceleration_timing(
    m: StrokeMetrics,
    profile: ReferenceProfile,
    cfg: TechniqueSeverityConfig,
) -> TechniqueIssue | None:
    offset = m.peak_elbow_omega_offset_frames
    frac = m.acceleration_phase_fraction
    if offset is None and frac is None:
        return None

    timing_metric = profile.get_metric(METRIC_ELBOW_PEAK_TIMING)
    frac_metric = profile.get_metric(METRIC_ACCEL_FRACTION)

    timing_bad = (
        offset is not None
        and timing_metric is not None
        and _is_outside_band(float(offset), timing_metric)
    )
    frac_bad = (
        frac is not None
        and frac_metric is not None
        and _is_outside_band(float(frac), frac_metric)
    )
    if not timing_bad and not frac_bad:
        return None

    if timing_bad and offset is not None and timing_metric is not None:
        measured = float(offset)
        metric = timing_metric
        desc = (
            "Peak elbow angular velocity is poorly timed relative to estimated contact."
        )
        unit = "frames"
    elif frac is not None and frac_metric is not None:
        measured = float(frac)
        metric = frac_metric
        desc = "Acceleration phase is too short relative to preparation-to-contact."
        unit = "ratio"
    else:
        return None

    issue = _make_issue(
        code="POOR_ARM_ACCELERATION_TIMING",
        phase=SmashPhase.ACCELERATION,
        measured=measured,
        metric=metric,
        profile=profile,
        cfg=cfg,
        phase_confidence=m.phase_confidence,
        description=desc,
    )
    # Preserve unit override for timing vs fraction.
    issue.unit = unit
    return issue


def _check_contact_posture(
    m: StrokeMetrics,
    profile: ReferenceProfile,
    cfg: TechniqueSeverityConfig,
) -> TechniqueIssue | None:
    if m.contact_wrist_y_normalized is None:
        return None
    metric = profile.get_metric(METRIC_CONTACT_WRIST_Y)
    if metric is None:
        return None
    if not _is_outside_band(m.contact_wrist_y_normalized, metric):
        return None
    return _make_issue(
        code="LOW_CONTACT_POSTURE",
        phase=SmashPhase.ESTIMATED_CONTACT,
        measured=m.contact_wrist_y_normalized,
        metric=metric,
        profile=profile,
        cfg=cfg,
        phase_confidence=m.phase_confidence,
        description="Contact point appears too low (wrist y above reference).",
    )


def _check_follow_through(
    m: StrokeMetrics,
    profile: ReferenceProfile,
    cfg: TechniqueSeverityConfig,
) -> TechniqueIssue | None:
    ratio = m.follow_through_speed_ratio
    frames = m.follow_through_frame_count
    if ratio is None and frames is None:
        return None

    ratio_metric = profile.get_metric(METRIC_FOLLOW_THROUGH_RETENTION)
    frames_metric = profile.get_metric(METRIC_FOLLOW_THROUGH_FRAMES)

    ratio_bad = (
        ratio is not None
        and ratio_metric is not None
        and _is_outside_band(float(ratio), ratio_metric)
    )
    frames_bad = (
        frames is not None
        and frames_metric is not None
        and _is_outside_band(float(frames), frames_metric)
    )
    if not ratio_bad and not frames_bad:
        return None

    if ratio_bad and ratio is not None and ratio_metric is not None:
        measured = float(ratio)
        metric = ratio_metric
        unit = "speed_ratio"
    elif frames is not None and frames_metric is not None:
        measured = float(frames)
        metric = frames_metric
        unit = "frames"
    else:
        return None

    issue = _make_issue(
        code="WEAK_FOLLOW_THROUGH",
        phase=SmashPhase.FOLLOW_THROUGH,
        measured=measured,
        metric=metric,
        profile=profile,
        cfg=cfg,
        phase_confidence=m.phase_confidence,
        description="Follow-through lacks sustained arm speed after estimated contact.",
    )
    issue.unit = unit
    return issue


def _is_outside_band(measured: float, metric: MetricReference) -> bool:
    lo = metric.lower_percentile
    hi = metric.upper_percentile
    direction = metric.direction
    if direction == "higher_is_better":
        return lo is not None and measured < lo
    if direction == "lower_is_better":
        return hi is not None and measured > hi
    # in_range
    if lo is not None and measured < lo:
        return True
    if hi is not None and measured > hi:
        return True
    return False


def _range_from_metric(metric: MetricReference) -> ReferenceRange:
    if metric.direction == "higher_is_better":
        return ReferenceRange(min=metric.lower_percentile, max=metric.upper_percentile)
    if metric.direction == "lower_is_better":
        return ReferenceRange(min=metric.lower_percentile, max=metric.upper_percentile)
    return ReferenceRange(min=metric.lower_percentile, max=metric.upper_percentile)


def _higher_is_better_flag(metric: MetricReference) -> bool | None:
    if metric.direction == "higher_is_better":
        return True
    if metric.direction == "lower_is_better":
        return False
    return None


def _evidence_from_metric(metric: MetricReference) -> ReferenceEvidence:
    return ReferenceEvidence(
        metric_id=metric.metric_id,
        median=metric.median,
        lower_percentile=metric.lower_percentile,
        upper_percentile=metric.upper_percentile,
        sample_count=metric.sample_count,
        provenance=metric.provenance,
        confidence=metric.confidence,
        provisional=metric.provisional,
        direction=metric.direction,
    )


def _make_issue(
    *,
    code: str,
    phase: SmashPhase,
    measured: float,
    metric: MetricReference,
    profile: ReferenceProfile,
    cfg: TechniqueSeverityConfig,
    phase_confidence: float,
    description: str,
) -> TechniqueIssue:
    ref = _range_from_metric(metric)
    higher = _higher_is_better_flag(metric)
    severity = _severity(measured, ref, cfg, higher_is_better=higher)
    rule_conf = _rule_confidence(measured, ref, higher)
    # Blend phase confidence, rule violation strength, and profile metric confidence.
    confidence = float(
        max(
            0.1,
            min(
                0.98,
                0.55 * phase_confidence
                + 0.25 * rule_conf
                + 0.20 * float(metric.confidence),
            ),
        )
    )
    return TechniqueIssue(
        code=code,
        phase=phase,
        severity=severity,
        confidence=confidence,
        measured_value=measured,
        reference_range=ref,
        unit=metric.unit,
        description=description,
        reference_profile_id=profile.profile_id,
        reference_evidence=_evidence_from_metric(metric),
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
