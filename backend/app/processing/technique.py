"""V1 rule-based smash technique evaluation on StrokeMetrics."""

from __future__ import annotations

import math

from app.processing.technique_config import TechniqueRuleConfig
from app.schemas.phases import SmashPhase
from app.schemas.stroke_metrics import StrokeMetrics
from app.schemas.technique import (
    IssueSeverity,
    ReferenceRange,
    TechniqueEvaluation,
    TechniqueIssue,
)


def evaluate_technique(
    metrics: StrokeMetrics,
    config: TechniqueRuleConfig | None = None,
) -> TechniqueEvaluation:
    """Run configurable smash technique rules; no LLM / scoring."""
    cfg = config or TechniqueRuleConfig()
    issues: list[TechniqueIssue] = []

    rules = (
        _check_elbow_extension,
        _check_knee_contribution,
        _check_acceleration_timing,
        _check_contact_posture,
        _check_follow_through,
    )
    for rule in rules:
        issue = rule(metrics, cfg)
        if issue is not None:
            issues.append(issue)

    return TechniqueEvaluation(
        video=metrics.video,
        issues=issues,
        confidence=_evaluation_confidence(metrics, issues),
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
    cfg: TechniqueRuleConfig,
) -> TechniqueIssue | None:
    if m.contact_elbow_angle_deg is None:
        return None
    ref = ReferenceRange(min=cfg.min_contact_elbow_angle_deg, max=180.0)
    if m.contact_elbow_angle_deg >= ref.min:
        return None
    return _make_issue(
        code="INSUFFICIENT_ELBOW_EXTENSION",
        phase=SmashPhase.ESTIMATED_CONTACT,
        measured=m.contact_elbow_angle_deg,
        ref=ref,
        unit="deg",
        cfg=cfg,
        phase_confidence=m.phase_confidence,
        description="Right elbow is not sufficiently extended at estimated contact.",
        higher_is_better=True,
    )


def _check_knee_contribution(
    m: StrokeMetrics,
    cfg: TechniqueRuleConfig,
) -> TechniqueIssue | None:
    if m.knee_contribution_deg is None:
        return None
    ref = ReferenceRange(min=cfg.min_knee_contribution_deg, max=None)
    if m.knee_contribution_deg >= ref.min:
        return None
    return _make_issue(
        code="LOW_KNEE_CONTRIBUTION",
        phase=SmashPhase.ACCELERATION,
        measured=m.knee_contribution_deg,
        ref=ref,
        unit="deg",
        cfg=cfg,
        phase_confidence=m.phase_confidence,
        description="Limited knee extension from preparation to contact.",
        higher_is_better=True,
    )


def _check_acceleration_timing(
    m: StrokeMetrics,
    cfg: TechniqueRuleConfig,
) -> TechniqueIssue | None:
    offset = m.peak_elbow_omega_offset_frames
    frac = m.acceleration_phase_fraction
    if offset is None and frac is None:
        return None

    timing_bad = offset is not None and (
        offset > cfg.max_peak_elbow_omega_lead_frames
        or offset < cfg.min_peak_elbow_omega_lead_frames
    )
    frac_bad = frac is not None and frac < cfg.min_acceleration_phase_fraction

    if not timing_bad and not frac_bad:
        return None

    # Prefer offset as primary measured value when available.
    if offset is not None:
        measured = float(offset)
        ref = ReferenceRange(
            min=float(cfg.min_peak_elbow_omega_lead_frames),
            max=float(cfg.max_peak_elbow_omega_lead_frames),
        )
        desc = (
            "Peak elbow angular velocity is poorly timed relative to estimated contact."
        )
    else:
        measured = frac  # type: ignore[assignment]
        ref = ReferenceRange(min=cfg.min_acceleration_phase_fraction, max=1.0)
        desc = "Acceleration phase is too short relative to preparation-to-contact."

    return _make_issue(
        code="POOR_ARM_ACCELERATION_TIMING",
        phase=SmashPhase.ACCELERATION,
        measured=measured,
        ref=ref,
        unit="frames" if offset is not None else "ratio",
        cfg=cfg,
        phase_confidence=m.phase_confidence,
        description=desc,
        higher_is_better=None,
    )


def _check_contact_posture(
    m: StrokeMetrics,
    cfg: TechniqueRuleConfig,
) -> TechniqueIssue | None:
    if m.contact_wrist_y_normalized is None:
        return None
    ref = ReferenceRange(min=0.0, max=cfg.max_contact_wrist_y_normalized)
    if m.contact_wrist_y_normalized <= ref.max:
        return None
    return _make_issue(
        code="LOW_CONTACT_POSTURE",
        phase=SmashPhase.ESTIMATED_CONTACT,
        measured=m.contact_wrist_y_normalized,
        ref=ref,
        unit="normalized_y",
        cfg=cfg,
        phase_confidence=m.phase_confidence,
        description="Contact point appears too low (wrist y above reference).",
        higher_is_better=False,
    )


def _check_follow_through(
    m: StrokeMetrics,
    cfg: TechniqueRuleConfig,
) -> TechniqueIssue | None:
    ratio = m.follow_through_speed_ratio
    frames = m.follow_through_frame_count
    if ratio is None and frames is None:
        return None

    ratio_bad = ratio is not None and ratio < cfg.min_follow_through_speed_ratio
    frames_bad = frames is not None and frames < cfg.min_follow_through_frames
    if not ratio_bad and not frames_bad:
        return None

    measured = ratio if ratio is not None else float(frames or 0)
    ref = ReferenceRange(
        min=cfg.min_follow_through_speed_ratio
        if ratio is not None
        else float(cfg.min_follow_through_frames),
        max=1.0 if ratio is not None else None,
    )
    return _make_issue(
        code="WEAK_FOLLOW_THROUGH",
        phase=SmashPhase.FOLLOW_THROUGH,
        measured=measured,
        ref=ref,
        unit="speed_ratio" if ratio is not None else "frames",
        cfg=cfg,
        phase_confidence=m.phase_confidence,
        description="Follow-through lacks sustained arm speed after estimated contact.",
        higher_is_better=True,
    )


def _make_issue(
    *,
    code: str,
    phase: SmashPhase,
    measured: float,
    ref: ReferenceRange,
    unit: str,
    cfg: TechniqueRuleConfig,
    phase_confidence: float,
    description: str,
    higher_is_better: bool | None,
) -> TechniqueIssue:
    severity = _severity(measured, ref, cfg, higher_is_better=higher_is_better)
    rule_conf = _rule_confidence(measured, ref, higher_is_better)
    confidence = float(max(0.1, min(0.98, 0.65 * phase_confidence + 0.35 * rule_conf)))
    return TechniqueIssue(
        code=code,
        phase=phase,
        severity=severity,
        confidence=confidence,
        measured_value=measured,
        reference_range=ref,
        unit=unit,
        description=description,
    )


def _severity(
    measured: float,
    ref: ReferenceRange,
    cfg: TechniqueRuleConfig,
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
            span = max(abs(ref.max - ref.min) if ref.max is not None else abs(ref.min), 1e-6)
            return (ref.min - measured) / span
        if ref.max is not None and measured > ref.max:
            span = max(abs(ref.max - ref.min) if ref.min is not None else abs(ref.max), 1e-6)
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
