"""Forehand clear technique evaluation (provisional reference distributions)."""

from __future__ import annotations

from collections.abc import Sequence

from app.processing.reference_profile_selector import (
    MATCH_NONE,
    ReferenceProfileSelector,
)
from app.processing.strokes.clear.reference_profiles import (
    METRIC_CLEAR_BACKSWING_ELBOW,
    METRIC_CLEAR_CONTACT_ELBOW,
    METRIC_CLEAR_CONTACT_WRIST_Y,
    METRIC_CLEAR_ELBOW_TIMING,
    METRIC_CLEAR_FOLLOW_RATIO,
    METRIC_CLEAR_PREP_ELBOW,
    METRIC_CLEAR_PREP_KNEE,
    METRIC_CLEAR_RECOVERY_FRAMES,
    default_clear_reference_profiles,
)
from app.processing.technique import (
    TechniqueDecisionConfig,
    _ConfidenceContext,
    _maybe_issue_from_distribution,
    _profile_reference_confidence,
)
from app.schemas.clear_metrics import ForehandClearMetrics
from app.schemas.final_analysis import FinalAnalysisState
from app.schemas.phases import SmashPhase
from app.schemas.reference import ReferenceProfile
from app.schemas.technique import (
    TECHNIQUE_RULE_VERSION_FALLBACK,
    TechniqueEvaluation,
    TechniqueIssue,
)
from app.schemas.technique_calibration import (
    IssueStatus,
    build_evaluation_confidence,
    default_severity_calibration,
)
from app.schemas.technique import IssueSeverity, ReferenceRange

CLEAR_RULE_VERSION = "clear_reference_distribution_v1"


def evaluate_clear_technique(
    metrics: ForehandClearMetrics,
    *,
    profile: ReferenceProfile | None = None,
    profiles: Sequence[ReferenceProfile] | None = None,
    handedness: str | None = None,
    camera_view: str | None = None,
    skill_level: str | None = None,
    quality_confidence: float | None = None,
    pose_confidence: float | None = None,
    decision_config: TechniqueDecisionConfig | None = None,
) -> TechniqueEvaluation:
    """Compare clear metrics to clear-only reference profiles."""
    decision = decision_config or TechniqueDecisionConfig()
    calib = decision.severity_calibration or default_severity_calibration()
    catalog = (
        list(profiles)
        if profiles is not None
        else default_clear_reference_profiles()
    )

    if profile is not None:
        if str(profile.stroke_type).upper() not in {"FOREHAND_CLEAR", "CLEAR"}:
            raise ValueError(
                f"Clear evaluator cannot use non-clear profile "
                f"'{profile.profile_id}' (stroke_type={profile.stroke_type})"
            )
        selection_profile = profile
        match_level = "explicit"
        decision_mode = "reference_distribution"
        rule_version = CLEAR_RULE_VERSION
    else:
        selection = ReferenceProfileSelector(catalog).select(
            stroke_type="FOREHAND_CLEAR",
            handedness=handedness,
            camera_view=camera_view,
            skill_level=skill_level,
        )
        if selection.has_valid_profile and selection.profile is not None:
            selection_profile = selection.profile
            match_level = selection.match_level
            decision_mode = "reference_distribution"
            rule_version = CLEAR_RULE_VERSION
        else:
            selection_profile = None
            match_level = MATCH_NONE
            decision_mode = "provisional_fallback"
            rule_version = TECHNIQUE_RULE_VERSION_FALLBACK

    video_quality_confidence = (
        float(max(0.0, min(1.0, quality_confidence)))
        if quality_confidence is not None
        else 1.0
    )
    pose_conf = (
        float(max(0.0, min(1.0, pose_confidence)))
        if pose_confidence is not None
        else float(decision.pose_confidence_default)
    )
    # Adapt clear metrics to the measurement-confidence helper shape.
    phase_confidence = float(metrics.phase_confidence)
    measurement_confidence = 0.0
    if metrics.estimated_contact_frame_index is not None:
        parts = [phase_confidence, pose_conf]
        if quality_confidence is not None:
            parts.append(float(max(0.0, min(1.0, quality_confidence))))
        # Contact confidence is clear-specific signal.
        parts.append(float(max(0.0, min(1.0, metrics.contact_confidence or 0.0))))
        measurement_confidence = float(sum(parts) / len(parts))

    context = _ConfidenceContext(
        measurement_confidence=measurement_confidence,
        phase_confidence=phase_confidence,
        video_quality_confidence=video_quality_confidence,
        pose_confidence=pose_conf,
        calibration=calib,
    )

    if selection_profile is not None:
        issues = _evaluate_clear_against_profile(
            metrics, selection_profile, decision, context, rule_version
        )
        profile_id = selection_profile.profile_id
        profile_version = selection_profile.profile_version
        ref_conf = _profile_reference_confidence(selection_profile)
    else:
        issues = _clear_provisional_fallback(metrics, context, calib)
        profile_id = "provisional_clear_fallback_v1"
        profile_version = "v1"
        ref_conf = 0.2

    eval_confidence = build_evaluation_confidence(
        measurement_confidence=measurement_confidence,
        phase_confidence=phase_confidence,
        video_quality_confidence=video_quality_confidence,
        reference_confidence=ref_conf,
        pose_confidence=pose_conf,
        config=calib,
    )

    return TechniqueEvaluation(
        video=metrics.video,
        issues=issues,
        confidence=float(eval_confidence.combined_confidence),
        reference_profile_id=profile_id,
        reference_profile_version=profile_version,
        profile_match_level=match_level,
        decision_mode=decision_mode,
        rule_version=rule_version,
        severity_calibration_version=calib.version,
        evaluation_confidence=eval_confidence,
    )


def evaluate_clear_from_final(
    state: FinalAnalysisState,
    metrics: ForehandClearMetrics,
    **kwargs,
) -> TechniqueEvaluation:
    if metrics.estimated_contact_frame_index != state.contact.frame_index:
        raise ValueError(
            "Clear metrics contact frame must match FinalAnalysisState.contact"
        )
    q = kwargs.pop("quality_confidence", None)
    if q is None:
        q = float(state.video_quality.analysis_confidence)
    return evaluate_clear_technique(metrics, quality_confidence=q, **kwargs)


def _evaluate_clear_against_profile(
    m: ForehandClearMetrics,
    profile: ReferenceProfile,
    decision: TechniqueDecisionConfig,
    context: _ConfidenceContext,
    rule_version: str,
) -> list[TechniqueIssue]:
    from app.processing.technique_config import TechniqueSeverityConfig

    cfg = TechniqueSeverityConfig()
    checks = (
        (
            "LIMITED_CLEAR_PREPARATION",
            SmashPhase.PREPARATION,
            m.preparation_elbow_angle_deg,
            METRIC_CLEAR_PREP_ELBOW,
            "Preparation elbow position sits outside the clear reference band.",
        ),
        (
            "LIMITED_CLEAR_PREPARATION",
            SmashPhase.PREPARATION,
            m.preparation_knee_angle_deg,
            METRIC_CLEAR_PREP_KNEE,
            "Preparation knee loading sits outside the clear reference band.",
        ),
        (
            "INSUFFICIENT_ARM_EXTENSION",
            SmashPhase.ESTIMATED_CONTACT,
            m.contact_elbow_angle_deg,
            METRIC_CLEAR_CONTACT_ELBOW,
            "Arm extension at estimated contact is below the clear reference group.",
        ),
        (
            "POOR_PROXIMAL_DISTAL_TIMING",
            SmashPhase.ACCELERATION,
            (
                float(m.peak_elbow_omega_offset_frames)
                if m.peak_elbow_omega_offset_frames is not None
                else None
            ),
            METRIC_CLEAR_ELBOW_TIMING,
            "Elbow-to-wrist peak timing sits outside the clear reference band.",
        ),
        (
            "LOW_CONTACT_POSTURE",
            SmashPhase.ESTIMATED_CONTACT,
            m.contact_wrist_y_normalized,
            METRIC_CLEAR_CONTACT_WRIST_Y,
            "Contact height appears low relative to the clear reference group.",
        ),
        (
            "RESTRICTED_FOLLOW_THROUGH",
            SmashPhase.FOLLOW_THROUGH,
            m.follow_through_speed_ratio,
            METRIC_CLEAR_FOLLOW_RATIO,
            "Follow-through speed retention is below the clear reference group.",
        ),
        (
            "SLOW_RECOVERY",
            SmashPhase.RECOVERY,
            (
                float(m.recovery_frame_count)
                if m.recovery_frame_count is not None
                else None
            ),
            METRIC_CLEAR_RECOVERY_FRAMES,
            "Recovery duration sits outside the clear reference band.",
        ),
    )
    issues: list[TechniqueIssue] = []
    seen: set[str] = set()
    for code, phase, measured, metric_id, description in checks:
        if measured is None:
            continue
        metric = profile.get_metric(metric_id)
        if metric is None:
            continue
        # Prefer first prep issue only once.
        if code in seen and code == "LIMITED_CLEAR_PREPARATION":
            continue
        issue = _maybe_issue_from_distribution(
            code=code,
            phase=phase,
            measured=float(measured),
            metric=metric,
            profile=profile,
            cfg=cfg,
            decision=decision,
            context=context,
            rule_version=rule_version,
            description=description,
        )
        if issue is not None:
            issues.append(issue)
            seen.add(code)
    return issues


def _clear_provisional_fallback(
    m: ForehandClearMetrics,
    context: _ConfidenceContext,
    calib,
) -> list[TechniqueIssue]:
    """Hard-coded clear fallbacks when no clear profile matches."""
    issues: list[TechniqueIssue] = []
    if m.contact_elbow_angle_deg is None:
        return issues
    if m.contact_elbow_angle_deg >= 145.0:
        return issues
    eval_conf = build_evaluation_confidence(
        measurement_confidence=context.measurement_confidence,
        phase_confidence=context.phase_confidence,
        video_quality_confidence=context.video_quality_confidence,
        reference_confidence=0.2,
        pose_confidence=context.pose_confidence,
        config=calib,
    )
    if eval_conf.combined_confidence < calib.min_combined_confidence:
        status = IssueStatus.INSUFFICIENT_EVIDENCE
        severity = IssueSeverity.LOW
        reason = "Low combined confidence for provisional clear fallback"
    else:
        status = IssueStatus.MODERATE
        severity = IssueSeverity.MEDIUM
        reason = "Provisional clear threshold (no clear reference profile)"
    issues.append(
        TechniqueIssue(
            code="INSUFFICIENT_ARM_EXTENSION",
            phase=SmashPhase.ESTIMATED_CONTACT,
            severity=severity,
            confidence=float(eval_conf.combined_confidence),
            measured_value=float(m.contact_elbow_angle_deg),
            reference_range=ReferenceRange(min=145.0, max=180.0),
            unit="deg",
            description=(
                "Arm extension at estimated contact is below provisional "
                f"clear threshold. [{reason}]"
            ),
            reference_profile_id="provisional_clear_fallback_v1",
            metric_name=METRIC_CLEAR_CONTACT_ELBOW,
            measurement_confidence=eval_conf.measurement_confidence,
            phase_confidence=eval_conf.phase_confidence,
            video_quality_confidence=eval_conf.video_quality_confidence,
            reference_confidence=eval_conf.reference_confidence,
            combined_confidence=eval_conf.combined_confidence,
            rule_version=TECHNIQUE_RULE_VERSION_FALLBACK,
            severity_calibration_version=calib.version,
            decision_mode="provisional_fallback",
            status=status.value,
            uncertain=status == IssueStatus.INSUFFICIENT_EVIDENCE,
            status_reason=reason,
        )
    )
    return issues
