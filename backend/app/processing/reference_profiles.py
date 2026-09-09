"""Provisional reference profiles and selection helpers.

Values are sample/configuration placeholders clearly marked provisional —
not scientifically validated population norms.
"""

from __future__ import annotations

from app.config import settings
from app.schemas.reference import MetricReference, ReferenceProfile

# Metric IDs used by the technique evaluator.
METRIC_CONTACT_ELBOW = "contact_elbow_angle_deg"
METRIC_PREP_KNEE = "preparation_knee_angle_deg"
METRIC_KNEE_CONTRIBUTION = "knee_contribution_deg"
METRIC_ELBOW_PEAK_TIMING = "peak_elbow_omega_offset_frames"
METRIC_FOLLOW_THROUGH_RETENTION = "follow_through_speed_ratio"
METRIC_ACCEL_FRACTION = "acceleration_phase_fraction"
METRIC_CONTACT_WRIST_Y = "contact_wrist_y_normalized"
METRIC_FOLLOW_THROUGH_FRAMES = "follow_through_frame_count"


def _metric(
    metric_id: str,
    *,
    unit: str,
    median: float,
    lower: float,
    upper: float,
    direction: str,
    sample_count: int = 24,
    confidence: float = 0.35,
) -> MetricReference:
    return MetricReference(
        metric_id=metric_id,
        unit=unit,
        median=median,
        lower_percentile=lower,
        upper_percentile=upper,
        sample_count=sample_count,
        provenance="provisional_config_sample_v1",
        confidence=confidence,
        provisional=True,
        direction=direction,
    )


def build_provisional_smash_right_side() -> ReferenceProfile:
    """Default smash / right / side-view provisional profile.

    Percentile bounds are seeded from prior TechniqueRuleConfig defaults so
    existing issue behaviour stays comparable while moving thresholds out of
    rule code.
    """
    metrics = {
        METRIC_CONTACT_ELBOW: _metric(
            METRIC_CONTACT_ELBOW,
            unit="deg",
            median=162.0,
            lower=float(settings.technique_min_contact_elbow_angle_deg),
            upper=180.0,
            direction="higher_is_better",
        ),
        METRIC_PREP_KNEE: _metric(
            METRIC_PREP_KNEE,
            unit="deg",
            median=138.0,
            lower=120.0,
            upper=155.0,
            direction="in_range",
        ),
        METRIC_KNEE_CONTRIBUTION: _metric(
            METRIC_KNEE_CONTRIBUTION,
            unit="deg",
            median=18.0,
            lower=float(settings.technique_min_knee_contribution_deg),
            upper=40.0,
            direction="higher_is_better",
        ),
        METRIC_ELBOW_PEAK_TIMING: _metric(
            METRIC_ELBOW_PEAK_TIMING,
            unit="frames",
            median=-2.0,
            lower=float(settings.technique_min_peak_elbow_omega_lead_frames),
            upper=float(settings.technique_max_peak_elbow_omega_lead_frames),
            direction="in_range",
        ),
        METRIC_FOLLOW_THROUGH_RETENTION: _metric(
            METRIC_FOLLOW_THROUGH_RETENTION,
            unit="speed_ratio",
            median=0.45,
            lower=float(settings.technique_min_follow_through_speed_ratio),
            upper=1.0,
            direction="higher_is_better",
        ),
        METRIC_ACCEL_FRACTION: _metric(
            METRIC_ACCEL_FRACTION,
            unit="ratio",
            median=0.25,
            lower=float(settings.technique_min_acceleration_phase_fraction),
            upper=0.55,
            direction="higher_is_better",
        ),
        METRIC_CONTACT_WRIST_Y: _metric(
            METRIC_CONTACT_WRIST_Y,
            unit="normalized_y",
            median=0.42,
            lower=0.15,
            upper=float(settings.technique_max_contact_wrist_y_normalized),
            direction="lower_is_better",
        ),
        METRIC_FOLLOW_THROUGH_FRAMES: _metric(
            METRIC_FOLLOW_THROUGH_FRAMES,
            unit="frames",
            median=6.0,
            lower=float(settings.technique_min_follow_through_frames),
            upper=20.0,
            direction="higher_is_better",
        ),
    }
    return ReferenceProfile(
        profile_id="smash_right_side_provisional_v1",
        stroke_type="SMASH",
        handedness="RIGHT",
        camera_view="SIDE",
        metrics=metrics,
        provisional=True,
        notes=(
            "Provisional smash reference (right-handed, side view) seeded from "
            "configuration/sample defaults — not scientifically validated."
        ),
    )


def build_provisional_smash_any() -> ReferenceProfile:
    """Fallback smash profile when handedness/camera are unknown."""
    base = build_provisional_smash_right_side()
    return ReferenceProfile(
        profile_id="smash_any_provisional_v1",
        stroke_type="SMASH",
        handedness=None,
        camera_view=None,
        metrics=dict(base.metrics),
        provisional=True,
        notes=(
            "Provisional smash fallback profile (any handedness/view) — "
            "not scientifically validated."
        ),
    )


def build_provisional_smash_left_side() -> ReferenceProfile:
    """Mirror of the right-side smash profile for left-handed tagging."""
    base = build_provisional_smash_right_side()
    return ReferenceProfile(
        profile_id="smash_left_side_provisional_v1",
        stroke_type="SMASH",
        handedness="LEFT",
        camera_view="SIDE",
        metrics=dict(base.metrics),
        provisional=True,
        notes=(
            "Provisional smash reference (left-handed, side view) — "
            "not scientifically validated; metrics mirrored from right-side sample."
        ),
    )


def default_reference_profiles() -> list[ReferenceProfile]:
    return [
        build_provisional_smash_right_side(),
        build_provisional_smash_left_side(),
        build_provisional_smash_any(),
    ]


def select_reference_profile(
    *,
    stroke_type: str = "SMASH",
    handedness: str | None = None,
    camera_view: str | None = None,
    profile_id: str | None = None,
    profiles: list[ReferenceProfile] | None = None,
) -> ReferenceProfile:
    """Select the best matching reference profile.

    Preference order:
    1. Explicit ``profile_id``
    2. Exact stroke + handedness + camera_view
    3. stroke + handedness + any camera
    4. stroke + any handedness + camera_view
    5. stroke + any + any
    6. First profile with matching stroke_type
    """
    catalog = list(profiles) if profiles is not None else default_reference_profiles()
    if not catalog:
        raise ValueError("No reference profiles available")

    if profile_id:
        for profile in catalog:
            if profile.profile_id == profile_id:
                return profile
        raise KeyError(f"Unknown reference profile_id '{profile_id}'")

    stroke = stroke_type.upper()
    hand = handedness.upper() if handedness else None
    view = camera_view.upper() if camera_view else None

    def _score(profile: ReferenceProfile) -> tuple[int, int, int]:
        if profile.stroke_type.upper() != stroke:
            return (-1, -1, -1)
        hand_score = (
            2
            if hand is not None and profile.handedness == hand
            else 1
            if profile.handedness is None
            else 0
            if hand is None
            else -1
        )
        view_score = (
            2
            if view is not None and profile.camera_view == view
            else 1
            if profile.camera_view is None
            else 0
            if view is None
            else -1
        )
        if hand_score < 0 or view_score < 0:
            return (-1, -1, -1)
        return (1, hand_score, view_score)

    ranked = sorted(
        (( _score(p), p) for p in catalog),
        key=lambda item: item[0],
        reverse=True,
    )
    best_score, best = ranked[0]
    if best_score[0] < 0:
        # No stroke match — return first catalog entry as last resort.
        return catalog[0]
    return best
