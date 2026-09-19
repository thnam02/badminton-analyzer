"""Provisional forehand-clear reference profiles (not smash distributions)."""

from __future__ import annotations

from app.schemas.reference import MetricReference, ReferenceProfile

# Clear metric IDs (must match ForehandClearMetrics fields / evaluator).
METRIC_CLEAR_CONTACT_ELBOW = "contact_elbow_angle_deg"
METRIC_CLEAR_PREP_ELBOW = "preparation_elbow_angle_deg"
METRIC_CLEAR_PREP_KNEE = "preparation_knee_angle_deg"
METRIC_CLEAR_BACKSWING_ELBOW = "backswing_min_elbow_angle_deg"
METRIC_CLEAR_ELBOW_TIMING = "peak_elbow_omega_offset_frames"
METRIC_CLEAR_CONTACT_WRIST_Y = "contact_wrist_y_normalized"
METRIC_CLEAR_FOLLOW_RATIO = "follow_through_speed_ratio"
METRIC_CLEAR_RECOVERY_FRAMES = "recovery_frame_count"


def _m(
    metric_id: str,
    *,
    unit: str,
    median: float,
    lower: float,
    upper: float,
    direction: str,
    std: float | None = None,
    sample_count: int = 20,
    confidence: float = 0.32,
) -> MetricReference:
    return MetricReference(
        metric_id=metric_id,
        unit=unit,
        median=median,
        mean=median,
        std=std,
        lower_percentile=lower,
        upper_percentile=upper,
        sample_count=sample_count,
        provenance="provisional_clear_config_sample_v1",
        confidence=confidence,
        provisional=True,
        direction=direction,
    )


def build_provisional_clear_right_side() -> ReferenceProfile:
    """Provisional clear / right / side-view profile — not coach-validated."""
    return ReferenceProfile(
        profile_id="clear_right_side_provisional_v1",
        stroke_type="FOREHAND_CLEAR",
        handedness="RIGHT",
        camera_view="SIDE",
        skill_level=None,
        provisional=True,
        profile_version="provisional_v1",
        source="provisional_clear_catalog",
        status="DRAFT",
        sample_count=20,
        notes=(
            "Provisional forehand-clear reference values from configuration "
            "placeholders — not scientifically validated. Do not reuse smash "
            "distributions."
        ),
        metrics={
            METRIC_CLEAR_CONTACT_ELBOW: _m(
                METRIC_CLEAR_CONTACT_ELBOW,
                unit="deg",
                median=158.0,
                lower=145.0,
                upper=178.0,
                direction="higher_is_better",
                std=8.0,
            ),
            METRIC_CLEAR_PREP_ELBOW: _m(
                METRIC_CLEAR_PREP_ELBOW,
                unit="deg",
                median=95.0,
                lower=70.0,
                upper=120.0,
                direction="in_range",
                std=12.0,
            ),
            METRIC_CLEAR_PREP_KNEE: _m(
                METRIC_CLEAR_PREP_KNEE,
                unit="deg",
                median=135.0,
                lower=115.0,
                upper=155.0,
                direction="in_range",
                std=10.0,
            ),
            METRIC_CLEAR_BACKSWING_ELBOW: _m(
                METRIC_CLEAR_BACKSWING_ELBOW,
                unit="deg",
                median=75.0,
                lower=50.0,
                upper=100.0,
                direction="in_range",
                std=12.0,
            ),
            METRIC_CLEAR_ELBOW_TIMING: _m(
                METRIC_CLEAR_ELBOW_TIMING,
                unit="frames",
                median=-2.0,
                lower=-10.0,
                upper=3.0,
                direction="in_range",
                std=2.0,
            ),
            METRIC_CLEAR_CONTACT_WRIST_Y: _m(
                METRIC_CLEAR_CONTACT_WRIST_Y,
                unit="normalized_y",
                median=0.28,
                lower=0.12,
                upper=0.42,
                direction="lower_is_better",
                std=0.06,
            ),
            METRIC_CLEAR_FOLLOW_RATIO: _m(
                METRIC_CLEAR_FOLLOW_RATIO,
                unit="speed_ratio",
                median=0.45,
                lower=0.25,
                upper=0.75,
                direction="higher_is_better",
                std=0.1,
            ),
            METRIC_CLEAR_RECOVERY_FRAMES: _m(
                METRIC_CLEAR_RECOVERY_FRAMES,
                unit="frames",
                median=12.0,
                lower=4.0,
                upper=30.0,
                direction="in_range",
                std=5.0,
            ),
        },
    )


def build_provisional_clear_any() -> ReferenceProfile:
    base = build_provisional_clear_right_side()
    return ReferenceProfile(
        profile_id="clear_any_provisional_v1",
        stroke_type="FOREHAND_CLEAR",
        handedness=None,
        camera_view=None,
        skill_level=None,
        metrics=dict(base.metrics),
        provisional=True,
        profile_version="provisional_v1",
        source="provisional_clear_catalog",
        status="DRAFT",
        sample_count=20,
        notes=(
            "Provisional forehand-clear fallback profile (any handedness/view) — "
            "not scientifically validated."
        ),
    )


def default_clear_reference_profiles() -> list[ReferenceProfile]:
    return [
        build_provisional_clear_right_side(),
        build_provisional_clear_any(),
    ]
