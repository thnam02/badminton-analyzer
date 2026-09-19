"""Severity tuning for technique rules (metric bands live in ReferenceProfile)."""

from __future__ import annotations

from dataclasses import dataclass

from app.config import settings
from app.processing.reference_profiles import select_reference_profile
from app.schemas.reference import ReferenceProfile


@dataclass(frozen=True, slots=True)
class TechniqueSeverityConfig:
    """How far outside a reference band maps to MEDIUM / HIGH severity."""

    medium_violation_fraction: float = 0.15
    high_violation_fraction: float = 0.35


# Backward-compatible alias used by older imports/tests.
TechniqueRuleConfig = TechniqueSeverityConfig


def technique_severity_config_from_settings() -> TechniqueSeverityConfig:
    return TechniqueSeverityConfig()


def technique_rule_config_from_settings() -> TechniqueSeverityConfig:
    """Legacy name — returns severity config only."""
    return technique_severity_config_from_settings()


def reference_profile_from_settings(
    *,
    stroke_type: str = "SMASH",
    handedness: str | None = None,
    camera_view: str | None = None,
    skill_level: str | None = None,
) -> ReferenceProfile:
    """Resolve the active provisional/default reference profile."""
    return select_reference_profile(
        stroke_type=stroke_type,
        handedness=handedness,
        camera_view=camera_view or settings.technique_default_camera_view or None,
        skill_level=skill_level,
        profile_id=settings.technique_reference_profile_id or None,
    )
