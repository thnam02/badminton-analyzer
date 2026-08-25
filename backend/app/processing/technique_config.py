"""Configurable thresholds for V1 smash technique rules."""

from __future__ import annotations

from dataclasses import dataclass

from app.config import settings


@dataclass(frozen=True, slots=True)
class TechniqueRuleConfig:
    """Rule thresholds; override per call or load from settings."""

    min_contact_elbow_angle_deg: float = 150.0
    min_knee_contribution_deg: float = 12.0
    max_peak_elbow_omega_lead_frames: int = 2
    min_peak_elbow_omega_lead_frames: int = -8
    min_acceleration_phase_fraction: float = 0.12
    max_contact_wrist_y_normalized: float = 0.58
    min_follow_through_speed_ratio: float = 0.30
    min_follow_through_frames: int = 2
    medium_violation_fraction: float = 0.15
    high_violation_fraction: float = 0.35


def technique_rule_config_from_settings() -> TechniqueRuleConfig:
    return TechniqueRuleConfig(
        min_contact_elbow_angle_deg=settings.technique_min_contact_elbow_angle_deg,
        min_knee_contribution_deg=settings.technique_min_knee_contribution_deg,
        max_peak_elbow_omega_lead_frames=settings.technique_max_peak_elbow_omega_lead_frames,
        min_peak_elbow_omega_lead_frames=settings.technique_min_peak_elbow_omega_lead_frames,
        min_acceleration_phase_fraction=settings.technique_min_acceleration_phase_fraction,
        max_contact_wrist_y_normalized=settings.technique_max_contact_wrist_y_normalized,
        min_follow_through_speed_ratio=settings.technique_min_follow_through_speed_ratio,
        min_follow_through_frames=settings.technique_min_follow_through_frames,
    )
