"""Versioned severity / confidence calibration for technique evaluation (C4)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

SEVERITY_CALIBRATION_VERSION = "1.0.0"

ConfidenceCombination = Literal["min", "weighted_mean", "geometric_mean"]


class IssueStatus(str, Enum):
    """Calibrated technique decision status (C4)."""

    NO_ISSUE = "NO_ISSUE"
    MINOR = "MINOR"
    MODERATE = "MODERATE"
    MAJOR = "MAJOR"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


@dataclass(frozen=True, slots=True)
class EvaluationConfidence:
    """Explicit confidence propagation for a technique decision."""

    measurement_confidence: float
    phase_confidence: float
    video_quality_confidence: float
    reference_confidence: float
    pose_confidence: float = 1.0
    combined_confidence: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "measurement_confidence": self.measurement_confidence,
            "phase_confidence": self.phase_confidence,
            "video_quality_confidence": self.video_quality_confidence,
            "reference_confidence": self.reference_confidence,
            "pose_confidence": self.pose_confidence,
            "combined_confidence": self.combined_confidence,
        }


@dataclass(frozen=True, slots=True)
class ConfidenceWeights:
    """Configurable weights for combined confidence (must sum > 0)."""

    pose: float = 0.15
    phase: float = 0.20
    measurement: float = 0.25
    video_quality: float = 0.15
    reference: float = 0.25

    def to_dict(self) -> dict[str, float]:
        return {
            "pose": self.pose,
            "phase": self.phase,
            "measurement": self.measurement,
            "video_quality": self.video_quality,
            "reference": self.reference,
        }


@dataclass(frozen=True, slots=True)
class SeverityCalibrationConfig:
    """Versioned percentile / confidence bands for severity assignment.

    For adverse low-side metrics (higher_is_better), lower reference percentile
    positions map to stronger severity. Central region → NO_ISSUE.
    """

    version: str = SEVERITY_CALIBRATION_VERSION
    # Percentile-position thresholds for adverse (low) side of higher_is_better metrics.
    # percentile_position <= major_max → MAJOR, etc.
    major_percentile_max: float = 1.0
    moderate_percentile_max: float = 5.0
    minor_percentile_max: float = 15.0
    # Central band where we treat the measurement as NO_ISSUE (in-range).
    central_percentile_low: float = 20.0
    central_percentile_high: float = 80.0
    # Robust-z absolute thresholds (optional secondary signal).
    minor_abs_z: float = 1.0
    moderate_abs_z: float = 1.75
    major_abs_z: float = 2.5
    # Confidence gates.
    min_combined_confidence: float = 0.50
    min_measurement_confidence: float = 0.30
    min_reference_confidence: float = 0.30
    min_reference_sample_count: int = 10
    # Combination method for EvaluationConfidence.combined_confidence.
    combination: ConfidenceCombination = "weighted_mean"
    weights: ConfidenceWeights = field(default_factory=ConfidenceWeights)

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity_calibration_version": self.version,
            "major_percentile_max": self.major_percentile_max,
            "moderate_percentile_max": self.moderate_percentile_max,
            "minor_percentile_max": self.minor_percentile_max,
            "central_percentile_low": self.central_percentile_low,
            "central_percentile_high": self.central_percentile_high,
            "minor_abs_z": self.minor_abs_z,
            "moderate_abs_z": self.moderate_abs_z,
            "major_abs_z": self.major_abs_z,
            "min_combined_confidence": self.min_combined_confidence,
            "min_measurement_confidence": self.min_measurement_confidence,
            "min_reference_confidence": self.min_reference_confidence,
            "min_reference_sample_count": self.min_reference_sample_count,
            "combination": self.combination,
            "weights": self.weights.to_dict(),
        }


def default_severity_calibration() -> SeverityCalibrationConfig:
    return SeverityCalibrationConfig()


def combine_confidence(
    *,
    pose: float,
    phase: float,
    measurement: float,
    video_quality: float,
    reference: float,
    config: SeverityCalibrationConfig,
) -> float:
    """Combine component confidences with a configurable, versioned method."""
    parts = {
        "pose": _clamp01(pose),
        "phase": _clamp01(phase),
        "measurement": _clamp01(measurement),
        "video_quality": _clamp01(video_quality),
        "reference": _clamp01(reference),
    }
    method = config.combination
    if method == "min":
        return float(min(parts.values()))
    if method == "geometric_mean":
        product = 1.0
        for value in parts.values():
            product *= max(value, 1e-6)
        return float(product ** (1.0 / len(parts)))
    # weighted_mean (default)
    w = config.weights
    weight_map = {
        "pose": w.pose,
        "phase": w.phase,
        "measurement": w.measurement,
        "video_quality": w.video_quality,
        "reference": w.reference,
    }
    total_w = sum(weight_map.values())
    if total_w <= 0:
        return float(sum(parts.values()) / len(parts))
    return float(
        sum(parts[k] * weight_map[k] for k in parts) / total_w
    )


def build_evaluation_confidence(
    *,
    measurement_confidence: float,
    phase_confidence: float,
    video_quality_confidence: float,
    reference_confidence: float,
    pose_confidence: float = 1.0,
    config: SeverityCalibrationConfig | None = None,
) -> EvaluationConfidence:
    cfg = config or default_severity_calibration()
    combined = combine_confidence(
        pose=pose_confidence,
        phase=phase_confidence,
        measurement=measurement_confidence,
        video_quality=video_quality_confidence,
        reference=reference_confidence,
        config=cfg,
    )
    return EvaluationConfidence(
        measurement_confidence=_clamp01(measurement_confidence),
        phase_confidence=_clamp01(phase_confidence),
        video_quality_confidence=_clamp01(video_quality_confidence),
        reference_confidence=_clamp01(reference_confidence),
        pose_confidence=_clamp01(pose_confidence),
        combined_confidence=_clamp01(combined),
    )


def calibrate_issue_status(
    *,
    direction: str,
    percentile_position: float | None,
    robust_z: float | None,
    eval_confidence: EvaluationConfidence,
    reference_sample_count: int,
    reference_provisional: bool,
    config: SeverityCalibrationConfig | None = None,
) -> tuple[IssueStatus, str]:
    """Map distribution position + confidence → calibrated status + reason.

    In-range measurements always yield ``NO_ISSUE`` (no issue object needed).
    Adverse deviations with weak evidence yield ``INSUFFICIENT_EVIDENCE`` rather
    than a forced coaching judgement.
    """
    cfg = config or default_severity_calibration()

    adverse_score = _adverse_severity_rank(
        direction=direction,
        percentile_position=percentile_position,
        robust_z=robust_z,
        config=cfg,
    )
    if adverse_score == 0:
        return IssueStatus.NO_ISSUE, "Within central reference region"

    if eval_confidence.combined_confidence < cfg.min_combined_confidence:
        return (
            IssueStatus.INSUFFICIENT_EVIDENCE,
            "Combined confidence below calibrated minimum",
        )
    if eval_confidence.measurement_confidence < cfg.min_measurement_confidence:
        return (
            IssueStatus.INSUFFICIENT_EVIDENCE,
            "Measurement confidence too low for a technique judgement",
        )
    if eval_confidence.reference_confidence < cfg.min_reference_confidence:
        return (
            IssueStatus.INSUFFICIENT_EVIDENCE,
            "Reference-profile confidence too low",
        )
    if reference_sample_count < cfg.min_reference_sample_count:
        return (
            IssueStatus.INSUFFICIENT_EVIDENCE,
            f"Reference sample_count {reference_sample_count} "
            f"< minimum {cfg.min_reference_sample_count}",
        )
    if (
        reference_provisional
        and reference_sample_count < cfg.min_reference_sample_count * 2
        and eval_confidence.combined_confidence < cfg.min_combined_confidence + 0.1
    ):
        return (
            IssueStatus.INSUFFICIENT_EVIDENCE,
            "Provisional reference with insufficient combined confidence",
        )

    if adverse_score == 1:
        return IssueStatus.MINOR, "Slightly outside reference region"
    if adverse_score == 2:
        return IssueStatus.MODERATE, "Clear deviation from reference distribution"
    return IssueStatus.MAJOR, "Extreme deviation from reference distribution"


def _adverse_severity_rank(
    *,
    direction: str,
    percentile_position: float | None,
    robust_z: float | None,
    config: SeverityCalibrationConfig,
) -> int:
    """Return 0=none, 1=minor, 2=moderate, 3=major."""
    # Percentile-based primary signal.
    rank_from_p = 0
    if percentile_position is not None:
        p = float(percentile_position)
        if direction == "higher_is_better":
            if config.central_percentile_low <= p <= config.central_percentile_high:
                rank_from_p = 0
            elif p <= config.major_percentile_max:
                rank_from_p = 3
            elif p <= config.moderate_percentile_max:
                rank_from_p = 2
            elif p <= config.minor_percentile_max:
                rank_from_p = 1
            elif p < config.central_percentile_low:
                rank_from_p = 1
            else:
                rank_from_p = 0
        elif direction == "lower_is_better":
            # High percentile is adverse.
            inv = 100.0 - p
            if config.central_percentile_low <= p <= config.central_percentile_high:
                rank_from_p = 0
            elif inv <= config.major_percentile_max or p >= 100.0 - config.major_percentile_max:
                rank_from_p = 3
            elif inv <= config.moderate_percentile_max or p >= 100.0 - config.moderate_percentile_max:
                rank_from_p = 2
            elif inv <= config.minor_percentile_max or p >= 100.0 - config.minor_percentile_max:
                rank_from_p = 1
            else:
                rank_from_p = 0
        else:  # in_range — adverse if outside central band
            if config.central_percentile_low <= p <= config.central_percentile_high:
                rank_from_p = 0
            elif p <= config.major_percentile_max or p >= 100.0 - config.major_percentile_max:
                rank_from_p = 3
            elif p <= config.moderate_percentile_max or p >= 100.0 - config.moderate_percentile_max:
                rank_from_p = 2
            else:
                rank_from_p = 1

    rank_from_z = 0
    if robust_z is not None:
        z = abs(float(robust_z))
        # Sign must be adverse for directed metrics.
        adverse = True
        if direction == "higher_is_better" and robust_z > 0:
            adverse = False
        if direction == "lower_is_better" and robust_z < 0:
            adverse = False
        if adverse:
            if z >= config.major_abs_z:
                rank_from_z = 3
            elif z >= config.moderate_abs_z:
                rank_from_z = 2
            elif z >= config.minor_abs_z:
                rank_from_z = 1

    # Prefer percentile when available; take max with z so extreme z can escalate.
    if percentile_position is None:
        return rank_from_z
    return max(rank_from_p, rank_from_z)


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))
