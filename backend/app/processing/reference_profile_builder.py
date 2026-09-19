"""Deterministic reference-profile builder from StrokeSample + analysis metrics.

Offline only — does not modify production technique rules or provisional profiles.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
from app.schemas.built_reference import (
    REFERENCE_PROFILE_BUILD_VERSION,
    BuiltReferenceProfile,
    BuiltReferenceProfileSet,
    MetricDistribution,
)
from app.schemas.reference_dataset import (
    REFERENCE_DATA_PROTOCOL_VERSION,
    REFERENCE_DATASET_SCHEMA_VERSION,
    ReferenceDatasetManifest,
    StrokeSample,
)
from app.schemas.video_quality import VideoQualityReport

# Additional metrics used by the system / requested for distributions.
METRIC_CONTACT_KNEE = "contact_knee_angle_deg"
METRIC_PEAK_WRIST_SPEED = "peak_wrist_speed"
METRIC_PEAK_ELBOW_OMEGA = "peak_elbow_angular_velocity"

# (metric_id, unit, direction)
PROFILE_METRIC_SPECS: tuple[tuple[str, str, str], ...] = (
    (METRIC_CONTACT_ELBOW, "deg", "higher_is_better"),
    (METRIC_PREP_KNEE, "deg", "in_range"),
    (METRIC_CONTACT_KNEE, "deg", "in_range"),
    (METRIC_KNEE_CONTRIBUTION, "deg", "higher_is_better"),
    (METRIC_PEAK_WRIST_SPEED, "norm_units_per_s", "higher_is_better"),
    (METRIC_PEAK_ELBOW_OMEGA, "deg_per_s", "higher_is_better"),
    (METRIC_ELBOW_PEAK_TIMING, "frames", "in_range"),
    (METRIC_CONTACT_WRIST_Y, "normalized_y", "lower_is_better"),
    (METRIC_FOLLOW_THROUGH_RETENTION, "speed_ratio", "higher_is_better"),
    (METRIC_FOLLOW_THROUGH_FRAMES, "frames", "higher_is_better"),
    (METRIC_ACCEL_FRACTION, "ratio", "higher_is_better"),
)

GroupKey = tuple[str, str, str, str]  # stroke_type, handedness, camera_view, skill_level


@dataclass(frozen=True, slots=True)
class _StrokeMetricRow:
    stroke: StrokeSample
    metrics: dict[str, float | None]
    analysis_confidence: float | None


class ReferenceProfileBuilder:
    """Build versioned empirical profiles from a C1 reference-data manifest."""

    def __init__(
        self,
        *,
        exclude_low_quality: bool = True,
        min_analysis_confidence: float = 0.5,
        require_usable_quality: bool = True,
        metric_specs: Sequence[tuple[str, str, str]] = PROFILE_METRIC_SPECS,
        artifact_root: Path | None = None,
    ) -> None:
        self.exclude_low_quality = bool(exclude_low_quality)
        self.min_analysis_confidence = float(min_analysis_confidence)
        self.require_usable_quality = bool(require_usable_quality)
        self.metric_specs = tuple(metric_specs)
        self.artifact_root = Path(artifact_root) if artifact_root is not None else None

    def build(
        self,
        manifest: ReferenceDatasetManifest,
        *,
        metrics_by_stroke_id: Mapping[str, Mapping[str, Any]] | None = None,
        quality_by_stroke_id: Mapping[str, VideoQualityReport | Mapping[str, Any]]
        | None = None,
    ) -> BuiltReferenceProfileSet:
        """Deterministically build profiles; same inputs → same outputs."""
        exclusion_reasons: dict[str, int] = defaultdict(int)
        rows: list[_StrokeMetricRow] = []

        # Stable stroke order for reproducibility.
        strokes = sorted(manifest.strokes, key=lambda s: s.stroke_id)
        for stroke in strokes:
            reason = self._exclusion_reason(
                stroke,
                quality_by_stroke_id=quality_by_stroke_id,
            )
            if reason is not None:
                exclusion_reasons[reason] += 1
                continue

            metrics = self._resolve_metrics(stroke, metrics_by_stroke_id)
            if metrics is None:
                exclusion_reasons["metrics_unavailable"] += 1
                continue

            conf = self._resolve_confidence(stroke, quality_by_stroke_id)
            rows.append(
                _StrokeMetricRow(
                    stroke=stroke,
                    metrics=metrics,
                    analysis_confidence=conf,
                )
            )

        grouped: dict[GroupKey, list[_StrokeMetricRow]] = defaultdict(list)
        for row in rows:
            grouped[_group_key(row.stroke)].append(row)

        profiles: list[BuiltReferenceProfile] = []
        for key in sorted(grouped.keys()):
            group_rows = sorted(
                grouped[key], key=lambda r: r.stroke.stroke_id
            )
            profiles.append(self._build_group_profile(key, group_rows))

        return BuiltReferenceProfileSet(
            build_version=REFERENCE_PROFILE_BUILD_VERSION,
            protocol_version=manifest.protocol_version
            or REFERENCE_DATA_PROTOCOL_VERSION,
            dataset_schema_version=manifest.schema_version
            or REFERENCE_DATASET_SCHEMA_VERSION,
            profiles=profiles,
            total_input_strokes=len(manifest.strokes),
            total_included_strokes=len(rows),
            total_excluded_strokes=len(manifest.strokes) - len(rows),
            exclusion_reasons=dict(sorted(exclusion_reasons.items())),
        )

    def _exclusion_reason(
        self,
        stroke: StrokeSample,
        *,
        quality_by_stroke_id: Mapping[str, VideoQualityReport | Mapping[str, Any]]
        | None,
    ) -> str | None:
        if stroke.acceptance_status != "accepted":
            return f"acceptance_{stroke.acceptance_status}"

        if not self.exclude_low_quality:
            return None

        quality = self._load_quality(stroke, quality_by_stroke_id)
        if quality is None:
            return None

        usable, confidence = _quality_fields(quality)
        if self.require_usable_quality and usable is False:
            return "quality_unusable"
        if (
            confidence is not None
            and confidence < self.min_analysis_confidence
        ):
            return "quality_low_confidence"
        return None

    def _resolve_metrics(
        self,
        stroke: StrokeSample,
        metrics_by_stroke_id: Mapping[str, Mapping[str, Any]] | None,
    ) -> dict[str, float | None] | None:
        if metrics_by_stroke_id and stroke.stroke_id in metrics_by_stroke_id:
            return _extract_metric_values(
                metrics_by_stroke_id[stroke.stroke_id],
                metric_ids=[m[0] for m in self.metric_specs],
            )

        payload = self._load_metrics_payload(stroke)
        if payload is None:
            return None
        return _extract_metric_values(
            payload,
            metric_ids=[m[0] for m in self.metric_specs],
        )

    def _load_metrics_payload(self, stroke: StrokeSample) -> dict[str, Any] | None:
        refs = stroke.artifact_refs
        candidates: list[Path] = []

        if refs.dataset_export_path:
            candidates.append(self._resolve_path(refs.dataset_export_path))

        analysis_id = refs.analysis_id.strip()
        if analysis_id:
            # Conventional finalized artifact names next to outputs/.
            for suffix in (
                "_stroke_metrics.json",
                "_dataset.json",
            ):
                candidates.append(self._resolve_path(f"{analysis_id}{suffix}"))
                if self.artifact_root is not None:
                    candidates.append(self.artifact_root / f"{analysis_id}{suffix}")

        seen: set[str] = set()
        for path in candidates:
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            if not path.is_file():
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            if "pose_metrics" in data and isinstance(data["pose_metrics"], dict):
                metrics = dict(data["pose_metrics"])
            else:
                metrics = dict(data)
            # Optionally enrich peak elbow ω from motion peaks.
            motion_path = self._sibling_motion_path(path, analysis_id)
            if motion_path is not None and motion_path.is_file():
                motion = json.loads(motion_path.read_text(encoding="utf-8"))
                peak = (motion.get("peaks") or {}).get("right_elbow_angular_velocity")
                if isinstance(peak, dict) and peak.get("value") is not None:
                    metrics.setdefault(
                        METRIC_PEAK_ELBOW_OMEGA, peak.get("value")
                    )
            return metrics
        return None

    def _sibling_motion_path(
        self, metrics_path: Path, analysis_id: str
    ) -> Path | None:
        stem = analysis_id or metrics_path.name
        for token in (
            "_stroke_metrics.json",
            "_dataset.json",
            "_pose.json",
        ):
            if stem.endswith(token):
                stem = stem[: -len(token)]
                break
        if analysis_id:
            stem = analysis_id
        sibling = metrics_path.with_name(f"{stem}_motion.json")
        if sibling.is_file():
            return sibling
        if self.artifact_root is not None:
            alt = self.artifact_root / f"{stem}_motion.json"
            if alt.is_file():
                return alt
        return None

    def _resolve_path(self, raw: str) -> Path:
        path = Path(raw)
        if path.is_absolute():
            return path
        if self.artifact_root is not None:
            return self.artifact_root / path
        return path

    def _resolve_confidence(
        self,
        stroke: StrokeSample,
        quality_by_stroke_id: Mapping[str, VideoQualityReport | Mapping[str, Any]]
        | None,
    ) -> float | None:
        quality = self._load_quality(stroke, quality_by_stroke_id)
        if quality is None:
            return None
        _, confidence = _quality_fields(quality)
        return confidence

    def _load_quality(
        self,
        stroke: StrokeSample,
        quality_by_stroke_id: Mapping[str, VideoQualityReport | Mapping[str, Any]]
        | None,
    ) -> VideoQualityReport | Mapping[str, Any] | None:
        if quality_by_stroke_id and stroke.stroke_id in quality_by_stroke_id:
            return quality_by_stroke_id[stroke.stroke_id]
        path_raw = stroke.artifact_refs.video_quality_json_path
        if not path_raw:
            analysis_id = stroke.artifact_refs.analysis_id.strip()
            if analysis_id:
                path_raw = f"{analysis_id}_video_quality.json"
        if not path_raw:
            return None
        path = self._resolve_path(path_raw)
        if self.artifact_root is not None and not path.is_file():
            path = self.artifact_root / Path(path_raw).name
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _build_group_profile(
        self,
        key: GroupKey,
        rows: Sequence[_StrokeMetricRow],
    ) -> BuiltReferenceProfile:
        stroke_type, handedness, camera_view, skill_level = key
        profile_id = _profile_id(key)
        confidences = [
            float(r.analysis_confidence)
            for r in rows
            if r.analysis_confidence is not None
        ]
        metrics: dict[str, MetricDistribution] = {}
        for metric_id, unit, direction in self.metric_specs:
            values: list[float] = []
            value_confidences: list[float] = []
            for row in rows:
                raw = row.metrics.get(metric_id)
                if raw is None:
                    continue
                values.append(float(raw))
                if row.analysis_confidence is not None:
                    value_confidences.append(float(row.analysis_confidence))
            metrics[metric_id] = summarize_metric_values(
                metric_id=metric_id,
                unit=unit,
                direction=direction,
                values=values,
                sample_count=len(rows),
                quality_summary={
                    "mean_analysis_confidence": _mean(value_confidences),
                    "observed_with_confidence_count": len(value_confidences),
                },
            )

        return BuiltReferenceProfile(
            profile_id=profile_id,
            stroke_type=stroke_type,
            handedness=handedness,
            camera_view=camera_view,
            skill_level=skill_level,
            build_version=REFERENCE_PROFILE_BUILD_VERSION,
            stroke_count=len(rows),
            included_stroke_ids=[r.stroke.stroke_id for r in rows],
            excluded_stroke_ids=[],
            metrics=metrics,
            quality_summary={
                "mean_analysis_confidence": _mean(confidences),
                "median_analysis_confidence": _percentile(sorted(confidences), 50.0)
                if confidences
                else None,
                "strokes_with_confidence": len(confidences),
            },
        )


def build_reference_profiles(
    manifest: ReferenceDatasetManifest,
    *,
    metrics_by_stroke_id: Mapping[str, Mapping[str, Any]] | None = None,
    quality_by_stroke_id: Mapping[str, VideoQualityReport | Mapping[str, Any]]
    | None = None,
    exclude_low_quality: bool = True,
    min_analysis_confidence: float = 0.5,
    artifact_root: Path | None = None,
) -> BuiltReferenceProfileSet:
    """Module-level convenience wrapper around ``ReferenceProfileBuilder``."""
    return ReferenceProfileBuilder(
        exclude_low_quality=exclude_low_quality,
        min_analysis_confidence=min_analysis_confidence,
        artifact_root=artifact_root,
    ).build(
        manifest,
        metrics_by_stroke_id=metrics_by_stroke_id,
        quality_by_stroke_id=quality_by_stroke_id,
    )


def export_built_reference_profiles(
    profile_set: BuiltReferenceProfileSet,
    output_path: Path,
) -> Path:
    """Write versioned ``reference_profiles_built.json`` (or caller-chosen name)."""
    return profile_set.save_json(Path(output_path))


def summarize_metric_values(
    *,
    metric_id: str,
    unit: str,
    direction: str,
    values: Sequence[float],
    sample_count: int,
    quality_summary: Mapping[str, Any] | None = None,
) -> MetricDistribution:
    """Compute deterministic distribution stats for a known value list."""
    observed = [float(v) for v in values]
    observed_count = len(observed)
    missing_rate = (
        float(sample_count - observed_count) / float(sample_count)
        if sample_count > 0
        else 0.0
    )
    if not observed:
        return MetricDistribution(
            metric_id=metric_id,
            unit=unit,
            direction=direction,
            sample_count=sample_count,
            observed_count=0,
            missing_rate=missing_rate,
            quality_summary=dict(quality_summary or {}),
        )

    ordered = sorted(observed)
    p10 = _percentile(ordered, 10.0)
    p25 = _percentile(ordered, 25.0)
    p75 = _percentile(ordered, 75.0)
    p90 = _percentile(ordered, 90.0)
    median = _percentile(ordered, 50.0)
    mean = _mean(ordered)
    std = _sample_std(ordered)
    iqr = (
        (p75 - p25)
        if p75 is not None and p25 is not None
        else None
    )
    return MetricDistribution(
        metric_id=metric_id,
        unit=unit,
        direction=direction,
        sample_count=sample_count,
        observed_count=observed_count,
        missing_rate=missing_rate,
        median=median,
        mean=mean,
        std=std,
        p10=p10,
        p25=p25,
        p75=p75,
        p90=p90,
        iqr=iqr,
        quality_summary=dict(quality_summary or {}),
    )


def _extract_metric_values(
    payload: Mapping[str, Any],
    *,
    metric_ids: Sequence[str],
) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for metric_id in metric_ids:
        raw = payload.get(metric_id)
        if raw is None:
            out[metric_id] = None
            continue
        try:
            out[metric_id] = float(raw)
        except (TypeError, ValueError):
            out[metric_id] = None
    return out


def _group_key(stroke: StrokeSample) -> GroupKey:
    return (
        str(stroke.stroke_type).upper(),
        str(stroke.handedness).upper(),
        str(stroke.camera_view).lower(),
        str(stroke.skill_group).lower(),
    )


def _profile_id(key: GroupKey) -> str:
    stroke_type, handedness, camera_view, skill_level = key
    parts = [
        stroke_type.lower(),
        handedness.lower(),
        camera_view.lower().replace("-", "_"),
        skill_level.lower().replace("-", "_"),
        f"built_v{REFERENCE_PROFILE_BUILD_VERSION.replace('.', '_')}",
    ]
    return "_".join(parts)


def _quality_fields(
    quality: VideoQualityReport | Mapping[str, Any],
) -> tuple[bool | None, float | None]:
    if isinstance(quality, VideoQualityReport):
        return bool(quality.usable), float(quality.analysis_confidence)
    usable = quality.get("usable")
    conf = quality.get("analysis_confidence")
    return (
        bool(usable) if usable is not None else None,
        float(conf) if conf is not None else None,
    )


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _sample_std(values: Sequence[float]) -> float | None:
    n = len(values)
    if n == 0:
        return None
    if n == 1:
        return 0.0
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return float(math.sqrt(var))


def _percentile(sorted_values: Sequence[float], percentile: float) -> float | None:
    """Linear-interpolation percentile on a pre-sorted ascending sequence."""
    if not sorted_values:
        return None
    if not 0.0 <= percentile <= 100.0:
        raise ValueError(f"percentile must be in [0, 100], got {percentile}")
    vals = list(sorted_values)
    if len(vals) == 1:
        return float(vals[0])
    rank = (percentile / 100.0) * (len(vals) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return float(vals[lo])
    weight = rank - lo
    return float(vals[lo] * (1.0 - weight) + vals[hi] * weight)
