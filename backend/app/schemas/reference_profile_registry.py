"""Reference-profile provenance, status, freezing, and fingerprints (C6)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.schemas.reference import ReferenceProfile

REFERENCE_PROFILE_SCHEMA_VERSION = "1.0.0"
METRIC_DEFINITION_VERSION = "stroke_metrics_v1"


class ProfileStatus(str, Enum):
    DRAFT = "DRAFT"
    VALIDATING = "VALIDATING"
    VALIDATED = "VALIDATED"
    DEPRECATED = "DEPRECATED"


class ImmutableProfileError(ValueError):
    """Raised when mutating a VALIDATED / frozen profile."""


@dataclass(slots=True)
class ProfileProvenance:
    """Dataset / build provenance attached to a released profile."""

    dataset_id: str = ""
    dataset_version: str = ""
    dataset_fingerprint: str = ""
    sample_count: int = 0
    valid_sample_count: int = 0
    included_sample_ids: list[str] = field(default_factory=list)
    excluded_sample_ids: list[str] = field(default_factory=list)
    exclusion_reasons: dict[str, int] = field(default_factory=dict)
    quality_thresholds: dict[str, Any] = field(default_factory=dict)
    builder_version: str = ""
    metric_definition_version: str = METRIC_DEFINITION_VERSION
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "dataset_fingerprint": self.dataset_fingerprint,
            "sample_count": self.sample_count,
            "valid_sample_count": self.valid_sample_count,
            "included_samples": self.valid_sample_count,
            "excluded_samples": len(self.excluded_sample_ids),
            "included_sample_ids": list(self.included_sample_ids),
            "excluded_sample_ids": list(self.excluded_sample_ids),
            "exclusion_reasons": dict(self.exclusion_reasons),
            "quality_thresholds": dict(self.quality_thresholds),
            "builder_version": self.builder_version,
            "metric_definition_version": self.metric_definition_version,
            "created_at": self.created_at,
        }


@dataclass(slots=True)
class VersionedReferenceProfile:
    """Immutable-capable wrapper around ReferenceProfile + provenance (C6)."""

    profile: ReferenceProfile
    schema_version: str = REFERENCE_PROFILE_SCHEMA_VERSION
    status: ProfileStatus = ProfileStatus.DRAFT
    provenance: ProfileProvenance = field(default_factory=ProfileProvenance)
    frozen: bool = False

    @property
    def profile_id(self) -> str:
        return self.profile.profile_id

    @property
    def profile_version(self) -> str:
        return self.profile.profile_version

    def to_dict(self) -> dict[str, Any]:
        base = self.profile.to_dict()
        base.update(
            {
                "schema_version": self.schema_version,
                "status": self.status.value,
                "frozen": self.frozen,
                "provenance": self.provenance.to_dict(),
            }
        )
        return base

    def save_json(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    def freeze_as_validated(self) -> VersionedReferenceProfile:
        """Mark VALIDATED and immutable. Idempotent if already validated/frozen."""
        if self.status == ProfileStatus.VALIDATED and self.frozen:
            return self
        if self.frozen and self.status != ProfileStatus.VALIDATED:
            raise ImmutableProfileError(
                f"Profile {self.profile_id} is frozen with status {self.status}"
            )
        self.status = ProfileStatus.VALIDATED
        self.frozen = True
        self.profile = _with_status_fields(self.profile, status=ProfileStatus.VALIDATED)
        return self

    def mark_deprecated(self) -> VersionedReferenceProfile:
        if self.status == ProfileStatus.VALIDATED and self.frozen:
            # Allowed transition: VALIDATED → DEPRECATED (still immutable content).
            self.status = ProfileStatus.DEPRECATED
            self.profile = _with_status_fields(
                self.profile, status=ProfileStatus.DEPRECATED
            )
            return self
        if self.frozen:
            raise ImmutableProfileError(
                f"Cannot deprecate frozen profile {self.profile_id} "
                f"from status {self.status}"
            )
        self.status = ProfileStatus.DEPRECATED
        self.profile = _with_status_fields(self.profile, status=ProfileStatus.DEPRECATED)
        return self

    def replace_metrics(self, metrics: Mapping[str, Any]) -> None:
        if self.frozen or self.status == ProfileStatus.VALIDATED:
            raise ImmutableProfileError(
                f"Validated profile {self.profile_id} is immutable; create a new version"
            )
        raise NotImplementedError(
            "Use bump_profile_version() to create vN+1 instead of mutating metrics"
        )


def bump_profile_version(
    current: VersionedReferenceProfile,
    *,
    new_profile: ReferenceProfile,
    provenance: ProfileProvenance | None = None,
) -> VersionedReferenceProfile:
    """Create the next draft version rather than rewriting a validated profile."""
    next_version = _next_version_label(current.profile_version or "v1")
    base_id = _strip_version_suffix(current.profile_id)
    bumped = ReferenceProfile(
        profile_id=f"{base_id}_{next_version}",
        stroke_type=new_profile.stroke_type,
        handedness=new_profile.handedness,
        camera_view=new_profile.camera_view,
        skill_level=new_profile.skill_level,
        metrics=dict(new_profile.metrics),
        provisional=True,
        profile_version=next_version,
        source=new_profile.source,
        notes=new_profile.notes,
        status=ProfileStatus.DRAFT.value,
        dataset_id=(provenance.dataset_id if provenance else ""),
        dataset_version=(provenance.dataset_version if provenance else ""),
        dataset_fingerprint=(provenance.dataset_fingerprint if provenance else ""),
        sample_count=(provenance.valid_sample_count if provenance else 0),
    )
    return VersionedReferenceProfile(
        profile=bumped,
        status=ProfileStatus.DRAFT,
        provenance=provenance or ProfileProvenance(created_at=_utc_now()),
        frozen=False,
    )


def deterministic_dataset_fingerprint(
    *,
    sample_ids: Sequence[str],
    metric_payloads: Mapping[str, Mapping[str, Any]],
    metric_keys: Sequence[str] | None = None,
) -> str:
    """Fingerprint from canonical sample IDs + metric values (no timestamps/paths)."""
    keys = list(metric_keys) if metric_keys is not None else None
    rows: list[dict[str, Any]] = []
    for sample_id in sorted(str(s) for s in sample_ids):
        payload = metric_payloads.get(sample_id) or {}
        if keys is None:
            metric_items = {
                str(k): _canonical_number(payload.get(k))
                for k in sorted(payload.keys())
            }
        else:
            metric_items = {
                str(k): _canonical_number(payload.get(k)) for k in keys
            }
        rows.append({"sample_id": sample_id, "metrics": metric_items})
    blob = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def build_profile_manifest(versioned: VersionedReferenceProfile) -> dict[str, Any]:
    prov = versioned.provenance
    return {
        "profile_id": versioned.profile_id,
        "profile_version": versioned.profile_version,
        "schema_version": versioned.schema_version,
        "status": versioned.status.value,
        "stroke_type": versioned.profile.stroke_type,
        "handedness": versioned.profile.handedness,
        "camera_view": versioned.profile.camera_view,
        "skill_level": versioned.profile.skill_level,
        "dataset_id": prov.dataset_id,
        "dataset_version": prov.dataset_version,
        "dataset_fingerprint": prov.dataset_fingerprint,
        "included_samples": prov.valid_sample_count or len(prov.included_sample_ids),
        "excluded_samples": len(prov.excluded_sample_ids),
        "exclusion_reasons": dict(prov.exclusion_reasons),
        "builder_version": prov.builder_version,
        "metric_definition_version": prov.metric_definition_version,
        "created_at": prov.created_at or _utc_now(),
        "frozen": versioned.frozen,
    }


class ReferenceProfileRegistry:
    """In-memory registry of versioned profiles for selection / reproduction."""

    def __init__(self, profiles: Sequence[VersionedReferenceProfile] | None = None) -> None:
        self._by_id: dict[str, VersionedReferenceProfile] = {}
        for profile in profiles or []:
            self.register(profile)

    def register(self, profile: VersionedReferenceProfile) -> None:
        self._by_id[profile.profile_id] = profile

    def get(self, profile_id: str) -> VersionedReferenceProfile | None:
        return self._by_id.get(profile_id)

    def require(self, profile_id: str) -> VersionedReferenceProfile:
        found = self.get(profile_id)
        if found is None:
            raise KeyError(f"Unknown reference profile_id '{profile_id}'")
        return found

    def list_profiles(
        self, *, status: ProfileStatus | None = None
    ) -> list[VersionedReferenceProfile]:
        values = list(self._by_id.values())
        if status is not None:
            values = [p for p in values if p.status == status]
        return sorted(values, key=lambda p: p.profile_id)

    def latest_validated_compatible(
        self,
        *,
        stroke_type: str,
        handedness: str | None = None,
        camera_view: str | None = None,
        skill_level: str | None = None,
    ) -> VersionedReferenceProfile | None:
        """Pick newest VALIDATED profile matching dimensions (never returns 'latest')."""
        from app.processing.reference_profile_selector import (
            _normalize_hand,
            _normalize_skill,
            _normalize_stroke,
            _normalize_view,
        )

        stroke = _normalize_stroke(stroke_type)
        hand = _normalize_hand(handedness)
        view = _normalize_view(camera_view)
        skill = _normalize_skill(skill_level)
        candidates: list[VersionedReferenceProfile] = []
        for item in self.list_profiles(status=ProfileStatus.VALIDATED):
            p = item.profile
            if _normalize_stroke(p.stroke_type) != stroke:
                continue
            if hand is not None and _normalize_hand(p.handedness) not in (hand, None):
                if _normalize_hand(p.handedness) != hand:
                    continue
            if view is not None and _normalize_view(p.camera_view) not in (view, None):
                if _normalize_view(p.camera_view) != view:
                    continue
            if skill is not None and _normalize_skill(p.skill_level) not in (skill, None):
                if _normalize_skill(p.skill_level) != skill:
                    continue
            candidates.append(item)
        if not candidates:
            return None
        # Prefer exact dimension matches, then highest version label.
        def _score(item: VersionedReferenceProfile) -> tuple[int, int, int, str]:
            p = item.profile
            hand_s = 2 if hand and _normalize_hand(p.handedness) == hand else 1
            view_s = 2 if view and _normalize_view(p.camera_view) == view else 1
            skill_s = 2 if skill and _normalize_skill(p.skill_level) == skill else 1
            return (hand_s, view_s, skill_s, item.profile_version)

        candidates.sort(key=_score, reverse=True)
        return candidates[0]

    def as_reference_profiles(
        self, *, include_deprecated: bool = False
    ) -> list[ReferenceProfile]:
        out: list[ReferenceProfile] = []
        for item in self.list_profiles():
            if item.status == ProfileStatus.DEPRECATED and not include_deprecated:
                continue
            if item.status == ProfileStatus.DRAFT:
                continue
            out.append(item.profile)
        return out


def _with_status_fields(
    profile: ReferenceProfile, *, status: ProfileStatus
) -> ReferenceProfile:
    return ReferenceProfile(
        profile_id=profile.profile_id,
        stroke_type=profile.stroke_type,
        handedness=profile.handedness,
        camera_view=profile.camera_view,
        skill_level=profile.skill_level,
        metrics=dict(profile.metrics),
        provisional=profile.provisional and status != ProfileStatus.VALIDATED,
        profile_version=profile.profile_version,
        source=profile.source,
        notes=profile.notes,
        status=status.value,
        dataset_id=profile.dataset_id,
        dataset_version=profile.dataset_version,
        dataset_fingerprint=profile.dataset_fingerprint,
        sample_count=profile.sample_count,
    )


def _next_version_label(current: str) -> str:
    raw = current.strip().lower()
    if raw.startswith("v") and raw[1:].isdigit():
        return f"v{int(raw[1:]) + 1}"
    # builder versions like 1.0.0 → keep as v2 style bump from trailing int
    if raw.replace(".", "").isdigit():
        parts = raw.split(".")
        try:
            parts[-1] = str(int(parts[-1]) + 1)
            return ".".join(parts)
        except ValueError:
            pass
    return "v2"


def _strip_version_suffix(profile_id: str) -> str:
    for sep in ("_v", "_built_v"):
        idx = profile_id.rfind(sep)
        if idx > 0:
            return profile_id[:idx]
    return profile_id


def _canonical_number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return None


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
