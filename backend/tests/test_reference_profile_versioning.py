"""C6 reference profile provenance, freezing, and versioning tests."""

from __future__ import annotations

import pytest

from app.processing.reference_profile_selector import (
    MATCH_NONE,
    ReferenceProfileSelector,
    select_latest_validated_compatible,
)
from app.processing.technique import evaluate_technique
from app.schemas.reference import MetricReference, ReferenceProfile
from app.schemas.reference_profile_registry import (
    ImmutableProfileError,
    ProfileProvenance,
    ProfileStatus,
    ReferenceProfileRegistry,
    VersionedReferenceProfile,
    build_profile_manifest,
    bump_profile_version,
    deterministic_dataset_fingerprint,
)
from app.processing.reference_profiles import METRIC_CONTACT_ELBOW
from tests.test_technique import _build_pipeline
from tests.test_technique_calibration import _elbow_only_profile


def _versioned(
    profile_id: str = "smash_right_rear45_advanced_v1",
    *,
    version: str = "v1",
    status: ProfileStatus = ProfileStatus.DRAFT,
    skill: str = "advanced_expert",
    view: str = "rear_45",
) -> VersionedReferenceProfile:
    profile = ReferenceProfile(
        profile_id=profile_id,
        stroke_type="SMASH",
        handedness="RIGHT",
        camera_view=view,
        skill_level=skill,
        metrics={
            METRIC_CONTACT_ELBOW: MetricReference(
                metric_id=METRIC_CONTACT_ELBOW,
                unit="deg",
                median=165.0,
                lower_percentile=150.0,
                upper_percentile=180.0,
                sample_count=50,
                confidence=0.85,
                provisional=False,
                direction="higher_is_better",
                std=6.0,
                mean=165.0,
            )
        },
        provisional=False,
        profile_version=version,
        status=status.value,
        sample_count=50,
        dataset_id="reference_smash_v1",
        dataset_version="reference_smash_v1",
    )
    return VersionedReferenceProfile(
        profile=profile,
        status=status,
        provenance=ProfileProvenance(
            dataset_id="reference_smash_v1",
            dataset_version="reference_smash_v1",
            dataset_fingerprint="abc",
            sample_count=60,
            valid_sample_count=50,
            included_sample_ids=[f"s{i}" for i in range(50)],
            excluded_sample_ids=[f"x{i}" for i in range(10)],
            exclusion_reasons={"low_pose_quality": 6, "missing_contact": 4},
            builder_version="1.0.0",
        ),
    )


def test_immutable_validated_profiles() -> None:
    item = _versioned()
    item.freeze_as_validated()
    assert item.status == ProfileStatus.VALIDATED
    assert item.frozen is True
    with pytest.raises(ImmutableProfileError):
        item.replace_metrics({})


def test_new_version_creation() -> None:
    v1 = _versioned()
    v1.freeze_as_validated()
    new_profile = ReferenceProfile(
        profile_id="ignored",
        stroke_type="SMASH",
        handedness="RIGHT",
        camera_view="rear_45",
        skill_level="advanced_expert",
        metrics=dict(v1.profile.metrics),
        provisional=True,
        profile_version="v2",
    )
    v2 = bump_profile_version(v1, new_profile=new_profile)
    assert v2.profile_version == "v2"
    assert v2.profile_id.endswith("_v2")
    assert v2.status == ProfileStatus.DRAFT
    assert v2.frozen is False
    # v1 unchanged
    assert v1.profile_id.endswith("_v1")
    assert v1.status == ProfileStatus.VALIDATED


def test_deterministic_dataset_fingerprints() -> None:
    ids = ["b", "a", "c"]
    payloads = {
        "a": {"contact_elbow_angle_deg": 160.1234567, "path": "/tmp/x"},
        "b": {"contact_elbow_angle_deg": 155.0},
        "c": {"contact_elbow_angle_deg": 170.0},
    }
    fp1 = deterministic_dataset_fingerprint(
        sample_ids=ids,
        metric_payloads=payloads,
        metric_keys=["contact_elbow_angle_deg"],
    )
    fp2 = deterministic_dataset_fingerprint(
        sample_ids=list(reversed(ids)),
        metric_payloads=payloads,
        metric_keys=["contact_elbow_angle_deg"],
    )
    assert fp1 == fp2
    # Changing a metric value changes fingerprint.
    payloads2 = dict(payloads)
    payloads2["a"] = {"contact_elbow_angle_deg": 161.0}
    fp3 = deterministic_dataset_fingerprint(
        sample_ids=ids,
        metric_payloads=payloads2,
        metric_keys=["contact_elbow_angle_deg"],
    )
    assert fp3 != fp1


def test_exact_profile_resolution_latest_validated() -> None:
    v1 = _versioned("smash_right_rear45_advanced_v1", version="v1")
    v1.freeze_as_validated()
    v2_draft = bump_profile_version(v1, new_profile=v1.profile)
    # Only v1 validated.
    registry = ReferenceProfileRegistry([v1, v2_draft])
    selection = select_latest_validated_compatible(
        stroke_type="SMASH",
        handedness="RIGHT",
        camera_view="rear_45",
        skill_level="advanced",
        registry=registry,
    )
    assert selection.resolved_profile_id == "smash_right_rear45_advanced_v1"
    assert selection.resolved_profile_id != "latest"


def test_deprecated_profile_handling() -> None:
    item = _versioned()
    item.freeze_as_validated()
    item.mark_deprecated()
    assert item.status == ProfileStatus.DEPRECATED
    registry = ReferenceProfileRegistry([item])
    assert registry.as_reference_profiles() == []
    assert registry.as_reference_profiles(include_deprecated=True)[0].profile_id == item.profile_id


def test_reproduction_of_historical_analysis() -> None:
    _, _, _, _, metrics = _build_pipeline()
    metrics.contact_elbow_angle_deg = 120.0
    metrics.phase_confidence = 0.95
    v1 = _elbow_only_profile()
    v1 = ReferenceProfile(
        profile_id="hist_v1",
        stroke_type=v1.stroke_type,
        handedness=v1.handedness,
        camera_view=v1.camera_view,
        skill_level=v1.skill_level,
        metrics=dict(v1.metrics),
        provisional=False,
        profile_version="v1",
        status="VALIDATED",
        sample_count=40,
    )
    a = evaluate_technique(metrics, profile=v1, quality_confidence=0.95)
    # Later catalog adds v2 — historical call still uses explicit v1.
    v2 = ReferenceProfile(
        profile_id="hist_v2",
        stroke_type=v1.stroke_type,
        handedness=v1.handedness,
        camera_view=v1.camera_view,
        skill_level=v1.skill_level,
        metrics={
            METRIC_CONTACT_ELBOW: MetricReference(
                metric_id=METRIC_CONTACT_ELBOW,
                unit="deg",
                median=140.0,
                lower_percentile=100.0,
                upper_percentile=180.0,
                sample_count=40,
                confidence=0.85,
                provisional=False,
                direction="higher_is_better",
                std=6.0,
                mean=140.0,
            )
        },
        provisional=False,
        profile_version="v2",
        status="VALIDATED",
    )
    b = evaluate_technique(
        metrics,
        profiles=[v1, v2],
        profile_id="hist_v1",
        quality_confidence=0.95,
    )
    assert a.reference_profile_id == b.reference_profile_id == "hist_v1"
    assert a.to_dict()["issues"] == b.to_dict()["issues"]


def test_profile_selector_fallback_and_validated_only() -> None:
    draft = _versioned(status=ProfileStatus.DRAFT).profile
    validated = _versioned(
        "smash_right_rear45_advanced_v1", status=ProfileStatus.VALIDATED
    )
    validated.freeze_as_validated()
    selector = ReferenceProfileSelector([draft, validated.profile])
    none_sel = selector.select(
        stroke_type="SMASH",
        handedness="RIGHT",
        camera_view="rear_45",
        skill_level="advanced_expert",
        require_validated=True,
    )
    assert none_sel.has_valid_profile
    assert none_sel.resolved_profile_id == validated.profile_id

    empty = ReferenceProfileSelector([draft]).select(
        stroke_type="SMASH",
        handedness="RIGHT",
        camera_view="rear_45",
        require_validated=True,
    )
    assert empty.match_level == MATCH_NONE


def test_rejection_of_incompatible_profile_dimensions() -> None:
    right = _versioned().profile
    selector = ReferenceProfileSelector([right], allow_stroke_wildcard=False)
    selection = selector.select(
        stroke_type="SMASH",
        handedness="LEFT",
        camera_view="rear_45",
        skill_level="advanced_expert",
    )
    assert selection.match_level == MATCH_NONE


def test_manifest_provenance() -> None:
    item = _versioned()
    item.freeze_as_validated()
    manifest = build_profile_manifest(item)
    assert manifest["profile_id"] == item.profile_id
    assert manifest["included_samples"] == 50
    assert manifest["excluded_samples"] == 10
    assert manifest["exclusion_reasons"]["low_pose_quality"] == 6
    assert manifest["status"] == "VALIDATED"
    assert manifest["frozen"] is True


def test_selector_rejects_latest_literal() -> None:
    with pytest.raises(ValueError, match="latest"):
        ReferenceProfileSelector([]).select(profile_id="latest")
