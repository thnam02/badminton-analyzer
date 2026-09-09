"""Tests for research-only AQA scaffold (no training, no production wiring)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.research.aqa import get_aqa_model
from app.research.aqa.loaders import AQADataset, sample_from_dataset_export
from app.research.aqa.mock_model import MockActionQualityModel
from app.research.aqa.protocol import ActionQualityModel
from app.research.aqa.schemas import QUALITY_DIMENSION_NAMES, AQASample
from app.schemas.annotation import CoachAnnotation, QualityRating, QualityScore


def _write_mini_dataset(tmp_path: Path) -> Path:
    analysis_id = "aqa1"
    dataset = {
        "dataset_export_version": "1.0.0",
        "analysis_id": analysis_id,
        "created_at": "2026-09-09T00:00:00+00:00",
        "stroke_type": "SMASH",
        "video_metadata": {"fps": 30.0, "width": 1280, "height": 720},
        "pose_metrics": {
            "video": "aqa1_pose.mp4",
            "contact_elbow_angle_deg": 155.0,
            "knee_contribution_deg": 14.0,
            "peak_elbow_omega_offset_frames": -2,
            "follow_through_speed_ratio": 0.4,
            "peak_wrist_speed": 2.0,
        },
        "phases": {
            "estimated_contact_frame_index": 28,
            "segments": [
                {
                    "phase": "PREPARATION",
                    "start_frame_index": 0,
                    "end_frame_index": 7,
                },
                {
                    "phase": "ESTIMATED_CONTACT",
                    "start_frame_index": 28,
                    "end_frame_index": 28,
                },
            ],
        },
        "contact_event": {
            "contact_type": "KINEMATIC_ESTIMATE",
            "frame_index": 28,
            "timestamp": 1.4,
            "confidence": 0.8,
        },
        "technique_issues": [
            {
                "code": "LOW_KNEE_CONTRIBUTION",
                "severity": "moderate",
                "measured_value": 8.0,
            }
        ],
        "keyframes": [
            {
                "phase": "ESTIMATED_CONTACT",
                "frame_index": 28,
                "file_path": "/tmp/contact.jpg",
            }
        ],
        "artifact_refs": {
            "smoothed_pose_json": "aqa1_pose_smoothed.json",
            "stroke_metrics_json": "aqa1_pose_stroke_metrics.json",
        },
        "coach_annotations": {"annotations": []},
    }
    path = tmp_path / f"{analysis_id}_dataset.json"
    path.write_text(json.dumps(dataset), encoding="utf-8")

    pose = {
        "video": "aqa1_pose.mp4",
        "frames": [
            {
                "frame_index": 0,
                "timestamp": 0.0,
                "keypoints": {"right_wrist": {"x": 0.5, "y": 0.5, "confidence": 0.9}},
            }
        ],
    }
    (tmp_path / "aqa1_pose_smoothed.json").write_text(
        json.dumps(pose), encoding="utf-8"
    )

    ann = CoachAnnotation(
        analysis_id=analysis_id,
        coach_id="coach_a",
        preparation_quality=QualityScore(QualityRating.GOOD.value),
        kinetic_chain_timing=QualityScore(QualityRating.FAIR.value),
        contact_quality=QualityScore(QualityRating.EXCELLENT.value),
        follow_through_quality=QualityScore(QualityRating.GOOD.value),
        overall_issue_labels=["late_hip_drive"],
    )
    (tmp_path / f"{analysis_id}_annotation_coach_a.json").write_text(
        json.dumps(ann.to_dict()), encoding="utf-8"
    )
    return path


def test_mock_model_satisfies_protocol() -> None:
    model = get_aqa_model("mock")
    assert isinstance(model, ActionQualityModel)
    assert model.is_available()
    assert isinstance(model, MockActionQualityModel)


def test_loader_builds_sample_with_pose_labels_and_evidence(tmp_path: Path) -> None:
    dataset_path = _write_mini_dataset(tmp_path)
    sample = sample_from_dataset_export(
        dataset_path,
        outputs_dir=tmp_path,
        include_pose_frames=True,
    )
    assert sample.analysis_id == "aqa1"
    assert len(sample.pose_frames) == 1
    assert sample.rgb_keyframe_features.keyframe_paths == ["/tmp/contact.jpg"]
    assert "ESTIMATED_CONTACT" in sample.phase_normalized_motion.phases
    assert sample.coach_labels[0]["coach_id"] == "coach_a"
    assert sample.deterministic_evidence.technique_issues[0]["code"] == (
        "LOW_KNEE_CONTRIBUTION"
    )


def test_mock_predict_outputs_dimensions_issues_uncertainty(tmp_path: Path) -> None:
    sample = sample_from_dataset_export(
        _write_mini_dataset(tmp_path),
        outputs_dir=tmp_path,
        include_pose_frames=False,
    )
    pred = get_aqa_model().predict(sample)
    assert pred.research_only is True
    assert pred.affects_production_feedback is False
    names = [d.name for d in pred.quality_dimensions]
    assert list(names) == list(QUALITY_DIMENSION_NAMES)
    for dim in pred.quality_dimensions:
        assert dim.uncertainty.std >= 0.0
        assert 0.0 <= dim.uncertainty.confidence <= 1.0
        assert abs(sum(dim.rating_probabilities.values()) - 1.0) < 1e-5
    codes = {i.code for i in pred.issue_probabilities}
    assert "LOW_KNEE_CONTRIBUTION" in codes
    assert "late_hip_drive" in codes
    assert pred.deterministic_evidence is not None


def test_dataset_iteration(tmp_path: Path) -> None:
    _write_mini_dataset(tmp_path)
    ds = AQADataset(tmp_path, include_pose_frames=False)
    assert len(ds) == 1
    sample = ds[0]
    assert isinstance(sample, AQASample)
    assert list(ds)[0].analysis_id == "aqa1"


def test_unknown_backend_raises() -> None:
    with pytest.raises(ValueError, match="Unknown AQA backend"):
        get_aqa_model("st_gcn")


def test_analyze_route_does_not_import_aqa() -> None:
    """Guardrail: production analyze must stay free of research AQA imports."""
    route_path = (
        Path(__file__).resolve().parents[1] / "app" / "routes" / "analyze.py"
    )
    source = route_path.read_text(encoding="utf-8")
    assert "research.aqa" not in source
    assert "ActionQuality" not in source
    assert "get_aqa_model" not in source
