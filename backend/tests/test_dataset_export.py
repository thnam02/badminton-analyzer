"""Tests for dataset export and coach annotation schemas."""

from __future__ import annotations

import json
from pathlib import Path

from app.schemas.annotation import (
    ANNOTATION_SCHEMA_VERSION,
    CoachAnnotation,
    QualityRating,
    QualityScore,
)
from app.schemas.dataset import DATASET_EXPORT_VERSION, DatasetExport
from app.services.dataset_exporter import DatasetExporter
from app.services.video_service import (
    annotation_template_json_path_for,
    coach_annotation_json_path_for,
    dataset_export_json_path_for,
)


def test_coach_annotation_roundtrip(tmp_path: Path) -> None:
    ann = CoachAnnotation(
        analysis_id="abc",
        coach_id="coach_jane",
        annotated_at="2026-09-09T00:00:00+00:00",
        preparation_quality=QualityScore(QualityRating.GOOD.value, notes="solid"),
        kinetic_chain_timing=QualityScore(QualityRating.FAIR.value),
        contact_quality=QualityScore(QualityRating.EXCELLENT.value),
        follow_through_quality=QualityScore(QualityRating.POOR.value),
        overall_issue_labels=["LOW_KNEE_CONTRIBUTION", "late_contact"],
        free_text_notes="Needs better knee drive.",
        agrees_with_system_contact=False,
        corrected_contact_frame_index=31,
    )
    path = tmp_path / "abc_annotation_coach_jane.json"
    ann.save_json(path)
    loaded = CoachAnnotation.from_dict(json.loads(path.read_text(encoding="utf-8")))
    assert loaded.annotation_version == ANNOTATION_SCHEMA_VERSION
    assert loaded.coach_id == "coach_jane"
    assert loaded.preparation_quality.rating == "GOOD"
    assert loaded.preparation_quality.score == 3
    assert loaded.overall_issue_labels == ["LOW_KNEE_CONTRIBUTION", "late_contact"]
    assert loaded.corrected_contact_frame_index == 31


def test_multiple_coaches_independent_files(tmp_path: Path) -> None:
    stem = tmp_path / "run_pose.mp4"
    a = CoachAnnotation.blank_template(analysis_id="run", coach_id="jane")
    b = CoachAnnotation.blank_template(analysis_id="run", coach_id="lee")
    path_a = coach_annotation_json_path_for(stem, "jane")
    path_b = coach_annotation_json_path_for(stem, "lee")
    # Paths are under tmp parent of stem
    path_a = tmp_path / path_a.name
    path_b = tmp_path / path_b.name
    a.preparation_quality = QualityScore(QualityRating.GOOD.value)
    b.preparation_quality = QualityScore(QualityRating.POOR.value)
    a.save_json(path_a)
    b.save_json(path_b)
    assert path_a != path_b
    la = CoachAnnotation.from_dict(json.loads(path_a.read_text(encoding="utf-8")))
    lb = CoachAnnotation.from_dict(json.loads(path_b.read_text(encoding="utf-8")))
    assert la.preparation_quality.rating == "GOOD"
    assert lb.preparation_quality.rating == "POOR"


def test_dataset_exporter_assembles_without_recompute(tmp_path: Path) -> None:
    stem = tmp_path / "xyz_pose.mp4"
    phases_path = tmp_path / "xyz_pose_phases.json"
    metrics_path = tmp_path / "xyz_pose_stroke_metrics.json"
    contact_path = tmp_path / "xyz_contact.json"
    technique_path = tmp_path / "xyz_pose_technique.json"
    keyframes_path = tmp_path / "xyz_pose_keyframes.json"
    quality_path = tmp_path / "xyz_pose_video_quality.json"

    phases_path.write_text(
        json.dumps(
            {
                "video": "xyz_pose.mp4",
                "estimated_contact_frame_index": 28,
                "estimated_contact_timestamp": 1.4,
                "confidence": 0.8,
                "segments": [],
            }
        ),
        encoding="utf-8",
    )
    metrics_path.write_text(
        json.dumps(
            {
                "video": "xyz_pose.mp4",
                "estimated_contact_frame_index": 28,
                "contact_elbow_angle_deg": 155.0,
            }
        ),
        encoding="utf-8",
    )
    contact_path.write_text(
        json.dumps(
            {
                "contact_type": "KINEMATIC_ESTIMATE",
                "frame_index": 28,
                "timestamp": 1.4,
                "confidence": 0.8,
            }
        ),
        encoding="utf-8",
    )
    technique_path.write_text(
        json.dumps(
            {
                "video": "xyz_pose.mp4",
                "issues": [
                    {
                        "code": "LOW_KNEE_CONTRIBUTION",
                        "severity": "moderate",
                        "measured_value": 5.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    keyframes_path.write_text(
        json.dumps(
            {
                "video": "xyz_pose.mp4",
                "keyframes": [
                    {
                        "phase": "ESTIMATED_CONTACT",
                        "frame_index": 28,
                        "timestamp": 1.4,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    quality_path.write_text(
        json.dumps(
            {
                "video": "xyz_pose.mp4",
                "usable": True,
                "analysis_confidence": 0.9,
                "metrics": {"fps": 30.0, "width": 1280, "height": 720},
            }
        ),
        encoding="utf-8",
    )

    dataset_path, template_path, export = DatasetExporter().export_analysis(
        output_stem=stem,
        phases_json_path=phases_path,
        metrics_json_path=metrics_path,
        contact_json_path=contact_path,
        technique_json_path=technique_path,
        keyframes_json_path=keyframes_path,
        quality_json_path=quality_path,
    )

    assert dataset_path == dataset_export_json_path_for(stem)
    assert template_path == annotation_template_json_path_for(stem)
    assert dataset_path.is_file() and template_path.is_file()
    assert export.dataset_export_version == DATASET_EXPORT_VERSION
    assert export.analysis_id == "xyz"
    assert export.pose_metrics["contact_elbow_angle_deg"] == 155.0
    assert export.contact_event["frame_index"] == 28
    assert export.technique_issues[0]["code"] == "LOW_KNEE_CONTRIBUTION"
    assert export.keyframes[0]["frame_index"] == 28
    assert export.coach_annotations.annotations == []
    assert export.video_metadata["fps"] == 30.0

    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    assert payload["coach_annotations"]["annotation_count"] == 0
    assert payload["annotation_template"]["coach_id"] == "COACH_ID_HERE"
    assert "preparation_quality" in payload["annotation_template"]

    template = json.loads(template_path.read_text(encoding="utf-8"))
    assert template["analysis_id"] == "xyz"
    assert template["annotation_version"] == ANNOTATION_SCHEMA_VERSION


def test_dataset_export_paths() -> None:
    stem = Path("/out/abc_pose.mp4")
    assert dataset_export_json_path_for(stem).name == "abc_dataset.json"
    assert annotation_template_json_path_for(stem).name == "abc_annotation_template.json"
    assert (
        coach_annotation_json_path_for(stem, "coach jane").name
        == "abc_annotation_coach_jane.json"
    )
