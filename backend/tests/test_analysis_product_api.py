"""Backend tests for Phase D analysis aggregation / history."""

from __future__ import annotations

import json
from pathlib import Path

from app.services.analysis_aggregator import (
    compare_analyses,
    get_analysis_result,
    list_analyses,
    register_completed_analysis,
)
from app.services.analysis_index import load_index, manifest_path


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _seed_analysis(root: Path, analysis_id: str, *, elbow: float = 149.0) -> None:
    # Minimal pose video placeholder (empty file is enough for existence checks
    # only when we also have JSON — aggregator tolerates missing video).
    video = root / f"{analysis_id}_pose.mp4"
    video.write_bytes(b"")
    _write(
        root / f"{analysis_id}_pose_phases.json",
        {
            "video": video.name,
            "confidence": 0.9,
            "estimated_contact_frame_index": 66,
            "estimated_contact_timestamp": 2.2,
            "segments": [
                {
                    "phase": "PREPARATION",
                    "start_frame_index": 0,
                    "end_frame_index": 40,
                    "start_timestamp": 0.0,
                    "end_timestamp": 1.3,
                    "confidence": 0.85,
                },
                {
                    "phase": "ACCELERATION",
                    "start_frame_index": 41,
                    "end_frame_index": 65,
                    "start_timestamp": 1.35,
                    "end_timestamp": 2.15,
                    "confidence": 0.9,
                },
                {
                    "phase": "ESTIMATED_CONTACT",
                    "start_frame_index": 66,
                    "end_frame_index": 66,
                    "start_timestamp": 2.2,
                    "end_timestamp": 2.2,
                    "confidence": 0.95,
                },
                {
                    "phase": "FOLLOW_THROUGH",
                    "start_frame_index": 67,
                    "end_frame_index": 90,
                    "start_timestamp": 2.25,
                    "end_timestamp": 3.0,
                    "confidence": 0.8,
                },
            ],
        },
    )
    _write(
        root / f"{analysis_id}_pose_stroke_metrics.json",
        {
            "video": video.name,
            "estimated_contact_timestamp": 2.2,
            "estimated_contact_frame_index": 66,
            "phase_confidence": 0.9,
            "contact_elbow_angle_deg": elbow,
            "peak_wrist_speed": 1.7,
            "knee_contribution_deg": 12.0,
        },
    )
    _write(
        root / f"{analysis_id}_pose_technique.json",
        {
            "video": video.name,
            "confidence": 0.88,
            "reference_profile_id": "smash_right_side_provisional_v1",
            "issues": [
                {
                    "code": "INSUFFICIENT_ELBOW_EXTENSION",
                    "phase": "ESTIMATED_CONTACT",
                    "severity": "HIGH",
                    "status": "MAJOR",
                    "confidence": 0.9,
                    "measured_value": elbow,
                    "unit": "deg",
                    "reference_range": {"min": 150.0, "max": 180.0},
                    "reference_median": 165.0,
                    "reference_percentile": 6.0,
                    "reference_profile_id": "smash_right_side_provisional_v1",
                    "description": "Right elbow is not sufficiently extended.",
                },
                {
                    "code": "LOW_KNEE_CONTRIBUTION",
                    "phase": "ACCELERATION",
                    "severity": "LOW",
                    "status": "INSUFFICIENT_EVIDENCE",
                    "confidence": 0.3,
                    "measured_value": 4.0,
                    "unit": "deg",
                    "reference_range": {"min": 8.0, "max": 40.0},
                    "description": "Low confidence knee tracking.",
                    "status_reason": "Combined confidence below calibrated minimum",
                },
            ],
        },
    )
    _write(
        root / f"{analysis_id}_contact.json",
        {
            "contact_type": "KINEMATIC_ESTIMATE",
            "confidence": 0.81,
            "frame_index": 66,
            "timestamp": 2.2,
        },
    )
    _write(
        root / f"{analysis_id}_pose_video_quality.json",
        {"analysis_confidence": 0.89, "pose_confidence": 0.9},
    )
    _write(
        root / f"{analysis_id}_pose_coaching.json",
        {
            "status": "ok",
            "summary": "Reach higher at contact.",
            "prioritized_issues": [
                {
                    "issue_code": "INSUFFICIENT_ELBOW_EXTENSION",
                    "priority": 1,
                    "explanation": "Your arm remained more flexed than the reference group.",
                    "related_metric_hints": ["contact_elbow_angle_deg"],
                }
            ],
            "strengths": [
                {
                    "description": "Smooth preparation rhythm",
                    "evidence_refs": ["phase_confidence"],
                }
            ],
            "drills": [
                {
                    "name": "High-contact shadow smash",
                    "description": "10 reps × 3 sets. Reach upward through contact.",
                    "targets_issue_codes": ["INSUFFICIENT_ELBOW_EXTENSION"],
                }
            ],
            "caveats": [],
        },
    )


def test_aggregate_result_shape(tmp_path: Path) -> None:
    _seed_analysis(tmp_path, "aaa111")
    result = get_analysis_result("aaa111", output_dir=tmp_path)
    assert result["analysis_id"] == "aaa111"
    assert result["analysis_status"] == "COMPLETE"
    assert result["coaching_status"] == "COMPLETE"
    assert result["video"]["available"] is True
    assert len(result["phases"]) >= 3
    contact_phase = next(p for p in result["phases"] if p["is_contact_event"])
    assert contact_phase["seek_timestamp"] == 2.2
    findings = result["findings"]
    assert any(i["code"] == "INSUFFICIENT_ELBOW_EXTENSION" for i in findings)
    assert any(i["is_insufficient_evidence"] for i in result["issues"])
    assert result["coaching"]["main_focus"]["issue_code"] == "INSUFFICIENT_ELBOW_EXTENSION"
    assert result["coaching"]["drills"][0]["title"] == "High-contact shadow smash"
    assert result["confidence"]["overall"] in {"HIGH", "MODERATE", "LOW"}
    assert result["reference_comparisons"]


def test_coaching_unavailable_does_not_fail(tmp_path: Path) -> None:
    _seed_analysis(tmp_path, "bbb222")
    (tmp_path / "bbb222_pose_coaching.json").unlink()
    result = get_analysis_result("bbb222", output_dir=tmp_path)
    assert result["analysis_status"] == "COMPLETE"
    assert result["coaching_status"] == "UNAVAILABLE"
    assert result["coaching"]["available"] is False


def test_history_index_and_list(tmp_path: Path) -> None:
    _seed_analysis(tmp_path, "ccc333")
    register_completed_analysis("ccc333", output_dir=tmp_path)
    assert manifest_path(tmp_path).is_file()
    entries = load_index(tmp_path)
    assert entries[0].analysis_id == "ccc333"
    listed = list_analyses(output_dir=tmp_path, limit=10)
    assert any(a["analysis_id"] == "ccc333" for a in listed)


def test_compare_compatible(tmp_path: Path) -> None:
    _seed_analysis(tmp_path, "left1", elbow=149.0)
    _seed_analysis(tmp_path, "right1", elbow=158.0)
    # bump wrist on right
    metrics = json.loads((tmp_path / "right1_pose_stroke_metrics.json").read_text())
    metrics["peak_wrist_speed"] = 1.91
    (tmp_path / "right1_pose_stroke_metrics.json").write_text(json.dumps(metrics))
    cmp = compare_analyses("left1", "right1", output_dir=tmp_path)
    assert cmp["compatible"] is True
    labels = {r["label"] for r in cmp["rows"]}
    assert "Elbow contact" in labels
    elbow = next(r for r in cmp["rows"] if r["label"] == "Elbow contact")
    assert elbow["change"] in {"Improved", "Changed", "Similar"}
