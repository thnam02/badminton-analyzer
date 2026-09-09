"""Mocked tests for optional OpenAI coaching layer."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.ai.coaching import (
    CoachingParseError,
    build_fallback_report,
    coaching_report_from_model,
    generate_coaching_report,
    is_coaching_configured,
    validate_coaching_model,
)
from app.ai.prompts import SYSTEM_INSTRUCTIONS, build_user_prompt
from app.ai.schema_models import CoachingReportModel
from app.config import settings
from app.schemas.coaching import CoachingStatus
from app.schemas.evidence import (
    EVIDENCE_VERSION,
    STROKE_TYPE_SMASH,
    ContactEvidence,
    EvidencePackage,
)
from app.schemas.video_quality import VideoQualityMetrics, VideoQualityReport


def _evidence(*, with_issues: bool = True) -> EvidencePackage:
    issues = []
    if with_issues:
        issues = [
            {
                "code": "INSUFFICIENT_ELBOW_EXTENSION",
                "phase": "ESTIMATED_CONTACT",
                "severity": "HIGH",
                "confidence": 0.8,
                "measured_value": 120.0,
                "reference_range": {"min": 150.0, "max": 180.0},
                "unit": "deg",
                "description": "Right elbow is not sufficiently extended at estimated contact.",
            },
            {
                "code": "LOW_KNEE_CONTRIBUTION",
                "phase": "ACCELERATION",
                "severity": "MEDIUM",
                "confidence": 0.7,
                "measured_value": 5.0,
                "reference_range": {"min": 12.0, "max": None},
                "unit": "deg",
                "description": "Limited knee extension from preparation to contact.",
            },
        ]
    return EvidencePackage(
        evidence_version=EVIDENCE_VERSION,
        video="smash.mp4",
        stroke_type=STROKE_TYPE_SMASH,
        handedness=None,
        analysis_confidence=0.7,
        video_quality=VideoQualityReport(
            video="smash.mp4",
            usable=True,
            analysis_confidence=0.8,
            metrics=VideoQualityMetrics(fps=30.0, width=1280, height=720),
        ).to_dict(),
        phase_boundaries=[],
        phase_confidence=0.75,
        contact=ContactEvidence(
            confidence=0.9,
            frame_index=28,
            timestamp=1.4,
        ),
        metrics={
            "contact_elbow_angle_deg": 120.0,
            "knee_contribution_deg": 5.0,
            "peak_wrist_speed": 2.0,
        },
        technique_issues=issues,
        technique_confidence=0.7,
        keyframes=[
            {
                "phase": "ESTIMATED_CONTACT",
                "frame_index": 28,
                "timestamp": 1.4,
                "file_path": "/nonexistent/ESTIMATED_CONTACT_f000028.jpg",
                "confidence": 0.9,
            }
        ],
        keyframes_output_dir="/nonexistent",
    )


def _valid_payload() -> dict:
    return {
        "summary": "Elbow extension at contact is the main limiter.",
        "prioritized_issues": [
            {
                "issue_code": "INSUFFICIENT_ELBOW_EXTENSION",
                "priority": 1,
                "explanation": "Evidence shows contact elbow at 120° vs 150° target.",
                "related_metric_hints": ["contact_elbow_angle_deg"],
            }
        ],
        "strengths": [
            {
                "description": "Peak wrist speed is present in the metrics.",
                "evidence_refs": ["metrics.peak_wrist_speed"],
            }
        ],
        "drills": [
            {
                "name": "Shadow smash extension",
                "description": "Slow-motion shadow smashes focusing on full elbow extension at contact height.",
                "targets_issue_codes": ["INSUFFICIENT_ELBOW_EXTENSION"],
            }
        ],
        "caveats": [
            "Contact is estimated from peak wrist speed, not shuttle tracking."
        ],
    }


def test_schema_validation_accepts_valid_payload() -> None:
    model = validate_coaching_model(_valid_payload())
    assert isinstance(model, CoachingReportModel)
    assert model.prioritized_issues[0].priority == 1
    report = coaching_report_from_model(
        model,
        status=CoachingStatus.OK.value,
        model_name="gpt-4o",
        evidence=_evidence(),
    )
    assert report.status == "ok"
    assert report.model == "gpt-4o"
    assert report.evidence_version == EVIDENCE_VERSION
    assert len(report.prioritized_issues) == 1


def test_schema_validation_rejects_invalid_payload() -> None:
    with pytest.raises(CoachingParseError):
        validate_coaching_model(
            {
                "summary": "bad",
                "prioritized_issues": [
                    {
                        "issue_code": "X",
                        "priority": 9,  # must be 1..3
                        "explanation": "nope",
                        "related_metric_hints": [],
                    }
                ],
                "strengths": [],
                "drills": [],
                "caveats": [],
            }
        )


def test_missing_api_configuration_skips(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "openai_coaching_enabled", True)
    monkeypatch.setattr(settings, "openai_api_key", "")
    assert is_coaching_configured() is False
    report = generate_coaching_report(_evidence())
    assert report.status == CoachingStatus.SKIPPED.value
    assert "OPENAI_API_KEY" in report.caveats[0]


def test_disabled_coaching_skips(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "openai_coaching_enabled", False)
    monkeypatch.setattr(settings, "openai_api_key", "sk-test")
    report = generate_coaching_report(_evidence())
    assert report.status == CoachingStatus.SKIPPED.value
    assert "disabled" in report.caveats[0].lower()


def test_invalid_response_uses_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "openai_coaching_enabled", True)
    monkeypatch.setattr(settings, "openai_api_key", "sk-test")
    monkeypatch.setattr(settings, "openai_model", "gpt-4o")

    def _boom(*_args, **_kwargs):
        raise CoachingParseError("not valid JSON schema output")

    monkeypatch.setattr("app.ai.coaching._call_responses_api", _boom)
    evidence = _evidence(with_issues=True)
    report = generate_coaching_report(evidence)
    assert report.status == CoachingStatus.FALLBACK.value
    assert report.prioritized_issues
    assert report.prioritized_issues[0].issue_code == "INSUFFICIENT_ELBOW_EXTENSION"
    assert "not valid JSON schema output" in report.caveats[0]


def test_fallback_builder_lists_deterministic_issues() -> None:
    evidence = _evidence(with_issues=True)
    report = build_fallback_report(evidence, reason="timeout")
    assert report.status == CoachingStatus.FALLBACK.value
    assert len(report.prioritized_issues) == 2
    assert report.drills == []
    assert "timeout" in report.caveats[0]


def test_prompt_forbids_unsupported_claims() -> None:
    text = SYSTEM_INSTRUCTIONS.lower()
    assert "muscle activation" in text
    assert "injury" in text
    assert "recalculate" in text
    assert "override" in text
    assert "defer" in text
    user = build_user_prompt(_evidence().to_dict(), keyframe_count=2)
    assert "EvidencePackage" in user
    assert "2 attached keyframe" in user


def test_successful_mocked_responses_api(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(settings, "openai_coaching_enabled", True)
    monkeypatch.setattr(settings, "openai_api_key", "sk-test")
    monkeypatch.setattr(settings, "openai_model", "gpt-4o-mini")

    # Create a tiny keyframe so include path is exercised without OpenAI.
    frame = tmp_path / "contact.jpg"
    frame.write_bytes(b"\xff\xd8\xff\xd9")  # minimal JPEG-like bytes
    evidence = _evidence()
    evidence.keyframes[0]["file_path"] = str(frame)

    parsed = validate_coaching_model(_valid_payload())

    def _fake_call(ev, *, include_keyframes: bool):
        assert include_keyframes is True
        assert ev.video == "smash.mp4"
        return parsed

    monkeypatch.setattr("app.ai.coaching._call_responses_api", _fake_call)
    report = generate_coaching_report(evidence)
    assert report.status == CoachingStatus.OK.value
    assert report.summary.startswith("Elbow extension")
    out = tmp_path / "coaching.json"
    report.save_json(out)
    assert out.exists()
    assert '"status": "ok"' in out.read_text(encoding="utf-8")


def test_sanitize_drops_unknown_issue_codes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "openai_coaching_enabled", True)
    monkeypatch.setattr(settings, "openai_api_key", "sk-test")

    bad = validate_coaching_model(
        {
            **_valid_payload(),
            "prioritized_issues": [
                {
                    "issue_code": "INVENTED_CODE",
                    "priority": 1,
                    "explanation": "should be dropped",
                    "related_metric_hints": [],
                }
            ],
        }
    )
    monkeypatch.setattr("app.ai.coaching._call_responses_api", lambda *a, **k: bad)
    report = generate_coaching_report(_evidence(with_issues=True))
    assert report.status == CoachingStatus.OK.value
    assert report.prioritized_issues == []
    assert any("unknown issue codes" in c.lower() for c in report.caveats)
