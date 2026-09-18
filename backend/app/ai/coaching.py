"""Optional OpenAI Responses API coaching layer (isolated under app.ai).

Enabled only when OPENAI_COACHING_ENABLED=true and OPENAI_API_KEY is set.
Never recalculates biomechanics — explains EvidencePackage only.
"""

from __future__ import annotations

import base64
import logging
import mimetypes
from pathlib import Path
from typing import Any

from app.ai.prompts import SYSTEM_INSTRUCTIONS, build_user_prompt
from app.ai.schema_models import CoachingReportModel
from app.config import settings
from app.schemas.coaching import (
    CoachingReport,
    CoachingStatus,
    DrillSuggestion,
    PrioritizedIssue,
    Strength,
)
from app.schemas.evidence import EvidencePackage
from app.schemas.final_analysis import FinalAnalysisState

logger = logging.getLogger(__name__)


class CoachingConfigError(RuntimeError):
    """Raised when coaching is requested but not configured."""


class CoachingParseError(RuntimeError):
    """Raised when the model response cannot be parsed into CoachingReportModel."""


def is_coaching_configured() -> bool:
    """True when env enables coaching and an API key is present."""
    return bool(settings.openai_coaching_enabled and settings.openai_api_key.strip())


def generate_coaching_report_from_final(
    state: FinalAnalysisState,
    evidence: EvidencePackage,
    *,
    include_keyframes: bool = True,
) -> CoachingReport:
    """Generate coaching from evidence packaged against ``FinalAnalysisState``.

    Does not change OpenAI behavior — only asserts contact/phase identity so
    coaching cannot silently use a stale contact frame.
    """
    if evidence.contact.frame_index != state.contact.frame_index:
        raise ValueError(
            "EvidencePackage contact must match FinalAnalysisState.contact; "
            f"got evidence={evidence.contact.frame_index}, "
            f"state={state.contact.frame_index}."
        )
    if evidence.contact.contact_type != state.contact.contact_type:
        raise ValueError(
            "EvidencePackage contact_type must match FinalAnalysisState.contact."
        )
    return generate_coaching_report(
        evidence,
        include_keyframes=include_keyframes,
    )


def generate_coaching_report(
    evidence: EvidencePackage,
    *,
    include_keyframes: bool = True,
) -> CoachingReport:
    """Produce a CoachingReport; never raises for missing/invalid OpenAI output."""
    if not settings.openai_coaching_enabled:
        return build_skipped_report(
            evidence,
            reason="OpenAI coaching is disabled (OPENAI_COACHING_ENABLED=false).",
        )
    if not settings.openai_api_key.strip():
        return build_skipped_report(
            evidence,
            reason="OpenAI coaching enabled but OPENAI_API_KEY is not configured.",
        )

    try:
        parsed = _call_responses_api(evidence, include_keyframes=include_keyframes)
        report = coaching_report_from_model(
            parsed,
            status=CoachingStatus.OK.value,
            model_name=settings.openai_model,
            evidence=evidence,
        )
        return _sanitize_against_evidence(report, evidence)
    except CoachingParseError as exc:
        logger.warning("Coaching parse failed: %s", exc)
        return build_fallback_report(evidence, reason=str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Coaching request failed: %s", exc)
        return build_fallback_report(
            evidence,
            reason=f"OpenAI coaching request failed: {exc}",
        )


def validate_coaching_model(payload: dict[str, Any]) -> CoachingReportModel:
    """Validate a dict against the Structured Output schema (no API call)."""
    try:
        return CoachingReportModel.model_validate(payload)
    except Exception as exc:  # noqa: BLE001
        raise CoachingParseError(f"Invalid coaching payload: {exc}") from exc


def coaching_report_from_model(
    model: CoachingReportModel,
    *,
    status: str,
    model_name: str | None,
    evidence: EvidencePackage,
) -> CoachingReport:
    return CoachingReport(
        status=status,
        summary=model.summary,
        prioritized_issues=[
            PrioritizedIssue(
                issue_code=item.issue_code,
                priority=item.priority,
                explanation=item.explanation,
                related_metric_hints=list(item.related_metric_hints),
            )
            for item in model.prioritized_issues[:3]
        ],
        strengths=[
            Strength(
                description=item.description,
                evidence_refs=list(item.evidence_refs),
            )
            for item in model.strengths
        ],
        drills=[
            DrillSuggestion(
                name=item.name,
                description=item.description,
                targets_issue_codes=list(item.targets_issue_codes),
            )
            for item in model.drills
        ],
        caveats=list(model.caveats),
        model=model_name,
        evidence_version=evidence.evidence_version,
        video=evidence.video,
    )


def build_skipped_report(evidence: EvidencePackage, *, reason: str) -> CoachingReport:
    return CoachingReport(
        status=CoachingStatus.SKIPPED.value,
        summary="Coaching layer was not run.",
        prioritized_issues=[],
        strengths=[],
        drills=[],
        caveats=[reason],
        model=None,
        evidence_version=evidence.evidence_version,
        video=evidence.video,
    )


def build_fallback_report(evidence: EvidencePackage, *, reason: str) -> CoachingReport:
    """Deterministic fallback: surface top technique issues without LLM prose."""
    prioritized: list[PrioritizedIssue] = []
    for idx, issue in enumerate(evidence.technique_issues[:3]):
        code = str(issue.get("code", "UNKNOWN"))
        description = str(
            issue.get("description")
            or "See deterministic technique evaluation for details."
        )
        prioritized.append(
            PrioritizedIssue(
                issue_code=code,
                priority=idx + 1,
                explanation=description,
                related_metric_hints=[],
            )
        )
    return CoachingReport(
        status=CoachingStatus.FALLBACK.value,
        summary=(
            "Generative coaching unavailable; listing top deterministic "
            "technique issues from the evidence package."
        ),
        prioritized_issues=prioritized,
        strengths=[],
        drills=[],
        caveats=[
            reason,
            "Fallback report does not invent drills or visual interpretations.",
        ],
        model=None,
        evidence_version=evidence.evidence_version,
        video=evidence.video,
    )


def _sanitize_against_evidence(
    report: CoachingReport,
    evidence: EvidencePackage,
) -> CoachingReport:
    """Drop invented issue codes; keep at most 3 priorities."""
    known = {str(i.get("code")) for i in evidence.technique_issues if i.get("code")}
    filtered = [
        item
        for item in report.prioritized_issues
        if not known or item.issue_code in known
    ]
    # Re-rank remaining priorities 1..n
    filtered = filtered[:3]
    for i, item in enumerate(filtered):
        item.priority = i + 1
    if known and report.prioritized_issues and not filtered:
        report.caveats = list(report.caveats) + [
            "Model priorities referenced unknown issue codes; priorities cleared."
        ]
    report.prioritized_issues = filtered
    return report


def _call_responses_api(
    evidence: EvidencePackage,
    *,
    include_keyframes: bool,
) -> CoachingReportModel:
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover
        raise CoachingParseError(
            "openai package is not installed. pip install openai"
        ) from exc

    keyframe_paths: list[Path] = []
    if include_keyframes:
        keyframe_paths = _existing_keyframe_paths(evidence)

    user_content: list[dict[str, Any]] = [
        {
            "type": "input_text",
            "text": build_user_prompt(
                evidence.to_dict(),
                keyframe_count=len(keyframe_paths),
            ),
        }
    ]
    for path in keyframe_paths:
        user_content.append(_image_content_part(path))

    client = OpenAI(api_key=settings.openai_api_key)
    response = client.responses.parse(
        model=settings.openai_model,
        instructions=SYSTEM_INSTRUCTIONS,
        input=[{"role": "user", "content": user_content}],
        text_format=CoachingReportModel,
    )
    parsed = getattr(response, "output_parsed", None)
    if parsed is None:
        # Some SDK paths expose parsed content on output items.
        parsed = _extract_parsed_from_response(response)
    if parsed is None:
        raise CoachingParseError("Responses API returned no parsed CoachingReportModel")
    if isinstance(parsed, CoachingReportModel):
        return parsed
    if isinstance(parsed, dict):
        return validate_coaching_model(parsed)
    raise CoachingParseError(f"Unexpected parsed type: {type(parsed)!r}")


def _extract_parsed_from_response(response: Any) -> Any | None:
    output = getattr(response, "output", None) or []
    for item in output:
        content = getattr(item, "content", None) or []
        for part in content:
            parsed = getattr(part, "parsed", None)
            if parsed is not None:
                return parsed
    return None


def _existing_keyframe_paths(evidence: EvidencePackage) -> list[Path]:
    paths: list[Path] = []
    for kf in evidence.keyframes:
        raw = kf.get("file_path")
        if not raw:
            continue
        path = Path(str(raw))
        if path.is_file():
            paths.append(path)
    return paths


def _image_content_part(path: Path) -> dict[str, Any]:
    mime, _ = mimetypes.guess_type(str(path))
    if mime is None:
        mime = "image/jpeg"
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return {
        "type": "input_image",
        "image_url": f"data:{mime};base64,{b64}",
    }
