"""Pydantic models for OpenAI Structured Outputs (Responses API)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PrioritizedIssueModel(BaseModel):
    issue_code: str = Field(
        description="Must match a technique issue code from the evidence package."
    )
    priority: int = Field(ge=1, le=3, description="1 = most important.")
    explanation: str = Field(
        description="Plain-language coaching explanation grounded in provided metrics."
    )
    related_metric_hints: list[str] = Field(
        default_factory=list,
        description="Optional metric field names cited from evidence.metrics.",
    )


class StrengthModel(BaseModel):
    description: str
    evidence_refs: list[str] = Field(
        default_factory=list,
        description="Evidence keys or issue/metric references supporting the strength.",
    )


class DrillSuggestionModel(BaseModel):
    name: str
    description: str = Field(description="Practical badminton drill a player can try.")
    targets_issue_codes: list[str] = Field(default_factory=list)


class CoachingReportModel(BaseModel):
    """Strict schema enforced by OpenAI Structured Outputs."""

    summary: str = Field(description="Short overall coaching summary for the smash.")
    prioritized_issues: list[PrioritizedIssueModel] = Field(
        default_factory=list,
        max_length=3,
        description="Most important 1–3 issues from the provided technique_issues.",
    )
    strengths: list[StrengthModel] = Field(
        default_factory=list,
        description="Evidence-supported strengths only.",
    )
    drills: list[DrillSuggestionModel] = Field(
        default_factory=list,
        description="Practical drills targeting prioritized issues.",
    )
    caveats: list[str] = Field(
        default_factory=list,
        description="Uncertainty notes; deferrals when visuals conflict with metrics.",
    )
