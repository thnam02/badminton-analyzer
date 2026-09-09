"""Typed coaching report persisted as coaching.json."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class CoachingStatus(str, Enum):
    OK = "ok"
    SKIPPED = "skipped"
    FALLBACK = "fallback"
    ERROR = "error"


@dataclass(slots=True)
class PrioritizedIssue:
    issue_code: str
    priority: int
    explanation: str
    related_metric_hints: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "issue_code": self.issue_code,
            "priority": self.priority,
            "explanation": self.explanation,
            "related_metric_hints": list(self.related_metric_hints),
        }


@dataclass(slots=True)
class Strength:
    description: str
    evidence_refs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "evidence_refs": list(self.evidence_refs),
        }


@dataclass(slots=True)
class DrillSuggestion:
    name: str
    description: str
    targets_issue_codes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "targets_issue_codes": list(self.targets_issue_codes),
        }


@dataclass(slots=True)
class CoachingReport:
    """LLM coaching narrative constrained by deterministic EvidencePackage."""

    status: str
    summary: str
    prioritized_issues: list[PrioritizedIssue] = field(default_factory=list)
    strengths: list[Strength] = field(default_factory=list)
    drills: list[DrillSuggestion] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    model: str | None = None
    evidence_version: str | None = None
    video: str = ""
    notes: str = (
        "Coaching layer explains deterministic evidence only; "
        "it must not recalculate biomechanics measurements."
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "summary": self.summary,
            "prioritized_issues": [i.to_dict() for i in self.prioritized_issues],
            "strengths": [s.to_dict() for s in self.strengths],
            "drills": [d.to_dict() for d in self.drills],
            "caveats": list(self.caveats),
            "model": self.model,
            "evidence_version": self.evidence_version,
            "video": self.video,
            "notes": self.notes,
        }

    def save_json(self, path: Path) -> Path:
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path
