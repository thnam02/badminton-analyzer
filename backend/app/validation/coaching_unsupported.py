"""Patterns for claims the smash pipeline cannot currently measure."""

from __future__ import annotations

import re
from dataclasses import dataclass

# Categories the user asked to flag explicitly, plus close biomechanics cousins.
UNSUPPORTED_CLAIM_PATTERNS: tuple[tuple[str, str], ...] = (
    (
        "muscle_activation",
        r"\b(?:muscle\s+activ(?:ation|ity)|emg|recruit(?:ment|s)?\s+(?:the\s+)?(?:deltoid|quad|glute|core|forearm)|activator?\s+pattern)\b",
    ),
    (
        "joint_force_torque",
        r"\b(?:joint\s+(?:force|torque|moment)s?|(?:knee|elbow|shoulder|hip)\s+(?:torque|moment|joint\s+force)|net\s+joint\s+(?:force|moment))\b",
    ),
    (
        "ground_reaction_force",
        r"\b(?:ground[- ]reaction\s+force|grf|force\s+plate|vertical\s+ground\s+force)\b",
    ),
    (
        "injury_risk",
        r"\b(?:injury\s+risk|risk\s+of\s+injury|likely\s+to\s+(?:tear|sprain|injure)|will\s+(?:cause|lead\s+to)\s+(?:an?\s+)?injur)\b",
    ),
    (
        "exact_racket_speed",
        r"\b(?:racket\s+(?:head\s+)?speed\s+(?:of\s+)?\d+|exact\s+racket\s+(?:head\s+)?speed|\d+\s*(?:km/?h|mph)\s+(?:racket|racquet))\b",
    ),
    (
        "unavailable_biomechanics",
        r"\b(?:internal\s+joint\s+load|ligament\s+strain|cartilage\s+stress|muscle\s+force\s+estimation|inverse\s+dynamics\s+torque|metabolic\s+power|vo2)\b",
    ),
)

_COMPILED: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (category, re.compile(pattern, re.IGNORECASE))
    for category, pattern in UNSUPPORTED_CLAIM_PATTERNS
)


@dataclass(slots=True)
class UnsupportedClaim:
    category: str
    matched_text: str
    source_field: str

    def to_dict(self) -> dict[str, str]:
        return {
            "category": self.category,
            "matched_text": self.matched_text,
            "source_field": self.source_field,
        }


def detect_unsupported_claims(
    texts: list[tuple[str, str]],
) -> list[UnsupportedClaim]:
    """Scan ``(source_field, text)`` pairs for unavailable biomechanics claims."""
    found: list[UnsupportedClaim] = []
    seen: set[tuple[str, str, str]] = set()
    for source_field, text in texts:
        if not text:
            continue
        for category, pattern in _COMPILED:
            for match in pattern.finditer(text):
                snippet = match.group(0)
                key = (category, snippet.lower(), source_field)
                if key in seen:
                    continue
                seen.add(key)
                found.append(
                    UnsupportedClaim(
                        category=category,
                        matched_text=snippet,
                        source_field=source_field,
                    )
                )
    return found
