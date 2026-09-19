"""Canonical stroke-type identifiers for multi-stroke analysis."""

from __future__ import annotations

from enum import Enum


class StrokeType(str, Enum):
    FOREHAND_SMASH = "FOREHAND_SMASH"
    FOREHAND_CLEAR = "FOREHAND_CLEAR"


# User-facing / API aliases → canonical StrokeType.
_STROKE_ALIASES: dict[str, StrokeType] = {
    "SMASH": StrokeType.FOREHAND_SMASH,
    "FOREHAND_SMASH": StrokeType.FOREHAND_SMASH,
    "SMASH_FOREHAND": StrokeType.FOREHAND_SMASH,
    "FOREHANDSMASH": StrokeType.FOREHAND_SMASH,
    "CLEAR": StrokeType.FOREHAND_CLEAR,
    "FOREHAND_CLEAR": StrokeType.FOREHAND_CLEAR,
    "CLEAR_FOREHAND": StrokeType.FOREHAND_CLEAR,
    "FOREHANDCLEAR": StrokeType.FOREHAND_CLEAR,
}


SUPPORTED_STROKE_TYPES: tuple[StrokeType, ...] = (
    StrokeType.FOREHAND_SMASH,
    StrokeType.FOREHAND_CLEAR,
)


class UnsupportedStrokeError(ValueError):
    """Raised when the requested stroke type has no registered analyzer."""


def normalize_stroke_type(value: str | StrokeType | None) -> StrokeType:
    """Normalize API / legacy labels to a canonical ``StrokeType``."""
    if isinstance(value, StrokeType):
        return value
    raw = str(value or "FOREHAND_SMASH").strip().upper().replace("-", "_").replace(" ", "_")
    if raw in _STROKE_ALIASES:
        return _STROKE_ALIASES[raw]
    raise UnsupportedStrokeError(
        f"Unsupported stroke type '{value}'. "
        f"Supported: {[s.value for s in SUPPORTED_STROKE_TYPES]}"
    )


def stroke_type_label(value: str | StrokeType | None) -> str:
    try:
        st = normalize_stroke_type(value)
    except UnsupportedStrokeError:
        return str(value or "Stroke").replace("_", " ").title()
    if st == StrokeType.FOREHAND_SMASH:
        return "Forehand Smash"
    if st == StrokeType.FOREHAND_CLEAR:
        return "Forehand Clear"
    return st.value.replace("_", " ").title()


def reference_stroke_key(value: str | StrokeType | None) -> str:
    """Key used on ReferenceProfile.stroke_type (smash catalog uses SMASH)."""
    st = normalize_stroke_type(value)
    if st == StrokeType.FOREHAND_SMASH:
        return "SMASH"
    return st.value
