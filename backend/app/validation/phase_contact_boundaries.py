"""Boundary names for offline phase / contact timing validation."""

from __future__ import annotations

from app.schemas.phases import SmashPhase

# Manual annotation keys (where applicable on a clip).
BOUNDARY_PREPARATION_END = "PREPARATION_END"
BOUNDARY_BACKSWING_END = "BACKSWING_END"
BOUNDARY_ACCELERATION_START = "ACCELERATION_START"
BOUNDARY_CONTACT = "CONTACT"
BOUNDARY_FOLLOW_THROUGH_END = "FOLLOW_THROUGH_END"

BOUNDARY_NAMES: tuple[str, ...] = (
    BOUNDARY_PREPARATION_END,
    BOUNDARY_BACKSWING_END,
    BOUNDARY_ACCELERATION_START,
    BOUNDARY_CONTACT,
    BOUNDARY_FOLLOW_THROUGH_END,
)

# Map annotation boundary → (phase, "start"|"end") on PhaseSequence segments.
# CONTACT is taken from ContactEvent (preferred) or phase estimated contact.
BOUNDARY_TO_SEGMENT: dict[str, tuple[SmashPhase, str]] = {
    BOUNDARY_PREPARATION_END: (SmashPhase.PREPARATION, "end"),
    BOUNDARY_BACKSWING_END: (SmashPhase.BACKSWING, "end"),
    BOUNDARY_ACCELERATION_START: (SmashPhase.ACCELERATION, "start"),
    BOUNDARY_FOLLOW_THROUGH_END: (SmashPhase.FOLLOW_THROUGH, "end"),
}

# Frame / time proximity tolerances for hit-rate reporting.
FRAME_TOLERANCES: tuple[int, ...] = (1, 2)
TIME_TOLERANCES_MS: tuple[int, ...] = (33, 50, 100)
