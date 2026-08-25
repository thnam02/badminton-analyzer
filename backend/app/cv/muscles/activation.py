"""Phase-based muscle involvement and temporal smoothing."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from app.schemas.phases import SmashPhase


class MuscleGroup(str, Enum):
    RIGHT_DELTOID = "right_deltoid"
    RIGHT_TRICEPS = "right_triceps"
    RIGHT_FOREARM = "right_forearm"
    CORE_OBLIQUE = "core_oblique"
    RIGHT_QUADRICEPS = "right_quadriceps"
    RIGHT_CALF = "right_calf"


_DEFAULT_TABLE: dict[SmashPhase, dict[MuscleGroup, float]] = {
    SmashPhase.PREPARATION: {
        MuscleGroup.RIGHT_DELTOID: 0.35,
        MuscleGroup.RIGHT_TRICEPS: 0.25,
        MuscleGroup.RIGHT_FOREARM: 0.20,
        MuscleGroup.CORE_OBLIQUE: 0.55,
        MuscleGroup.RIGHT_QUADRICEPS: 0.65,
        MuscleGroup.RIGHT_CALF: 0.45,
    },
    SmashPhase.BACKSWING: {
        MuscleGroup.RIGHT_DELTOID: 0.55,
        MuscleGroup.RIGHT_TRICEPS: 0.40,
        MuscleGroup.RIGHT_FOREARM: 0.45,
        MuscleGroup.CORE_OBLIQUE: 0.50,
        MuscleGroup.RIGHT_QUADRICEPS: 0.50,
        MuscleGroup.RIGHT_CALF: 0.35,
    },
    SmashPhase.ACCELERATION: {
        MuscleGroup.RIGHT_DELTOID: 0.75,
        MuscleGroup.RIGHT_TRICEPS: 0.80,
        MuscleGroup.RIGHT_FOREARM: 0.70,
        MuscleGroup.CORE_OBLIQUE: 0.70,
        MuscleGroup.RIGHT_QUADRICEPS: 0.85,
        MuscleGroup.RIGHT_CALF: 0.60,
    },
    SmashPhase.ESTIMATED_CONTACT: {
        MuscleGroup.RIGHT_DELTOID: 0.85,
        MuscleGroup.RIGHT_TRICEPS: 0.90,
        MuscleGroup.RIGHT_FOREARM: 0.95,
        MuscleGroup.CORE_OBLIQUE: 0.80,
        MuscleGroup.RIGHT_QUADRICEPS: 0.75,
        MuscleGroup.RIGHT_CALF: 0.55,
    },
    SmashPhase.FOLLOW_THROUGH: {
        MuscleGroup.RIGHT_DELTOID: 0.45,
        MuscleGroup.RIGHT_TRICEPS: 0.55,
        MuscleGroup.RIGHT_FOREARM: 0.60,
        MuscleGroup.CORE_OBLIQUE: 0.50,
        MuscleGroup.RIGHT_QUADRICEPS: 0.40,
        MuscleGroup.RIGHT_CALF: 0.30,
    },
}


@dataclass(frozen=True, slots=True)
class MuscleActivationMapper:
    """Map smash phase → muscle involvement in [0, 1] (demo only, not EMG)."""

    table: dict[SmashPhase, dict[MuscleGroup, float]] = field(
        default_factory=lambda: dict(_DEFAULT_TABLE)
    )
    default_involvement: float = 0.0

    def involvement(self, phase: SmashPhase | None, muscle: MuscleGroup) -> float:
        if phase is None:
            return self.default_involvement
        phase_table = self.table.get(phase)
        if phase_table is None:
            return self.default_involvement
        value = phase_table.get(muscle, self.default_involvement)
        return float(max(0.0, min(1.0, value)))

    def involvements_for_phase(
        self, phase: SmashPhase | None
    ) -> dict[MuscleGroup, float]:
        return {muscle: self.involvement(phase, muscle) for muscle in MuscleGroup}


class InvolvementSmoother:
    """EMA smoother for per-muscle involvement (reduces overlay flicker)."""

    def __init__(self, alpha: float = 0.4) -> None:
        self.alpha = max(0.0, min(1.0, float(alpha)))
        self._state: dict[MuscleGroup, float] = {}

    def reset(self) -> None:
        self._state.clear()

    def smooth(
        self, involvements: dict[MuscleGroup, float]
    ) -> dict[MuscleGroup, float]:
        out: dict[MuscleGroup, float] = {}
        for muscle in MuscleGroup:
            target = involvements.get(muscle, 0.0)
            if muscle not in self._state:
                self._state[muscle] = target
            else:
                prev = self._state[muscle]
                self._state[muscle] = self.alpha * target + (1.0 - self.alpha) * prev
            out[muscle] = self._state[muscle]
        return out
