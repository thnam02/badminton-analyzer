"""Mesh recovery backends (WHAM / SMPLer-X)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from app.cv.mesh.types import MeshSequence
from app.schemas.pose import PoseSequence


class MeshRecoveryError(RuntimeError):
    """Mesh recovery failed or backend is unavailable."""


class MeshRecoveryBackend(ABC):
    """Recover a temporally stable SMPL/SMPL-X mesh sequence from video."""

    name: str = "base"

    @abstractmethod
    def is_available(self) -> bool:
        ...

    @abstractmethod
    def recover(
        self,
        video_path: Path,
        *,
        pose_sequence: PoseSequence | None = None,
        image_width: int,
        image_height: int,
    ) -> MeshSequence:
        ...
