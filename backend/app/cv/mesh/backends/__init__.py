"""Factory for mesh recovery backends."""

from __future__ import annotations

from app.config import settings
from app.cv.mesh.backends.base import MeshRecoveryBackend, MeshRecoveryError
from app.cv.mesh.backends.wham import WhamBackend


def get_mesh_backend(name: str | None = None) -> MeshRecoveryBackend:
    backend_name = (name or settings.mesh_backend or "wham").strip().lower()
    if backend_name in {"wham", "auto"}:
        # Milestone: WHAM only (no silent proxy / SMPLer-X fallback).
        return WhamBackend()
    raise MeshRecoveryError(
        f"MESH_BACKEND='{backend_name}' is not supported for this milestone. "
        "Set MESH_BACKEND=wham."
    )
