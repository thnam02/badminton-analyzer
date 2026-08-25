"""SMPLer-X backend stub (swap-in when installed)."""

from __future__ import annotations

from pathlib import Path

from app.config import settings
from app.cv.mesh.backends.base import MeshRecoveryBackend, MeshRecoveryError
from app.cv.mesh.types import MeshSequence
from app.schemas.pose import PoseSequence


class SmplerXBackend(MeshRecoveryBackend):
    """Placeholder for caizhongang/SMPLer-X integration."""

    name = "smplerx"

    def is_available(self) -> bool:
        root = Path(settings.mesh_smplerx_root or "").expanduser()
        return bool(root) and root.is_dir()

    def recover(
        self,
        video_path: Path,
        *,
        pose_sequence: PoseSequence | None = None,
        image_width: int,
        image_height: int,
    ) -> MeshSequence:
        del video_path, pose_sequence, image_width, image_height
        raise MeshRecoveryError(
            "SMPLer-X backend is scaffolded but not wired yet. "
            "Set MESH_BACKEND=wham, or implement SmplerXBackend.recover() "
            "after installing SMPLer-X under MESH_SMPLERX_ROOT."
        )
