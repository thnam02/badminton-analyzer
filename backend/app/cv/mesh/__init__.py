"""3D human mesh feasibility prototype (WHAM / SMPLer-X). No muscle atlas yet."""

from app.cv.mesh.backends import get_mesh_backend
from app.cv.mesh.backends.base import MeshRecoveryBackend, MeshRecoveryError
from app.cv.mesh.pipeline import recover_mesh_sequence, render_mesh_debug_video
from app.cv.mesh.types import CameraParams, MeshFrame, MeshSequence

__all__ = [
    "CameraParams",
    "MeshFrame",
    "MeshRecoveryBackend",
    "MeshRecoveryError",
    "MeshSequence",
    "get_mesh_backend",
    "recover_mesh_sequence",
    "render_mesh_debug_video",
]
