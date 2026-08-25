"""3D body mesh types for WHAM / SMPLer-X feasibility prototype."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class CameraParams:
    """Pinhole camera used to project mesh vertices into the image."""

    fx: float
    fy: float
    cx: float
    cy: float
    # Optional extrinsics: world→camera. Identity = vertices already in camera space.
    R: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float64))
    t: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))

    def to_dict(self) -> dict[str, Any]:
        return {
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "R": self.R.tolist(),
            "t": self.t.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CameraParams:
        return cls(
            fx=float(data["fx"]),
            fy=float(data["fy"]),
            cx=float(data["cx"]),
            cy=float(data["cy"]),
            R=np.asarray(data.get("R", np.eye(3)), dtype=np.float64),
            t=np.asarray(data.get("t", np.zeros(3)), dtype=np.float64),
        )

    @classmethod
    def from_image_size(
        cls, width: int, height: int, *, focal_scale: float = 1.2
    ) -> CameraParams:
        f = focal_scale * max(width, height)
        return cls(fx=f, fy=f, cx=width * 0.5, cy=height * 0.5)


@dataclass(slots=True)
class MeshFrame:
    """One frame of recovered SMPL/SMPL-X geometry in camera or world space."""

    frame_index: int
    vertices: np.ndarray  # (V, 3) float
    faces: np.ndarray  # (F, 3) int
    camera: CameraParams
    joints_3d: np.ndarray | None = None  # (J, 3) optional
    confidence: float = 1.0

    def projected_vertices(self) -> np.ndarray:
        """Project 3D vertices to pixel coordinates (N, 2)."""
        verts = self.vertices.astype(np.float64)
        cam = self.camera
        pts = (cam.R @ verts.T).T + cam.t
        z = np.clip(pts[:, 2], 1e-4, None)
        u = cam.fx * (pts[:, 0] / z) + cam.cx
        v = cam.fy * (pts[:, 1] / z) + cam.cy
        return np.stack([u, v], axis=1)


@dataclass(slots=True)
class MeshSequence:
    """Temporally recovered body mesh for one video / primary player."""

    video: str
    frames: list[MeshFrame] = field(default_factory=list)
    backend: str = "unknown"
    notes: str = (
        "Feasibility prototype: mesh recovery only. "
        "No muscle atlas / DensePose overlay."
    )
    alignment: dict[str, Any] | None = None

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    def frame_at(self, frame_index: int) -> MeshFrame | None:
        for frame in self.frames:
            if frame.frame_index == frame_index:
                return frame
        return None

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "video": self.video,
            "backend": self.backend,
            "frame_count": self.frame_count,
            "notes": self.notes,
            "alignment": self.alignment,
            "frames": [
                {
                    "frame_index": f.frame_index,
                    "num_vertices": int(f.vertices.shape[0]),
                    "num_faces": int(f.faces.shape[0]),
                    "confidence": f.confidence,
                    "camera": f.camera.to_dict(),
                }
                for f in self.frames
            ],
        }

    def save_summary_json(self, path: Path) -> Path:
        path.write_text(json.dumps(self.to_summary_dict(), indent=2), encoding="utf-8")
        return path
