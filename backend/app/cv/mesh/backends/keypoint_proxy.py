"""Keypoint-driven proxy mesh for overlay pipeline bring-up (not WHAM/SMPLer-X)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from app.cv.mesh.backends.base import MeshRecoveryBackend, MeshRecoveryError
from app.cv.mesh.types import CameraParams, MeshFrame, MeshSequence
from app.schemas.pose import PoseFrame, PoseSequence

logger = logging.getLogger(__name__)

# Unit capsule template (along +Y), scaled/rotated into bone segments.
_CAPSULE_RINGS = 8
_CAPSULE_SEGS = 10


def _capsule_mesh(
    p0: np.ndarray, p1: np.ndarray, radius: float
) -> tuple[np.ndarray, np.ndarray]:
    axis = p1 - p0
    length = float(np.linalg.norm(axis))
    if length < 1e-4:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int32)
    direction = axis / length
    # Build orthonormal basis
    helper = np.array([1.0, 0.0, 0.0]) if abs(direction[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x_axis = np.cross(direction, helper)
    x_axis /= max(np.linalg.norm(x_axis), 1e-6)
    y_axis = np.cross(direction, x_axis)

    verts: list[np.ndarray] = []
    for i in range(_CAPSULE_RINGS + 1):
        t = i / _CAPSULE_RINGS
        center = p0 + t * axis
        for j in range(_CAPSULE_SEGS):
            ang = 2 * np.pi * j / _CAPSULE_SEGS
            offset = radius * (np.cos(ang) * x_axis + np.sin(ang) * y_axis)
            verts.append(center + offset)
    vertices = np.asarray(verts, dtype=np.float64)

    faces: list[list[int]] = []
    for i in range(_CAPSULE_RINGS):
        for j in range(_CAPSULE_SEGS):
            a = i * _CAPSULE_SEGS + j
            b = i * _CAPSULE_SEGS + (j + 1) % _CAPSULE_SEGS
            c = (i + 1) * _CAPSULE_SEGS + j
            d = (i + 1) * _CAPSULE_SEGS + (j + 1) % _CAPSULE_SEGS
            faces.append([a, c, b])
            faces.append([b, c, d])
    return vertices, np.asarray(faces, dtype=np.int32)


_BONES: list[tuple[str, str, float]] = [
    ("left_shoulder", "right_shoulder", 0.035),
    ("left_shoulder", "left_elbow", 0.03),
    ("left_elbow", "left_wrist", 0.025),
    ("right_shoulder", "right_elbow", 0.03),
    ("right_elbow", "right_wrist", 0.025),
    ("left_shoulder", "left_hip", 0.04),
    ("right_shoulder", "right_hip", 0.04),
    ("left_hip", "right_hip", 0.035),
    ("left_hip", "left_knee", 0.035),
    ("left_knee", "left_ankle", 0.03),
    ("right_hip", "right_knee", 0.035),
    ("right_knee", "right_ankle", 0.03),
]


class KeypointProxyBackend(MeshRecoveryBackend):
    """Build a temporally smoothed capsule body from RTMPose (debug only)."""

    name = "keypoint_proxy"

    def is_available(self) -> bool:
        return True

    def recover(
        self,
        video_path: Path,
        *,
        pose_sequence: PoseSequence | None = None,
        image_width: int,
        image_height: int,
    ) -> MeshSequence:
        if pose_sequence is None or pose_sequence.frame_count == 0:
            raise MeshRecoveryError(
                "keypoint_proxy backend requires a PoseSequence from RTMPose"
            )

        logger.warning(
            "Using keypoint_proxy mesh backend (not WHAM/SMPLer-X). "
            "Install WHAM and set MESH_BACKEND=wham for real SMPL recovery."
        )

        cam = CameraParams.from_image_size(image_width, image_height)
        frames: list[MeshFrame] = []
        prev_verts: np.ndarray | None = None
        smooth = 0.55

        for pose_frame in pose_sequence.frames:
            verts, faces = _mesh_from_pose(pose_frame, image_width, image_height, cam)
            if verts.size == 0:
                continue
            if prev_verts is not None and prev_verts.shape == verts.shape:
                verts = smooth * verts + (1.0 - smooth) * prev_verts
            prev_verts = verts
            frames.append(
                MeshFrame(
                    frame_index=pose_frame.frame_index,
                    vertices=verts,
                    faces=faces,
                    camera=cam,
                    confidence=0.4,
                )
            )

        if not frames:
            raise MeshRecoveryError("keypoint_proxy produced no mesh frames")

        return MeshSequence(
            video=video_path.name,
            frames=frames,
            backend="keypoint_proxy",
            notes=(
                "Proxy capsule mesh from RTMPose only — not SMPL/WHAM. "
                "Use to validate overlay rendering before WHAM install."
            ),
        )


def _kp_cam(
    pose_frame: PoseFrame,
    name: str,
    width: int,
    height: int,
    cam: CameraParams,
    *,
    depth: float,
) -> np.ndarray | None:
    kp = pose_frame.keypoints.get(name)
    if kp is None or kp.confidence < 0.4:
        return None
    u, v = kp.x * width, kp.y * height
    x = (u - cam.cx) / cam.fx * depth
    y = (v - cam.cy) / cam.fy * depth
    return np.array([x, y, depth], dtype=np.float64)


def _mesh_from_pose(
    pose_frame: PoseFrame,
    width: int,
    height: int,
    cam: CameraParams,
) -> tuple[np.ndarray, np.ndarray]:
    depth = 2.2
    all_v: list[np.ndarray] = []
    all_f: list[np.ndarray] = []
    offset = 0
    for a, b, radius in _BONES:
        p0 = _kp_cam(pose_frame, a, width, height, cam, depth=depth)
        p1 = _kp_cam(pose_frame, b, width, height, cam, depth=depth)
        if p0 is None or p1 is None:
            continue
        # Scale radius in meters-ish relative to bone length
        bone_len = float(np.linalg.norm(p1 - p0))
        verts, faces = _capsule_mesh(p0, p1, radius=max(0.015, radius * bone_len * 8))
        if verts.size == 0:
            continue
        all_v.append(verts)
        all_f.append(faces + offset)
        offset += len(verts)

    # Torso plate between shoulders and hips when available
    ls = _kp_cam(pose_frame, "left_shoulder", width, height, cam, depth=depth)
    rs = _kp_cam(pose_frame, "right_shoulder", width, height, cam, depth=depth)
    lh = _kp_cam(pose_frame, "left_hip", width, height, cam, depth=depth)
    rh = _kp_cam(pose_frame, "right_hip", width, height, cam, depth=depth)
    if ls is not None and rs is not None and lh is not None and rh is not None:
        torso = np.stack([ls, rs, rh, lh], axis=0)
        # Extrude slightly along view for thickness
        normal = np.cross(rs - ls, lh - ls)
        nrm = np.linalg.norm(normal)
        if nrm > 1e-6:
            normal = normal / nrm * 0.04
            front = torso + normal
            back = torso - normal
            tv = np.concatenate([front, back], axis=0)
            tf = np.array(
                [
                    [0, 1, 2],
                    [0, 2, 3],
                    [4, 6, 5],
                    [4, 7, 6],
                    [0, 4, 5],
                    [0, 5, 1],
                    [1, 5, 6],
                    [1, 6, 2],
                    [2, 6, 7],
                    [2, 7, 3],
                    [3, 7, 4],
                    [3, 4, 0],
                ],
                dtype=np.int32,
            )
            all_v.append(tv)
            all_f.append(tf + offset)

    if not all_v:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int32)
    return np.concatenate(all_v, axis=0), np.concatenate(all_f, axis=0)
