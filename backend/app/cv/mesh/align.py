"""RTMPose ↔ SMPL joint reprojection sanity checks."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from app.cv.mesh.types import MeshFrame, MeshSequence
from app.schemas.pose import PoseFrame, PoseSequence

logger = logging.getLogger(__name__)

# Focus joints for smash feasibility logging.
ALIGNMENT_JOINTS: dict[int, str] = {
    16: "left_shoulder",
    17: "right_shoulder",
    18: "left_elbow",
    19: "right_elbow",
    1: "left_hip",
    2: "right_hip",
}


def project_joints(mesh_frame: MeshFrame) -> np.ndarray | None:
    """Project mesh_frame.joints_3d to pixels (J, 2)."""
    if mesh_frame.joints_3d is None:
        return None
    joints = mesh_frame.joints_3d.astype(np.float64)
    cam = mesh_frame.camera
    pts = (cam.R @ joints.T).T + cam.t
    z = np.clip(pts[:, 2], 1e-4, None)
    u = cam.fx * (pts[:, 0] / z) + cam.cx
    v = cam.fy * (pts[:, 1] / z) + cam.cy
    return np.stack([u, v], axis=1)


def frame_alignment_errors(
    mesh_frame: MeshFrame,
    pose_frame: PoseFrame | None,
    *,
    image_width: int,
    image_height: int,
    min_confidence: float,
) -> dict[str, float]:
    """Per-joint pixel errors for shoulder / elbow / hip."""
    projected = project_joints(mesh_frame)
    if projected is None or pose_frame is None:
        return {}

    errors: dict[str, float] = {}
    for j_idx, name in ALIGNMENT_JOINTS.items():
        if j_idx >= projected.shape[0]:
            continue
        kp = pose_frame.keypoints.get(name)
        if kp is None or kp.confidence < min_confidence:
            continue
        px, py = kp.x * image_width, kp.y * image_height
        mx, my = float(projected[j_idx, 0]), float(projected[j_idx, 1])
        errors[name] = float(np.hypot(px - mx, py - my))
    return errors


def log_sequence_alignment(
    mesh_sequence: MeshSequence,
    pose_sequence: PoseSequence | None,
    *,
    image_width: int,
    image_height: int,
    min_confidence: float,
    sample_every: int = 5,
) -> dict[str, Any]:
    """Log and summarize alignment errors across the sequence."""
    if pose_sequence is None:
        logger.info("Skipping RTMPose alignment check (no pose sequence)")
        return {"available": False}

    pose_by = {f.frame_index: f for f in pose_sequence.frames}
    per_joint: dict[str, list[float]] = {name: [] for name in ALIGNMENT_JOINTS.values()}
    sampled = 0

    for mesh_frame in mesh_sequence.frames:
        if sample_every > 1 and mesh_frame.frame_index % sample_every != 0:
            continue
        errs = frame_alignment_errors(
            mesh_frame,
            pose_by.get(mesh_frame.frame_index),
            image_width=image_width,
            image_height=image_height,
            min_confidence=min_confidence,
        )
        if not errs:
            continue
        sampled += 1
        for name, value in errs.items():
            per_joint[name].append(value)
        logger.info(
            "mesh align f=%d %s",
            mesh_frame.frame_index,
            " ".join(f"{k}={v:.1f}px" for k, v in sorted(errs.items())),
        )

    summary: dict[str, Any] = {"available": True, "sampled_frames": sampled, "joints": {}}
    for name, values in per_joint.items():
        if not values:
            continue
        arr = np.asarray(values, dtype=np.float64)
        joint_summary = {
            "mean_px": float(arr.mean()),
            "median_px": float(np.median(arr)),
            "max_px": float(arr.max()),
            "count": int(arr.size),
        }
        summary["joints"][name] = joint_summary
        logger.info(
            "mesh align summary %s mean=%.1fpx median=%.1fpx max=%.1fpx (n=%d)",
            name,
            joint_summary["mean_px"],
            joint_summary["median_px"],
            joint_summary["max_px"],
            joint_summary["count"],
        )

    if summary["joints"]:
        all_means = [v["mean_px"] for v in summary["joints"].values()]
        summary["overall_mean_px"] = float(np.mean(all_means))
        logger.info(
            "mesh align overall mean=%.1fpx (shoulder/elbow/hip)",
            summary["overall_mean_px"],
        )
    return summary
