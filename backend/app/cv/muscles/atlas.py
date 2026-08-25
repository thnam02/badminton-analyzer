"""Muscle atlas: DensePose body parts → muscle regions and heat colors."""

from __future__ import annotations

import numpy as np

from app.cv.densepose.mapping import DensePoseFrameResult, DensePosePart
from app.cv.layers.helpers import keypoint_to_px
from app.cv.muscles.activation import MuscleGroup
from app.schemas.pose import PoseFrame

# Sports-science heat palette (BGR) — warm arm chain, cool core, green lower body.
MUSCLE_COLORS: dict[MuscleGroup, tuple[int, int, int]] = {
    MuscleGroup.RIGHT_DELTOID: (70, 130, 255),
    MuscleGroup.RIGHT_TRICEPS: (50, 90, 240),
    MuscleGroup.RIGHT_FOREARM: (90, 170, 255),
    MuscleGroup.CORE_OBLIQUE: (220, 140, 255),
    MuscleGroup.RIGHT_QUADRICEPS: (90, 230, 130),
    MuscleGroup.RIGHT_CALF: (190, 245, 110),
}

MUSCLE_DENSEPOSE_PARTS: dict[MuscleGroup, tuple[DensePosePart, ...]] = {
    MuscleGroup.RIGHT_DELTOID: (DensePosePart.RIGHT_UPPER_ARM,),
    MuscleGroup.RIGHT_TRICEPS: (DensePosePart.RIGHT_UPPER_ARM,),
    MuscleGroup.RIGHT_FOREARM: (DensePosePart.RIGHT_LOWER_ARM,),
    MuscleGroup.CORE_OBLIQUE: (DensePosePart.TORSO,),
    MuscleGroup.RIGHT_QUADRICEPS: (DensePosePart.RIGHT_UPPER_LEG,),
    MuscleGroup.RIGHT_CALF: (DensePosePart.RIGHT_LOWER_LEG,),
}


def build_muscle_masks(
    densepose: DensePoseFrameResult,
    pose_frame: PoseFrame | None,
    width: int,
    height: int,
    *,
    min_confidence: float,
) -> dict[MuscleGroup, np.ndarray | None]:
    """Combine DensePose part masks with optional pose-guided refinement."""
    masks: dict[MuscleGroup, np.ndarray | None] = {}
    for muscle, parts in MUSCLE_DENSEPOSE_PARTS.items():
        combined = _union_parts(densepose, parts, height, width)
        if combined is None:
            masks[muscle] = None
            continue

        if muscle is MuscleGroup.RIGHT_DELTOID:
            combined = _refine_upper_arm_proximal(
                combined, pose_frame, width, height, min_confidence
            )
        elif muscle is MuscleGroup.RIGHT_TRICEPS:
            combined = _refine_upper_arm_distal(
                combined, pose_frame, width, height, min_confidence
            )
        elif muscle is MuscleGroup.CORE_OBLIQUE:
            combined = _refine_torso_oblique(
                combined, pose_frame, width, height, min_confidence
            )

        masks[muscle] = combined if combined is not None and combined.any() else None
    return masks


def _union_parts(
    densepose: DensePoseFrameResult,
    parts: tuple[DensePosePart, ...],
    height: int,
    width: int,
) -> np.ndarray | None:
    combined = np.zeros((height, width), dtype=bool)
    found = False
    for part in parts:
        mask = densepose.part_masks.get(int(part))
        if mask is None:
            continue
        combined |= mask
        found = True
    return combined if found else None


def _refine_upper_arm_proximal(
    mask: np.ndarray,
    pose_frame: PoseFrame | None,
    width: int,
    height: int,
    min_confidence: float,
) -> np.ndarray | None:
    axis = _shoulder_elbow_axis(pose_frame, width, height, min_confidence)
    if axis is None:
        return mask
    shoulder, elbow, unit = axis
    ys, xs = np.nonzero(mask)
    pts = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    rel = pts - shoulder
    t = rel @ unit
    length = max(float(np.linalg.norm(elbow - shoulder)), 1.0)
    t_norm = t / length
    keep = (t_norm >= 0.0) & (t_norm <= 0.40)
    refined = np.zeros_like(mask)
    refined[ys[keep], xs[keep]] = True
    return refined if refined.any() else None


def _refine_upper_arm_distal(
    mask: np.ndarray,
    pose_frame: PoseFrame | None,
    width: int,
    height: int,
    min_confidence: float,
) -> np.ndarray | None:
    axis = _shoulder_elbow_axis(pose_frame, width, height, min_confidence)
    if axis is None:
        return mask
    shoulder, elbow, unit = axis
    ys, xs = np.nonzero(mask)
    pts = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    rel = pts - shoulder
    t = rel @ unit
    length = max(float(np.linalg.norm(elbow - shoulder)), 1.0)
    t_norm = t / length
    keep = (t_norm >= 0.28) & (t_norm <= 1.05)
    refined = np.zeros_like(mask)
    refined[ys[keep], xs[keep]] = True
    return refined if refined.any() else None


def _refine_torso_oblique(
    mask: np.ndarray,
    pose_frame: PoseFrame | None,
    width: int,
    height: int,
    min_confidence: float,
) -> np.ndarray | None:
    if pose_frame is None:
        return mask
    kp = pose_frame.keypoints
    ls = kp.get("left_shoulder")
    rs = kp.get("right_shoulder")
    if ls is None or rs is None:
        return mask
    lpx = keypoint_to_px(ls, width, height, min_confidence=min_confidence)
    rpx = keypoint_to_px(rs, width, height, min_confidence=min_confidence)
    if lpx is None or rpx is None:
        return mask
    mid_x = 0.5 * (lpx[0] + rpx[0])
    ys, xs = np.nonzero(mask)
    keep = xs.astype(np.float32) >= mid_x - 4.0
    refined = np.zeros_like(mask)
    refined[ys[keep], xs[keep]] = True
    return refined if refined.any() else None


def _shoulder_elbow_axis(
    pose_frame: PoseFrame | None,
    width: int,
    height: int,
    min_confidence: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if pose_frame is None:
        return None
    kp = pose_frame.keypoints
    shoulder_kp = kp.get("right_shoulder")
    elbow_kp = kp.get("right_elbow")
    if shoulder_kp is None or elbow_kp is None:
        return None
    shoulder = keypoint_to_px(shoulder_kp, width, height, min_confidence=min_confidence)
    elbow = keypoint_to_px(elbow_kp, width, height, min_confidence=min_confidence)
    if shoulder is None or elbow is None:
        return None
    s = np.array(shoulder, dtype=np.float32)
    e = np.array(elbow, dtype=np.float32)
    delta = e - s
    norm = float(np.linalg.norm(delta))
    if norm < 1e-3:
        return None
    return s, e, delta / norm
