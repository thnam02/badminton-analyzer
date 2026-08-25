"""Pose-derived player bounding box for DensePose crop fallback."""

from __future__ import annotations

from app.schemas.pose import PoseFrame


def pose_player_bbox(
    pose_frame: PoseFrame | None,
    width: int,
    height: int,
    *,
    min_confidence: float,
    padding_ratio: float = 0.20,
) -> tuple[int, int, int, int] | None:
    """Axis-aligned bbox around confident keypoints with padding."""
    if pose_frame is None or not pose_frame.keypoints:
        return None

    xs: list[float] = []
    ys: list[float] = []
    for kp in pose_frame.keypoints.values():
        if kp.confidence < min_confidence:
            continue
        xs.append(kp.x * width)
        ys.append(kp.y * height)

    if len(xs) < 2:
        return None

    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    pad_x = max(8.0, (x2 - x1) * padding_ratio)
    pad_y = max(8.0, (y2 - y1) * padding_ratio)

    ix1 = max(0, int(round(x1 - pad_x)))
    iy1 = max(0, int(round(y1 - pad_y)))
    ix2 = min(width, int(round(x2 + pad_x)))
    iy2 = min(height, int(round(y2 + pad_y)))

    if ix2 <= ix1 + 4 or iy2 <= iy1 + 4:
        return None
    return ix1, iy1, ix2, iy2
