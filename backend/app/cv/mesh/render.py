"""Semi-transparent mesh overlay + RTMPose reprojection sanity check."""

from __future__ import annotations

import numpy as np

from app.cv.mesh.align import ALIGNMENT_JOINTS, frame_alignment_errors, project_joints
from app.cv.mesh.types import MeshFrame
from app.schemas.pose import PoseFrame


def render_mesh_overlay(
    frame_bgr: np.ndarray,
    mesh_frame: MeshFrame,
    *,
    alpha: float = 0.45,
    color: tuple[int, int, int] = (80, 180, 255),
    draw_edges: bool = True,
) -> np.ndarray:
    """Rasterize projected mesh faces with a simple painter's algorithm."""
    import cv2

    h, w = frame_bgr.shape[:2]
    pts2d = mesh_frame.projected_vertices()
    verts = mesh_frame.vertices.astype(np.float64)
    cam = mesh_frame.camera
    cam_pts = (cam.R @ verts.T).T + cam.t
    faces = mesh_frame.faces

    face_depth = cam_pts[faces].mean(axis=1)[:, 2]
    order = np.argsort(-face_depth)

    overlay = frame_bgr.copy()
    fill = np.zeros_like(frame_bgr)
    edge_color = (40, 90, 200)

    for fi in order:
        tri = faces[fi]
        poly = pts2d[tri]
        if not np.isfinite(poly).all():
            continue
        if cam_pts[tri, 2].min() <= 1e-3:
            continue
        poly_i = np.round(poly).astype(np.int32)
        if (
            (poly_i[:, 0] < -w).any()
            or (poly_i[:, 0] > 2 * w).any()
            or (poly_i[:, 1] < -h).any()
            or (poly_i[:, 1] > 2 * h).any()
        ):
            continue
        cv2.fillConvexPoly(fill, poly_i, color, lineType=cv2.LINE_AA)
        if draw_edges:
            cv2.polylines(
                overlay,
                [poly_i],
                isClosed=True,
                color=edge_color,
                thickness=1,
                lineType=cv2.LINE_AA,
            )

    mask = fill.any(axis=2)
    out = frame_bgr.copy()
    out[mask] = (
        (1.0 - alpha) * frame_bgr[mask].astype(np.float32)
        + alpha * fill[mask].astype(np.float32)
    ).astype(np.uint8)
    edge_mask = np.any(overlay != frame_bgr, axis=2) & ~mask
    out[edge_mask] = overlay[edge_mask]
    return out


def draw_reprojection_residuals(
    frame_bgr: np.ndarray,
    mesh_frame: MeshFrame,
    pose_frame: PoseFrame | None,
    *,
    min_confidence: float,
) -> tuple[np.ndarray, float | None]:
    """Draw shoulder/elbow/hip residuals vs RTMPose; return mean pixel error."""
    import cv2

    h, w = frame_bgr.shape[:2]
    errors = frame_alignment_errors(
        mesh_frame,
        pose_frame,
        image_width=w,
        image_height=h,
        min_confidence=min_confidence,
    )
    if not errors:
        return frame_bgr, None

    out = frame_bgr.copy()
    projected = project_joints(mesh_frame)
    if projected is None or pose_frame is None:
        return out, None

    for j_idx, name in ALIGNMENT_JOINTS.items():
        if name not in errors or j_idx >= projected.shape[0]:
            continue
        kp = pose_frame.keypoints.get(name)
        if kp is None:
            continue
        px, py = int(kp.x * w), int(kp.y * h)
        mx, my = int(projected[j_idx, 0]), int(projected[j_idx, 1])
        cv2.circle(out, (px, py), 4, (0, 255, 0), -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (mx, my), 4, (0, 128, 255), -1, lineType=cv2.LINE_AA)
        cv2.line(out, (px, py), (mx, my), (255, 255, 255), 1, lineType=cv2.LINE_AA)

    mean_err = float(np.mean(list(errors.values())))
    cv2.putText(
        out,
        f"align mean={mean_err:.1f}px (sh/el/hip)",
        (12, h - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (240, 240, 240),
        1,
        lineType=cv2.LINE_AA,
    )
    return out, mean_err
