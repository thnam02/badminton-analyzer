"""Optional debug images for largest angle-error frames (offline only)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.schemas.pose import PoseSequence
from app.validation.angle_annotations import AngleValidationAnnotationSet
from app.validation.angle_joints import TRIPLET_BY_ANGLE
from app.validation.angle_report import AngleValidationReport
from app.validation.annotations import annotated_to_keypoint


def render_worst_angle_debug_images(
    *,
    video_path: Path,
    smoothed_pose: PoseSequence,
    annotations: AngleValidationAnnotationSet,
    report: AngleValidationReport,
    output_dir: Path,
    max_frames: int = 5,
) -> list[Path]:
    """Draw the three keypoints and predicted vs GT angle for worst frames.

    Green = GT geometry (from annotation keypoints when present, else pose),
    orange = predicted pose keypoints. Text shows GT vs pred degrees.
    """
    if max_frames <= 0:
        return []

    # Rank frames by max absolute error across angles.
    ranked = sorted(
        report.frames,
        key=lambda f: max(
            (e.absolute_error or -1.0 for e in f.angle_errors),
            default=-1.0,
        ),
        reverse=True,
    )
    ranked = [f for f in ranked if any(e.absolute_error is not None for e in f.angle_errors)]
    ranked = ranked[:max_frames]
    if not ranked:
        return []

    needed = {f.frame_index for f in ranked}
    images = _read_frames_by_index(Path(video_path), needed)
    pose_by = {f.frame_index: f for f in smoothed_pose.frames}
    ann_by = annotations.frame_by_index()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for result in ranked:
        frame = images.get(result.frame_index)
        if frame is None:
            continue
        pose = pose_by.get(result.frame_index)
        ann = ann_by.get(result.frame_index)
        canvas = frame.copy()
        h, w = canvas.shape[:2]

        # Prefer the angle with the largest error for visualization.
        worst = max(
            (e for e in result.angle_errors if e.absolute_error is not None),
            key=lambda e: float(e.absolute_error or 0.0),
            default=None,
        )
        if worst is None or worst.angle_name not in TRIPLET_BY_ANGLE:
            continue
        proximal, vertex, distal = TRIPLET_BY_ANGLE[worst.angle_name]

        gt_pts = _triplet_points_from_annotation(ann, proximal, vertex, distal, w, h)
        pred_pts = _triplet_points_from_pose(pose, proximal, vertex, distal, w, h)

        if gt_pts is not None:
            _draw_triplet(canvas, gt_pts, color=(0, 220, 0))
        if pred_pts is not None:
            _draw_triplet(canvas, pred_pts, color=(0, 128, 255))

        pred_txt = (
            f"{worst.pred_degrees:.1f}"
            if worst.pred_degrees is not None
            else "None"
        )
        label = (
            f"f{result.frame_index} {worst.angle_name} "
            f"GT={worst.gt_degrees:.1f} Pred={pred_txt} "
            f"|err|={worst.absolute_error:.1f}"
        )
        cv2.putText(
            canvas,
            label,
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            lineType=cv2.LINE_AA,
        )
        out_path = output_dir / f"angle_val_worst_f{result.frame_index:06d}.jpg"
        if cv2.imwrite(str(out_path), canvas):
            written.append(out_path)

    report.debug_image_paths = [str(p) for p in written]
    return written


def _triplet_points_from_annotation(ann, proximal, vertex, distal, w, h):
    if ann is None or not ann.keypoints:
        return None
    pts = []
    for name in (proximal, vertex, distal):
        kp = ann.keypoints.get(name)
        if kp is None:
            return None
        key = annotated_to_keypoint(kp)
        pts.append((int(round(key.x * w)), int(round(key.y * h))))
    return pts


def _triplet_points_from_pose(pose, proximal, vertex, distal, w, h):
    if pose is None:
        return None
    pts = []
    for name in (proximal, vertex, distal):
        kp = pose.keypoints.get(name)
        if kp is None:
            return None
        pts.append((int(round(kp.x * w)), int(round(kp.y * h))))
    return pts


def _draw_triplet(
    frame: np.ndarray,
    points: list[tuple[int, int]],
    *,
    color: tuple[int, int, int],
) -> None:
    p0, p1, p2 = points
    cv2.line(frame, p0, p1, color, 2, lineType=cv2.LINE_AA)
    cv2.line(frame, p1, p2, color, 2, lineType=cv2.LINE_AA)
    for p in points:
        cv2.circle(frame, p, 4, color, -1, lineType=cv2.LINE_AA)


def _read_frames_by_index(
    video_path: Path, frame_indices: set[int]
) -> dict[int, np.ndarray]:
    if not frame_indices:
        return {}
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return {}
    wanted = set(frame_indices)
    max_needed = max(wanted)
    out: dict[int, np.ndarray] = {}
    idx = 0
    try:
        while idx <= max_needed and wanted:
            ok, frame = capture.read()
            if not ok:
                break
            if idx in wanted:
                out[idx] = frame.copy()
                wanted.discard(idx)
            idx += 1
    finally:
        capture.release()
    return out
