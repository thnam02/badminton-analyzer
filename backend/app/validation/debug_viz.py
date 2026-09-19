"""Optional debug images overlaying GT vs predicted joints (offline only)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.schemas.pose import Keypoint, PoseSequence
from app.validation.annotations import PoseValidationAnnotationSet, annotated_to_keypoint
from app.validation.report import PoseValidationReport


def render_worst_frame_debug_images(
    *,
    video_path: Path,
    predictions: PoseSequence,
    annotations: PoseValidationAnnotationSet,
    report: PoseValidationReport,
    output_dir: Path,
    max_frames: int = 5,
) -> list[Path]:
    """Write BGR JPEGs for the highest mean-error annotated frames.

    Green = ground truth, red/orange = prediction. Does not alter inference.
    """
    if max_frames <= 0:
        return []

    ranked = sorted(
        (f for f in report.frames if f.mean_error is not None),
        key=lambda f: float(f.mean_error or 0.0),
        reverse=True,
    )[:max_frames]
    if not ranked:
        return []

    needed = {f.frame_index for f in ranked}
    images = _read_frames_by_index(Path(video_path), needed)
    pred_by = {f.frame_index: f for f in predictions.frames}
    ann_by = annotations.frame_by_index()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for result in ranked:
        frame = images.get(result.frame_index)
        if frame is None:
            continue
        ann = ann_by.get(result.frame_index)
        pred = pred_by.get(result.frame_index)
        if ann is None:
            continue

        canvas = frame.copy()
        gt_kps = {
            name: annotated_to_keypoint(kp) for name, kp in ann.keypoints.items()
        }
        pred_kps = pred.keypoints if pred is not None else {}
        _draw_keypoints(canvas, gt_kps, color=(0, 220, 0), radius=5)
        _draw_keypoints(canvas, pred_kps, color=(0, 80, 255), radius=4)
        # Lines between matched GT/pred for the same joint.
        h, w = canvas.shape[:2]
        for detail in result.joint_errors:
            if not detail.predicted or detail.pred_x is None or detail.pred_y is None:
                continue
            p_gt = (int(round(detail.gt_x * w)), int(round(detail.gt_y * h)))
            p_pr = (int(round(detail.pred_x * w)), int(round(detail.pred_y * h)))
            cv2.line(canvas, p_gt, p_pr, (255, 255, 0), 1, lineType=cv2.LINE_AA)

        err_txt = (
            f"frame={result.frame_index} mean_npe={result.mean_error:.4f}"
            if result.mean_error is not None
            else f"frame={result.frame_index}"
        )
        cv2.putText(
            canvas,
            err_txt,
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            lineType=cv2.LINE_AA,
        )
        out_path = output_dir / f"pose_val_worst_f{result.frame_index:06d}.jpg"
        if cv2.imwrite(str(out_path), canvas):
            written.append(out_path)

    report.debug_image_paths = [str(p) for p in written]
    return written


def _draw_keypoints(
    frame: np.ndarray,
    keypoints: dict[str, Keypoint],
    *,
    color: tuple[int, int, int],
    radius: int,
) -> None:
    h, w = frame.shape[:2]
    for kp in keypoints.values():
        x = int(round(kp.x * w))
        y = int(round(kp.y * h))
        cv2.circle(frame, (x, y), radius, color, -1, lineType=cv2.LINE_AA)


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
