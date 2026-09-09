"""Pose-guided racket detector (OpenCV only; no MMPose / YOLO required).

Searches a ROI extending past the hitting-hand wrist along the forearm axis
for elongated edge structure, then returns a bbox + tip-like point.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.config import settings
from app.cv.racket.base import RacketDetector, RawRacketDetection
from app.cv.racket.pose_context import (
    arm_keypoints,
    infer_hitting_hand,
    pose_frame_map,
)
from app.schemas.pose import PoseSequence


class PoseGuidedRacketDetector(RacketDetector):
    name = "pose_guided"

    def is_available(self) -> bool:
        return True

    def detect(
        self,
        video_path: Path,
        *,
        pose: PoseSequence | None = None,
        hitting_hand: str | None = None,
    ) -> list[RawRacketDetection]:
        if pose is None or not pose.frames:
            raise RuntimeError(
                "Pose-guided racket detection requires a PoseSequence "
                "(pass smoothed pose JSON). It does not run MMPose itself."
            )

        hand = infer_hitting_hand(
            pose,
            preferred=hitting_hand or settings.racket_hitting_hand or None,
            confidence_threshold=settings.racket_pose_confidence_threshold,
        )
        by_frame = pose_frame_map(pose)
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        detections: list[RawRacketDetection] = []
        frame_index = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                h, w = frame.shape[:2]
                pose_frame = by_frame.get(frame_index)
                det = _detect_in_frame(
                    frame,
                    pose_frame,
                    frame_index=frame_index,
                    hand=hand,
                    width=w,
                    height=h,
                    conf_thr=settings.racket_pose_confidence_threshold,
                    shaft_len_frac=settings.racket_shaft_length_frac,
                    roi_pad_frac=settings.racket_roi_pad_frac,
                )
                detections.append(det)
                frame_index += 1
        finally:
            capture.release()
        return detections


def _detect_in_frame(
    frame: np.ndarray,
    pose_frame,
    *,
    frame_index: int,
    hand: str,
    width: int,
    height: int,
    conf_thr: float,
    shaft_len_frac: float,
    roi_pad_frac: float,
) -> RawRacketDetection:
    empty = RawRacketDetection(
        frame_index=frame_index,
        x_px=None,
        y_px=None,
        x1_px=None,
        y1_px=None,
        x2_px=None,
        y2_px=None,
        confidence=0.0,
        hand=hand,
    )
    if pose_frame is None:
        return empty
    wrist, elbow, _shoulder = arm_keypoints(pose_frame, hand)
    if (
        wrist is None
        or elbow is None
        or wrist.confidence < conf_thr
        or elbow.confidence < conf_thr
    ):
        return empty

    wx, wy = wrist.x * width, wrist.y * height
    ex, ey = elbow.x * width, elbow.y * height
    dx, dy = wx - ex, wy - ey
    norm = float(np.hypot(dx, dy))
    if norm < 1e-3:
        return empty
    ux, uy = dx / norm, dy / norm
    shaft_px = shaft_len_frac * float(max(width, height))
    tip_x = wx + ux * shaft_px
    tip_y = wy + uy * shaft_px

    # ROI covering wrist → tip with padding.
    xs = [wx, tip_x]
    ys = [wy, tip_y]
    pad = roi_pad_frac * float(max(width, height))
    x1 = int(max(0, min(xs) - pad))
    y1 = int(max(0, min(ys) - pad))
    x2 = int(min(width - 1, max(xs) + pad))
    y2 = int(min(height - 1, max(ys) + pad))
    if x2 <= x1 + 2 or y2 <= y1 + 2:
        return empty

    roi = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    ys_e, xs_e = np.where(edges > 0)
    if len(xs_e) < 12:
        # Fall back to geometric shaft estimate when edges are weak.
        conf = float(min(1.0, 0.45 * min(wrist.confidence, elbow.confidence)))
        return RawRacketDetection(
            frame_index=frame_index,
            x_px=float(tip_x),
            y_px=float(tip_y),
            x1_px=float(x1),
            y1_px=float(y1),
            x2_px=float(x2),
            y2_px=float(y2),
            confidence=conf,
            hand=hand,
        )

    # Tighten bbox to edge mass; tip = farthest edge pixel from wrist along shaft.
    rx1 = int(xs_e.min()) + x1
    ry1 = int(ys_e.min()) + y1
    rx2 = int(xs_e.max()) + x1
    ry2 = int(ys_e.max()) + y1
    edge_xy = np.stack([xs_e + x1, ys_e + y1], axis=1).astype(np.float32)
    proj = (edge_xy[:, 0] - wx) * ux + (edge_xy[:, 1] - wy) * uy
    best = int(np.argmax(proj))
    tip_x = float(edge_xy[best, 0])
    tip_y = float(edge_xy[best, 1])
    edge_density = float(len(xs_e)) / float(max(1, (x2 - x1) * (y2 - y1)))
    conf = float(
        min(
            1.0,
            0.35
            + 0.4 * min(wrist.confidence, elbow.confidence)
            + 0.25 * min(1.0, edge_density * 40.0),
        )
    )
    return RawRacketDetection(
        frame_index=frame_index,
        x_px=tip_x,
        y_px=tip_y,
        x1_px=float(rx1),
        y1_px=float(ry1),
        x2_px=float(rx2),
        y2_px=float(ry2),
        confidence=conf,
        hand=hand,
    )
