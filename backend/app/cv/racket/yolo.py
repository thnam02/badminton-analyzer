"""Optional YOLO racket detector adapter (weights optional).

Keeps Ultralytics I/O inside this module and converts immediately to
RawRacketDetection. Pose context filters detections toward the hitting hand.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2

from app.config import settings
from app.cv.racket.base import RacketDetector, RawRacketDetection
from app.cv.racket.pose_context import (
    arm_keypoints,
    infer_hitting_hand,
    pose_frame_map,
)
from app.schemas.pose import PoseSequence

logger = logging.getLogger(__name__)


class YoloRacketDetector(RacketDetector):
    """YOLO box detector for class ``racket`` (or configured class id)."""

    name = "yolo"

    def __init__(
        self,
        *,
        weights: str | Path | None = None,
        class_id: int | None = None,
        conf_threshold: float | None = None,
    ) -> None:
        self.weights = Path(weights or settings.racket_yolo_weights)
        self.class_id = (
            settings.racket_yolo_class_id if class_id is None else class_id
        )
        self.conf_threshold = (
            settings.racket_yolo_conf_threshold
            if conf_threshold is None
            else conf_threshold
        )
        self._model = None

    def is_available(self) -> bool:
        if not str(self.weights).strip() or not self.weights.is_file():
            return False
        try:
            import ultralytics  # noqa: F401
        except ImportError:
            return False
        return True

    def detect(
        self,
        video_path: Path,
        *,
        pose: PoseSequence | None = None,
        hitting_hand: str | None = None,
    ) -> list[RawRacketDetection]:
        if not self.is_available():
            raise RuntimeError(
                "YOLO racket detector unavailable. Set RACKET_YOLO_WEIGHTS "
                "to a .pt file and install ultralytics, or use "
                "RACKET_BACKEND=pose_guided."
            )
        from ultralytics import YOLO

        if self._model is None:
            self._model = YOLO(str(self.weights))

        hand = "RIGHT"
        by_frame: dict = {}
        if pose is not None and pose.frames:
            hand = infer_hitting_hand(
                pose,
                preferred=hitting_hand or settings.racket_hitting_hand or None,
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
                results = self._model.predict(
                    frame,
                    conf=self.conf_threshold,
                    verbose=False,
                )
                best = _pick_box(
                    results,
                    class_id=self.class_id,
                    pose_frame=by_frame.get(frame_index),
                    hand=hand,
                    width=w,
                    height=h,
                    conf_thr=settings.racket_pose_confidence_threshold,
                )
                if best is None:
                    detections.append(
                        RawRacketDetection(
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
                    )
                else:
                    x1, y1, x2, y2, conf = best
                    detections.append(
                        RawRacketDetection(
                            frame_index=frame_index,
                            x_px=0.5 * (x1 + x2),
                            y_px=0.5 * (y1 + y2),
                            x1_px=x1,
                            y1_px=y1,
                            x2_px=x2,
                            y2_px=y2,
                            confidence=float(conf),
                            hand=hand,
                        )
                    )
                frame_index += 1
        finally:
            capture.release()
        return detections


def _pick_box(
    results,
    *,
    class_id: int,
    pose_frame,
    hand: str,
    width: int,
    height: int,
    conf_thr: float,
) -> tuple[float, float, float, float, float] | None:
    if not results:
        return None
    boxes = getattr(results[0], "boxes", None)
    if boxes is None or len(boxes) == 0:
        return None

    wrist_xy = None
    if pose_frame is not None:
        wrist, _e, _s = arm_keypoints(pose_frame, hand)
        if wrist is not None and wrist.confidence >= conf_thr:
            wrist_xy = (wrist.x * width, wrist.y * height)

    candidates: list[tuple[float, float, float, float, float, float]] = []
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    clss = boxes.cls.cpu().numpy() if boxes.cls is not None else None
    for i in range(len(xyxy)):
        if clss is not None and int(clss[i]) != int(class_id):
            continue
        x1, y1, x2, y2 = map(float, xyxy[i])
        conf = float(confs[i])
        cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        if wrist_xy is None:
            score = conf
        else:
            dist = ((cx - wrist_xy[0]) ** 2 + (cy - wrist_xy[1]) ** 2) ** 0.5
            # Prefer detections near the hitting-hand wrist.
            score = conf - 0.002 * dist
        candidates.append((score, x1, y1, x2, y2, conf))

    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0], reverse=True)
    _score, x1, y1, x2, y2, conf = candidates[0]
    return x1, y1, x2, y2, conf
