"""Lightweight heuristic shuttle tracker for tests / offline smoke runs.

Not a substitute for TrackNetV3 — detects small fast-moving blobs via frame
differencing so the shuttle pipeline can be exercised without GPU weights.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.cv.shuttle.base import RawShuttleDetection, ShuttleTracker


class HeuristicShuttleTracker(ShuttleTracker):
    name = "heuristic"

    def is_available(self) -> bool:
        return True

    def track(self, video_path: Path) -> list[RawShuttleDetection]:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        detections: list[RawShuttleDetection] = []
        prev_gray: np.ndarray | None = None
        frame_index = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (5, 5), 0)
                if prev_gray is None:
                    detections.append(
                        RawShuttleDetection(
                            frame_index=frame_index,
                            x_px=None,
                            y_px=None,
                            visibility=0,
                            confidence=0.0,
                        )
                    )
                else:
                    diff = cv2.absdiff(gray, prev_gray)
                    _, mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
                    contours, _ = cv2.findContours(
                        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    best = None
                    best_area = 0.0
                    for contour in contours:
                        area = float(cv2.contourArea(contour))
                        # Shuttle-like: small moving blob.
                        if 2.0 <= area <= 400.0 and area > best_area:
                            best_area = area
                            best = contour
                    if best is None:
                        detections.append(
                            RawShuttleDetection(
                                frame_index=frame_index,
                                x_px=None,
                                y_px=None,
                                visibility=0,
                                confidence=0.0,
                            )
                        )
                    else:
                        moments = cv2.moments(best)
                        if moments["m00"] == 0:
                            detections.append(
                                RawShuttleDetection(
                                    frame_index=frame_index,
                                    x_px=None,
                                    y_px=None,
                                    visibility=0,
                                    confidence=0.0,
                                )
                            )
                        else:
                            cx = moments["m10"] / moments["m00"]
                            cy = moments["m01"] / moments["m00"]
                            conf = float(min(1.0, best_area / 80.0))
                            detections.append(
                                RawShuttleDetection(
                                    frame_index=frame_index,
                                    x_px=float(cx),
                                    y_px=float(cy),
                                    visibility=1,
                                    confidence=conf,
                                )
                            )
                prev_gray = gray
                frame_index += 1
        finally:
            capture.release()
        return detections
