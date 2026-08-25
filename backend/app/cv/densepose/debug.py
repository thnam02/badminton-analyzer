"""Debug visualizations and artifact export for DensePose muscle pipeline."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.cv.densepose.mapping import (
    PART_NAMES,
    DensePoseFrameResult,
    DensePoseInferenceDiagnostics,
)

logger = logging.getLogger(__name__)

# High-contrast solid BGR colors for each DensePose part (debug only).
PART_DEBUG_COLORS: dict[int, tuple[int, int, int]] = {
    1: (0, 0, 255),
    2: (0, 128, 255),
    3: (0, 255, 255),
    4: (0, 255, 0),
    5: (255, 255, 0),
    6: (255, 128, 0),
    7: (255, 0, 128),
    8: (128, 0, 255),
    9: (255, 0, 255),
    10: (200, 200, 50),
    11: (50, 200, 200),
    12: (200, 50, 200),
    13: (50, 50, 200),
    14: (200, 200, 200),
}


def render_densepose_parts_overlay(
    frame_bgr: np.ndarray,
    result: DensePoseFrameResult,
    *,
    alpha: float = 0.72,
) -> np.ndarray:
    """Render clearly visible solid body-part regions (temporary debug mode)."""
    overlay = frame_bgr.copy()
    for part_id, mask in sorted(result.part_masks.items()):
        color = PART_DEBUG_COLORS.get(int(part_id), (180, 180, 180))
        overlay[mask] = color
    if result.bbox is not None:
        x1, y1, x2, y2 = result.bbox
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 2, lineType=cv2.LINE_AA)
    return cv2.addWeighted(frame_bgr, 1.0 - alpha, overlay, alpha, 0)


def render_muscle_mask_debug(
    frame_bgr: np.ndarray,
    muscle_masks: dict[str, np.ndarray | None],
    *,
    alpha: float = 0.72,
) -> np.ndarray:
    """Solid colors for mapped muscle masks (debug only)."""
    from app.cv.muscles.atlas import MUSCLE_COLORS
    from app.cv.muscles.activation import MuscleGroup

    overlay = frame_bgr.copy()
    for name, mask in muscle_masks.items():
        if mask is None or not mask.any():
            continue
        try:
            muscle = MuscleGroup(name)
            color = MUSCLE_COLORS[muscle]
        except ValueError:
            color = (128, 128, 128)
        overlay[mask] = color
    return cv2.addWeighted(frame_bgr, 1.0 - alpha, overlay, alpha, 0)


def log_frame_diagnostics(
    diag: DensePoseInferenceDiagnostics,
    *,
    involvements: dict[str, float] | None = None,
    muscle_mask_counts: dict[str, int] | None = None,
) -> None:
    payload: dict[str, Any] = {
        **diag.to_dict(),
        "part_names": {
            str(pid): PART_NAMES.get(pid, str(pid)) for pid in diag.detected_part_ids
        },
    }
    if involvements is not None:
        payload["muscle_activation"] = involvements
    if muscle_mask_counts is not None:
        payload["muscle_mask_pixel_counts"] = muscle_mask_counts

    logger.info("DensePose debug frame %s: %s", diag.frame_index, json.dumps(payload))


@dataclass
class DensePoseDebugSession:
    """Save debug PNGs + JSON for the first N frames of an analyze run."""

    output_dir: Path
    max_frames: int = 5
    _saved: int = 0
    logs: list[dict[str, Any]] = field(default_factory=list)

    def should_save(self, frame_index: int) -> bool:
        return self._saved < self.max_frames

    def record(
        self,
        frame_index: int,
        frame_bgr: np.ndarray,
        *,
        densepose: DensePoseFrameResult | None,
        diagnostics: DensePoseInferenceDiagnostics,
        involvements: dict[str, float],
        muscle_masks: dict[str, np.ndarray | None],
    ) -> None:
        if not self.should_save(frame_index):
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"frame_{frame_index:05d}"

        cv2.imwrite(str(self.output_dir / f"{prefix}_input.jpg"), frame_bgr)

        muscle_counts = {
            name: int(mask.sum()) if mask is not None else 0
            for name, mask in muscle_masks.items()
        }

        log_entry = {
            **diagnostics.to_dict(),
            "muscle_activation": involvements,
            "muscle_mask_pixel_counts": muscle_counts,
            "part_names": {
                str(pid): PART_NAMES.get(pid, str(pid))
                for pid in diagnostics.detected_part_ids
            },
        }
        self.logs.append(log_entry)
        log_frame_diagnostics(
            diagnostics,
            involvements=involvements,
            muscle_mask_counts=muscle_counts,
        )

        if densepose is not None:
            parts_vis = render_densepose_parts_overlay(frame_bgr, densepose)
            cv2.imwrite(str(self.output_dir / f"{prefix}_densepose_parts.jpg"), parts_vis)
            muscle_vis = render_muscle_mask_debug(frame_bgr, muscle_masks)
            cv2.imwrite(str(self.output_dir / f"{prefix}_muscle_masks.jpg"), muscle_vis)

        with (self.output_dir / f"{prefix}_diagnostics.json").open("w", encoding="utf-8") as fh:
            json.dump(log_entry, fh, indent=2)

        self._saved += 1
        if self._saved >= self.max_frames:
            summary_path = self.output_dir / "summary.json"
            summary_path.write_text(json.dumps(self.logs, indent=2), encoding="utf-8")
            logger.info("DensePose debug artifacts written to %s", self.output_dir)
