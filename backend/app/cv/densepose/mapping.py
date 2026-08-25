"""DensePose body-part labels, results, and inference diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


class DensePosePart(IntEnum):
    """Chart-based coarse segmentation labels (Detectron2 DensePose)."""

    BACKGROUND = 0
    TORSO = 1
    RIGHT_HAND = 2
    LEFT_HAND = 3
    RIGHT_FOOT = 4
    LEFT_FOOT = 5
    RIGHT_UPPER_LEG = 6
    LEFT_UPPER_LEG = 7
    RIGHT_LOWER_LEG = 8
    LEFT_LOWER_LEG = 9
    RIGHT_UPPER_ARM = 10
    LEFT_UPPER_ARM = 11
    RIGHT_LOWER_ARM = 12
    LEFT_LOWER_ARM = 13
    HEAD = 14


PART_NAMES: dict[int, str] = {int(p): p.name for p in DensePosePart}


@dataclass(slots=True)
class DensePoseFrameResult:
    """Per-frame DensePose output for the primary (largest) person."""

    person_mask: object  # np.ndarray bool H×W
    part_masks: dict[int, object] = field(default_factory=dict)
    confidence: float = 0.0
    bbox: tuple[int, int, int, int] | None = None
    inference_mode: str = "full_frame"

    def has_part(self, part: DensePosePart) -> bool:
        mask = self.part_masks.get(int(part))
        if mask is None:
            return False
        return bool(getattr(mask, "any", lambda: False)())

    def nonzero_pixel_count(self) -> int:
        mask = self.person_mask
        return int(getattr(mask, "sum", lambda: 0)())

    def detected_part_ids(self) -> list[int]:
        return sorted(self.part_masks.keys())

    def part_pixel_counts(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        for part_id, mask in self.part_masks.items():
            counts[int(part_id)] = int(getattr(mask, "sum", lambda: 0)())
        return counts


@dataclass(slots=True)
class DensePoseInferenceDiagnostics:
    frame_index: int
    num_detections: int = 0
    detection_scores: list[float] = field(default_factory=list)
    detection_boxes: list[tuple[int, int, int, int]] = field(default_factory=list)
    selected_index: int | None = None
    selected_bbox: tuple[int, int, int, int] | None = None
    selected_score: float | None = None
    inference_mode: str = "full_frame"
    pose_crop_bbox: tuple[int, int, int, int] | None = None
    densepose_nonzero_pixels: int = 0
    detected_part_ids: list[int] = field(default_factory=list)
    part_pixel_counts: dict[int, int] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "num_detections": self.num_detections,
            "detection_scores": self.detection_scores,
            "detection_boxes": self.detection_boxes,
            "selected_index": self.selected_index,
            "selected_bbox": self.selected_bbox,
            "selected_score": self.selected_score,
            "inference_mode": self.inference_mode,
            "pose_crop_bbox": self.pose_crop_bbox,
            "densepose_nonzero_pixels": self.densepose_nonzero_pixels,
            "detected_part_ids": self.detected_part_ids,
            "part_pixel_counts": self.part_pixel_counts,
            "error": self.error,
        }

    def update_from_result(self, result: DensePoseFrameResult | None) -> None:
        if result is None:
            self.densepose_nonzero_pixels = 0
            self.detected_part_ids = []
            self.part_pixel_counts = {}
            return
        self.inference_mode = result.inference_mode
        self.selected_bbox = result.bbox
        self.selected_score = result.confidence
        self.densepose_nonzero_pixels = result.nonzero_pixel_count()
        self.detected_part_ids = result.detected_part_ids()
        self.part_pixel_counts = result.part_pixel_counts()
