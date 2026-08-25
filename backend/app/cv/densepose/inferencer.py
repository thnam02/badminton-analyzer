"""DensePose / Detectron2 inference for body-surface segmentation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from app.config import settings
from app.cv.densepose.bbox import pose_player_bbox
from app.cv.densepose.mapping import (
    DensePoseFrameResult,
    DensePoseInferenceDiagnostics,
)
from app.schemas.pose import PoseFrame

if TYPE_CHECKING:
    from detectron2.engine import DefaultPredictor

logger = logging.getLogger(__name__)

_ASSETS_DIR = Path(__file__).resolve().parent / "assets"
_DEFAULT_CONFIG = _ASSETS_DIR / "densepose_rcnn_R_50_FPN_s1x.yaml"
_DEFAULT_WEIGHTS_URL = (
    "https://dl.fbaipublicfiles.com/densepose/"
    "densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl"
)


class DensePoseError(RuntimeError):
    """Raised when DensePose inference fails while muscle overlay is required."""


class DensePoseInferencer:
    """Run DensePose on each frame; returns masks for the largest detected person."""

    def __init__(
        self,
        *,
        config_path: str | Path | None = None,
        weights_path: str | None = None,
        device: str | None = None,
        score_threshold: float | None = None,
        min_person_pixels: int | None = None,
        crop_padding: float | None = None,
    ) -> None:
        self._config_path = Path(config_path or settings.densepose_config or _DEFAULT_CONFIG)
        self._weights_path = weights_path or settings.densepose_weights or _DEFAULT_WEIGHTS_URL
        self._device = device or settings.device
        self._score_threshold = (
            settings.densepose_score_threshold
            if score_threshold is None
            else score_threshold
        )
        self._min_person_pixels = (
            settings.densepose_min_person_pixels
            if min_person_pixels is None
            else min_person_pixels
        )
        self._crop_padding = (
            settings.densepose_crop_padding
            if crop_padding is None
            else crop_padding
        )
        self._predictor: DefaultPredictor | None = None

    def _ensure_predictor(self) -> DefaultPredictor:
        if self._predictor is not None:
            return self._predictor

        try:
            from app.cv.densepose.compat import apply_pillow_shims

            apply_pillow_shims()
            from detectron2.config import get_cfg
            from detectron2.engine import DefaultPredictor
            from densepose.config import add_densepose_config
        except ImportError as exc:
            raise DensePoseError(
                "DensePose requires detectron2 and the DensePose project. "
                "Run: backend/scripts/bootstrap_densepose.sh"
            ) from exc
        except AttributeError as exc:
            if "LINEAR" in str(exc):
                raise DensePoseError(
                    "Pillow/Detectron2 compatibility failed (PIL.Image.LINEAR). "
                    "Restart the backend after updating app/cv/densepose/compat.py."
                ) from exc
            raise

        if not self._config_path.is_file():
            raise DensePoseError(
                f"DensePose config not found: {self._config_path}. "
                "Set DENSEPOSE_CONFIG or run bootstrap_densepose.sh"
            )

        cfg = get_cfg()
        add_densepose_config(cfg)
        cfg.merge_from_file(str(self._config_path))
        cfg.MODEL.WEIGHTS = self._weights_path
        cfg.MODEL.DEVICE = self._device
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self._score_threshold
        self._predictor = DefaultPredictor(cfg)
        logger.info("DensePose model loaded (device=%s)", self._device)
        return self._predictor

    def predict(
        self,
        frame_bgr: np.ndarray,
        *,
        frame_index: int = 0,
        pose_frame: PoseFrame | None = None,
    ) -> tuple[DensePoseFrameResult | None, DensePoseInferenceDiagnostics]:
        """Infer body-part masks; retry on pose crop when full-frame is weak."""
        diag = DensePoseInferenceDiagnostics(frame_index=frame_index)
        predictor = self._ensure_predictor()

        result, det_diag = self._predict_instances(
            predictor,
            frame_bgr,
            full_shape=frame_bgr.shape[:2],
            offset=(0, 0),
            mode="full_frame",
        )
        self._merge_detection_diag(diag, det_diag)

        weak = (
            result is None
            or result.nonzero_pixel_count() < self._min_person_pixels
        )
        if weak and pose_frame is not None:
            crop = pose_player_bbox(
                pose_frame,
                frame_bgr.shape[1],
                frame_bgr.shape[0],
                min_confidence=settings.pose_confidence_threshold,
                padding_ratio=self._crop_padding,
            )
            if crop is not None:
                diag.pose_crop_bbox = crop
                crop_result, crop_diag = self._predict_crop(
                    predictor, frame_bgr, crop
                )
                self._merge_detection_diag(diag, crop_diag)
                if crop_result is not None and (
                    result is None
                    or crop_result.nonzero_pixel_count()
                    > result.nonzero_pixel_count()
                ):
                    result = crop_result
                    diag.inference_mode = "pose_crop"

        diag.update_from_result(result)
        if result is None:
            diag.error = "no_valid_densepose_result"
        elif result.nonzero_pixel_count() < self._min_person_pixels:
            diag.error = (
                f"densepose_too_sparse ({result.nonzero_pixel_count()} px "
                f"< {self._min_person_pixels})"
            )

        return result, diag

    def _predict_crop(
        self,
        predictor: DefaultPredictor,
        frame_bgr: np.ndarray,
        crop: tuple[int, int, int, int],
    ) -> tuple[DensePoseFrameResult | None, DensePoseInferenceDiagnostics]:
        x1, y1, x2, y2 = crop
        crop_img = frame_bgr[y1:y2, x1:x2]
        return self._predict_instances(
            predictor,
            crop_img,
            full_shape=frame_bgr.shape[:2],
            offset=(x1, y1),
            mode="pose_crop",
        )

    def _predict_instances(
        self,
        predictor: DefaultPredictor,
        image_bgr: np.ndarray,
        *,
        full_shape: tuple[int, int],
        offset: tuple[int, int],
        mode: str,
    ) -> tuple[DensePoseFrameResult | None, DensePoseInferenceDiagnostics]:
        diag = DensePoseInferenceDiagnostics(frame_index=0, inference_mode=mode)
        outputs = predictor(image_bgr)
        instances = outputs["instances"]

        if not instances.has("pred_densepose") or len(instances) == 0:
            diag.num_detections = 0
            diag.error = "no_person_detections"
            return None, diag

        boxes = instances.pred_boxes.tensor.cpu().numpy()
        scores = instances.scores.cpu().numpy()
        diag.num_detections = len(boxes)
        diag.detection_scores = [float(s) for s in scores.tolist()]

        ox, oy = offset
        parent_h, parent_w = full_shape

        # Detection boxes in full-frame coordinates for logging.
        full_boxes: list[tuple[int, int, int, int]] = []
        for box in boxes.tolist():
            bx1, by1, bx2, by2 = (int(v) for v in box)
            full_boxes.append((bx1 + ox, by1 + oy, bx2 + ox, by2 + oy))
        diag.detection_boxes = full_boxes

        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        idx = int(np.argmax(areas))
        diag.selected_index = idx
        score = float(scores[idx])
        diag.selected_score = score

        if score < self._score_threshold:
            diag.error = f"score_below_threshold ({score:.3f})"
            return None, diag

        box = boxes[idx]
        lx1, ly1, lx2, ly2 = (int(v) for v in box)
        x1, y1 = lx1 + ox, ly1 + oy
        x2, y2 = lx2 + ox, ly2 + oy
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(parent_w, x2), min(parent_h, y2)
        diag.selected_bbox = (x1, y1, x2, y2)

        if x2 <= x1 or y2 <= y1:
            diag.error = "invalid_bbox"
            return None, diag

        dp = instances.pred_densepose[idx]
        labels = _coarse_segm_to_labels(dp)
        bw, bh = max(1, x2 - x1), max(1, y2 - y1)
        labels = cv2.resize(labels, (bw, bh), interpolation=cv2.INTER_NEAREST)
        if labels.ndim != 2:
            raise DensePoseError(
                f"DensePose label map must be 2D after resize, got shape {labels.shape}"
            )
        # Guard against off-by-one bbox clipping vs resize size.
        labels = labels[: y2 - y1, : x2 - x1]
        region_h, region_w = labels.shape
        y2, x2 = y1 + region_h, x1 + region_w

        person_mask = np.zeros((parent_h, parent_w), dtype=bool)
        person_mask[y1:y2, x1:x2] = labels > 0

        part_masks: dict[int, np.ndarray] = {}
        for part_id in range(1, int(labels.max()) + 1):
            region = labels == part_id
            if not region.any():
                continue
            mask = np.zeros((parent_h, parent_w), dtype=bool)
            mask[y1:y2, x1:x2] = region
            part_masks[part_id] = mask

        if not person_mask.any():
            diag.error = "empty_person_mask"
            return None, diag

        return DensePoseFrameResult(
            person_mask=person_mask,
            part_masks=part_masks,
            confidence=score,
            bbox=(x1, y1, x2, y2),
            inference_mode=mode,
        ), diag

    @staticmethod
    def _merge_detection_diag(
        target: DensePoseInferenceDiagnostics,
        source: DensePoseInferenceDiagnostics,
    ) -> None:
        if source.num_detections > target.num_detections:
            target.num_detections = source.num_detections
            target.detection_scores = source.detection_scores
            target.detection_boxes = source.detection_boxes
            target.selected_index = source.selected_index
            target.selected_bbox = source.selected_bbox
            target.selected_score = source.selected_score
        if source.error and not target.error:
            target.error = source.error


def _coarse_segm_to_labels(dp_output: object) -> np.ndarray:
    """Convert DensePose coarse_segm tensor to a 2D uint8 part-id map.

    Handles common layouts: ``[C,H,W]``, ``[1,C,H,W]``, or already ``[H,W]``.
    """
    if not hasattr(dp_output, "coarse_segm"):
        raise DensePoseError("DensePose output missing coarse_segm")

    segm = dp_output.coarse_segm
    if hasattr(segm, "detach"):
        segm = segm.detach().cpu()

    # Torch tensor path
    if hasattr(segm, "ndim"):
        if segm.ndim == 4:
            # [N,C,H,W] → take first instance, then argmax over channels
            segm = segm[0]
        if segm.ndim == 3:
            # [C,H,W]
            labels = segm.argmax(dim=0)
        elif segm.ndim == 2:
            labels = segm
        else:
            raise DensePoseError(
                f"Unexpected DensePose coarse_segm ndim={segm.ndim}, shape={tuple(segm.shape)}"
            )
        return labels.numpy().astype(np.uint8)

    arr = np.asarray(segm)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3:
        return arr.argmax(axis=0).astype(np.uint8)
    if arr.ndim == 2:
        return arr.astype(np.uint8)
    raise DensePoseError(f"Unexpected DensePose coarse_segm shape {arr.shape}")
