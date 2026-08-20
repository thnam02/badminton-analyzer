"""Pure-PyTorch mmcv.ops.nms replacements for environments without mmcv._ext."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torchvision.ops import nms as tv_nms

array_like_type = Union[Tensor, np.ndarray]


def nms(
    boxes: array_like_type,
    scores: array_like_type,
    iou_threshold: float,
    offset: int = 0,
    score_threshold: float = 0,
    max_num: int = -1,
) -> Tuple[array_like_type, array_like_type]:
    del offset  # torchvision uses x2-x1 style boxes
    assert offset in (0, 1)
    is_numpy = isinstance(boxes, np.ndarray)
    if is_numpy:
        boxes_t = torch.from_numpy(boxes)
        scores_t = torch.from_numpy(scores)
    else:
        boxes_t = boxes
        scores_t = scores

    if score_threshold > 0:
        valid = scores_t > score_threshold
        boxes_t = boxes_t[valid]
        scores_t = scores_t[valid]
        valid_inds = torch.nonzero(valid, as_tuple=False).squeeze(1)
    else:
        valid_inds = None

    keep = tv_nms(boxes_t.float(), scores_t.float(), iou_threshold)
    if max_num > 0:
        keep = keep[:max_num]
    if valid_inds is not None:
        keep = valid_inds[keep]

    dets = torch.cat((boxes_t[keep], scores_t[keep].reshape(-1, 1)), dim=1)
    if is_numpy:
        return dets.cpu().numpy(), keep.cpu().numpy()
    return dets, keep


def soft_nms(
    boxes: array_like_type,
    scores: array_like_type,
    iou_threshold: float = 0.3,
    sigma: float = 0.5,
    min_score: float = 1e-3,
    method: str = "linear",
    offset: int = 0,
) -> Tuple[array_like_type, array_like_type]:
    # Soft-NMS fallback: hard NMS is sufficient for inference scaffolding.
    del sigma, method, offset
    return nms(boxes, scores, iou_threshold=iou_threshold, score_threshold=min_score)


def batched_nms(
    boxes: Tensor,
    scores: Tensor,
    idxs: Tensor,
    nms_cfg: Optional[Dict],
    class_agnostic: bool = False,
) -> Tuple[Tensor, Tensor]:
    if nms_cfg is None:
        scores_sorted, inds = scores.sort(descending=True)
        boxes = boxes[inds]
        return torch.cat([boxes, scores_sorted[:, None]], -1), inds

    nms_cfg_ = dict(nms_cfg)
    class_agnostic = nms_cfg_.pop("class_agnostic", class_agnostic)
    iou_threshold = float(nms_cfg_.pop("iou_threshold", nms_cfg_.pop("iou_thr", 0.5)))
    nms_cfg_.pop("type", None)
    max_num = int(nms_cfg_.pop("max_num", -1))
    nms_cfg_.pop("split_thr", None)
    # Drop unknown kwargs leftover from mmcv configs.
    nms_cfg_.clear()

    if class_agnostic:
        boxes_for_nms = boxes
    else:
        max_coordinate = boxes.max()
        offsets = idxs.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
        boxes_for_nms = boxes + offsets[:, None]

    # boxes may be 5-d; NMS uses first 4
    boxes_xyxy = boxes_for_nms[:, :4]
    keep = tv_nms(boxes_xyxy.float(), scores.float(), iou_threshold)
    if max_num > 0:
        keep = keep[:max_num]
    boxes_kept = boxes[keep]
    scores_kept = scores[keep]
    return torch.cat([boxes_kept, scores_kept[:, None]], -1), keep


def nms_match(dets: array_like_type, iou_threshold: float) -> List[array_like_type]:
    if dets.shape[0] == 0:
        return []
    if isinstance(dets, Tensor):
        boxes = dets[:, :4]
        scores = dets[:, 4]
        _, keep = nms(boxes, scores, iou_threshold)
        return [keep]
    boxes = dets[:, :4]
    scores = dets[:, 4]
    _, keep = nms(boxes, scores, iou_threshold)
    return [keep]


def nms_rotated(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError("nms_rotated requires mmcv ops")


def nms_quadri(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError("nms_quadri requires mmcv ops")
