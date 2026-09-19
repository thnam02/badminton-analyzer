"""Optional timeline debug images for worst contact-timing errors."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.validation.phase_contact_boundaries import BOUNDARY_NAMES
from app.validation.phase_contact_report import (
    PhaseContactValidationReport,
    VideoTimingResult,
)


def render_worst_timing_timelines(
    report: PhaseContactValidationReport,
    output_dir: Path,
    *,
    max_videos: int = 5,
    width: int = 900,
    height: int = 160,
) -> list[Path]:
    """Draw GT vs predicted boundary markers on a 1D timeline for worst videos.

    Ranking uses contact absolute frame error when present, else max boundary error.
    """
    ranked = sorted(
        report.videos,
        key=_video_error_score,
        reverse=True,
    )
    ranked = [v for v in ranked if _video_error_score(v) > 0][:max_videos]
    if not ranked:
        return []

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for result in ranked:
        canvas = _draw_timeline(result, width=width, height=height)
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in result.video)
        out_path = output_dir / f"phase_contact_val_worst_{safe}.png"
        if cv2.imwrite(str(out_path), canvas):
            written.append(out_path)

    report.debug_image_paths = [str(p) for p in written]
    return written


def _video_error_score(result: VideoTimingResult) -> float:
    if result.contact_frame_error is not None:
        return float(result.contact_frame_error)
    errs = [
        float(b.absolute_frame_error)
        for b in result.boundary_errors
        if b.absolute_frame_error is not None
    ]
    return max(errs) if errs else -1.0


def _draw_timeline(
    result: VideoTimingResult,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    canvas = np.full((height, width, 3), 32, dtype=np.uint8)
    # Axis
    y_gt, y_pred = height // 3, (2 * height) // 3
    cv2.line(canvas, (40, y_gt), (width - 20, y_gt), (80, 80, 80), 1)
    cv2.line(canvas, (40, y_pred), (width - 20, y_pred), (80, 80, 80), 1)
    cv2.putText(
        canvas,
        "GT",
        (8, y_gt + 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (0, 220, 0),
        1,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Pred",
        (4, y_pred + 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (0, 140, 255),
        1,
        lineType=cv2.LINE_AA,
    )

    frames = []
    for b in result.boundary_errors:
        frames.append(b.gt_frame_index)
        if b.pred_frame_index is not None:
            frames.append(b.pred_frame_index)
    if not frames:
        frames = [0, 1]
    f_min, f_max = min(frames), max(frames)
    if f_max <= f_min:
        f_max = f_min + 1

    def x_of(frame: int) -> int:
        t = (frame - f_min) / (f_max - f_min)
        return int(40 + t * (width - 60))

    colors = {
        "PREPARATION_END": (180, 180, 255),
        "BACKSWING_END": (255, 180, 120),
        "ACCELERATION_START": (120, 255, 255),
        "CONTACT": (0, 0, 255),
        "FOLLOW_THROUGH_END": (200, 200, 200),
    }

    for b in result.boundary_errors:
        color = colors.get(b.boundary, (200, 200, 200))
        xg = x_of(b.gt_frame_index)
        cv2.line(canvas, (xg, y_gt - 18), (xg, y_gt + 18), (0, 220, 0), 2)
        cv2.circle(canvas, (xg, y_gt), 4, (0, 220, 0), -1)
        if b.pred_frame_index is not None:
            xp = x_of(b.pred_frame_index)
            cv2.line(canvas, (xp, y_pred - 18), (xp, y_pred + 18), color, 2)
            cv2.circle(canvas, (xp, y_pred), 4, (0, 140, 255), -1)
            # Connector showing error span
            cv2.line(canvas, (xg, y_gt), (xp, y_pred), (90, 90, 90), 1)

    title = result.video
    if result.contact_frame_error is not None:
        title += f"  contact_|df|={result.contact_frame_error}"
        if result.contact_time_error_ms is not None:
            title += f"  |dt|={result.contact_time_error_ms:.1f}ms"
    if result.contact_type:
        title += f"  ({result.contact_type})"
    cv2.putText(
        canvas,
        title[:90],
        (40, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (240, 240, 240),
        1,
        lineType=cv2.LINE_AA,
    )
    # Legend of boundary short names
    legend_y = height - 12
    x = 40
    for name in BOUNDARY_NAMES:
        short = name.replace("_", "")[:6]
        cv2.putText(
            canvas,
            short,
            (x, legend_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            colors.get(name, (200, 200, 200)),
            1,
            lineType=cv2.LINE_AA,
        )
        x += 70
    return canvas
