"""Video I/O helpers: OpenCV read/write + FFmpeg browser-compatible encode."""

from __future__ import annotations

import shutil
import subprocess
import uuid
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from app.config import settings

# annotate_frame(frame, frame_index, fps) -> annotated BGR frame
FrameAnnotator = Callable[[np.ndarray, int, float], np.ndarray]


def new_upload_path(filename: str) -> Path:
    suffix = Path(filename).suffix.lower() or ".mp4"
    return settings.upload_dir / f"{uuid.uuid4().hex}{suffix}"


def new_output_path() -> Path:
    return settings.output_dir / f"{uuid.uuid4().hex}_pose.mp4"


def pose_json_path_for(video_path: Path) -> Path:
    """Map outputs/{id}_pose.mp4 -> outputs/{id}_pose.json (raw)."""
    return video_path.with_suffix(".json")


def smoothed_pose_json_path_for(video_path: Path) -> Path:
    """Map outputs/{id}_pose.mp4 -> outputs/{id}_pose_smoothed.json."""
    return video_path.with_name(f"{video_path.stem}_smoothed.json")


def angles_json_path_for(video_path: Path) -> Path:
    """Map outputs/{id}_pose.mp4 -> outputs/{id}_pose_angles.json."""
    return video_path.with_name(f"{video_path.stem}_angles.json")


def motion_json_path_for(video_path: Path) -> Path:
    """Map outputs/{id}_pose.mp4 -> outputs/{id}_pose_motion.json."""
    return video_path.with_name(f"{video_path.stem}_motion.json")


def process_video_frames(
    input_path: Path,
    output_path: Path,
    annotate_frame: FrameAnnotator,
) -> Path:
    """Read frames, annotate, write temp AVI/MP4, then remux/encode for browsers."""
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("Invalid video dimensions")

    raw_path = output_path.with_name(output_path.stem + "_raw.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(raw_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("Could not open VideoWriter for output")

    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            annotated = annotate_frame(frame, frame_index, float(fps))
            if annotated.shape[1] != width or annotated.shape[0] != height:
                annotated = cv2.resize(annotated, (width, height))
            writer.write(annotated)
            frame_index += 1
    finally:
        capture.release()
        writer.release()

    try:
        _ffmpeg_browser_mp4(raw_path, output_path, fps)
    finally:
        if raw_path.exists():
            raw_path.unlink(missing_ok=True)

    return output_path


def _ffmpeg_browser_mp4(src: Path, dst: Path, fps: float) -> None:
    """Re-encode to H.264 + yuv420p so Chrome/Safari can play the file."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        # Fall back to the OpenCV-written file if FFmpeg is unavailable.
        shutil.move(str(src), str(dst))
        return

    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(src),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-r",
        str(fps),
        "-movflags",
        "+faststart",
        "-an",
        str(dst),
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0 or not dst.exists():
        # Keep raw output rather than failing the whole request.
        if dst.exists():
            dst.unlink(missing_ok=True)
        shutil.move(str(src), str(dst))
        return
