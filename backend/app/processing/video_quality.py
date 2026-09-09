"""Assess whether an uploaded clip is suitable for smash analysis.

Reuses pose sequences + video metadata only. Does not alter pose estimation,
biomechanics, phases, technique rules, or rendering. Marks limitations via
warning codes; does not reject aggressively.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from app.schemas.pose import Keypoint, PoseFrame, PoseSequence
from app.schemas.video_quality import VideoQualityMetrics, VideoQualityReport

# Expected COCO-17 body joints (names only; no CV/MMPose dependency).
_COCO_JOINTS: tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

# Soft limitation thresholds (warnings only).
_MIN_FPS = 24.0
_MIN_WIDTH = 640
_MIN_HEIGHT = 360
_MIN_MEAN_CONFIDENCE = 0.55
_MIN_FULL_BODY_COVERAGE = 0.60
_MIN_RACKET_ARM_VISIBILITY = 0.70
_MAX_MISSING_FRACTION = 0.35
_MAX_INTERPOLATED_FRACTION = 0.25
_MIN_PLAYER_SIZE_RATIO = 0.25
_MIN_CAMERA_STABILITY = 0.45

_FULL_BODY_JOINTS: tuple[str, ...] = (
    "left_shoulder",
    "right_shoulder",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)
_RACKET_ARM_JOINTS: tuple[str, ...] = (
    "right_shoulder",
    "right_elbow",
    "right_wrist",
)


def assess_video_quality(
    raw_pose: PoseSequence,
    *,
    smoothed_pose: PoseSequence | None = None,
    fps: float | None = None,
    width: int | None = None,
    height: int | None = None,
    confidence_threshold: float = 0.5,
) -> VideoQualityReport:
    """Build a ``VideoQualityReport`` from raw (and optional smoothed) pose.

    ``smoothed_pose`` is used only to estimate the interpolated-keypoint fraction.
    """
    video = raw_pose.video
    frames = raw_pose.frames
    resolved_fps = _resolve_fps(fps, frames)

    if not frames:
        return VideoQualityReport(
            video=video,
            usable=False,
            analysis_confidence=0.0,
            metrics=VideoQualityMetrics(
                fps=resolved_fps,
                width=width,
                height=height,
            ),
            warnings=["NO_POSE_DETECTED"],
        )

    joint_names = list(_COCO_JOINTS)
    mean_conf = _mean_pose_confidence(frames, confidence_threshold)
    missing_frac = _missing_keypoint_fraction(
        frames, joint_names, confidence_threshold
    )
    interp_frac = _interpolated_keypoint_fraction(
        frames,
        smoothed_pose.frames if smoothed_pose is not None else None,
        joint_names,
        confidence_threshold,
    )
    full_body = _coverage_fraction(frames, _FULL_BODY_JOINTS, confidence_threshold)
    racket_arm = _coverage_fraction(frames, _RACKET_ARM_JOINTS, confidence_threshold)
    player_size = _mean_player_size_ratio(frames, confidence_threshold)
    stability = _camera_stability(frames, confidence_threshold)

    any_keypoints = any(bool(frame.keypoints) for frame in frames)
    if not any_keypoints:
        return VideoQualityReport(
            video=video,
            usable=False,
            analysis_confidence=0.0,
            metrics=VideoQualityMetrics(
                fps=resolved_fps,
                width=width,
                height=height,
                full_body_coverage=full_body,
                racket_arm_visibility=racket_arm,
                mean_pose_confidence=mean_conf,
                missing_keypoint_fraction=missing_frac,
                interpolated_keypoint_fraction=interp_frac,
                player_size_ratio=player_size,
                camera_stability=stability,
            ),
            warnings=["NO_POSE_DETECTED"],
        )

    metrics = VideoQualityMetrics(
        fps=resolved_fps,
        width=width,
        height=height,
        full_body_coverage=full_body,
        racket_arm_visibility=racket_arm,
        mean_pose_confidence=mean_conf,
        missing_keypoint_fraction=missing_frac,
        interpolated_keypoint_fraction=interp_frac,
        player_size_ratio=player_size,
        camera_stability=stability,
    )
    warnings = _collect_warnings(metrics)
    confidence = _analysis_confidence(metrics, warnings)

    return VideoQualityReport(
        video=video,
        usable=True,
        analysis_confidence=confidence,
        metrics=metrics,
        warnings=warnings,
    )


def _resolve_fps(fps: float | None, frames: Sequence[PoseFrame]) -> float | None:
    if fps is not None and math.isfinite(fps) and fps > 0:
        return float(fps)
    if len(frames) < 2:
        return None
    dt = frames[-1].timestamp - frames[0].timestamp
    if dt <= 0 or not math.isfinite(dt):
        return None
    return float((len(frames) - 1) / dt)


def _visible(kp: Keypoint | None, threshold: float) -> bool:
    return kp is not None and math.isfinite(kp.confidence) and kp.confidence >= threshold


def _mean_pose_confidence(
    frames: Sequence[PoseFrame],
    threshold: float,
) -> float | None:
    values: list[float] = []
    for frame in frames:
        for kp in frame.keypoints.values():
            if kp.confidence >= threshold and math.isfinite(kp.confidence):
                values.append(float(kp.confidence))
    if not values:
        # Fall back to all finite confidences so low-confidence clips still report.
        for frame in frames:
            for kp in frame.keypoints.values():
                if math.isfinite(kp.confidence):
                    values.append(float(kp.confidence))
    if not values:
        return None
    return float(sum(values) / len(values))


def _missing_keypoint_fraction(
    frames: Sequence[PoseFrame],
    joint_names: Sequence[str],
    threshold: float,
) -> float:
    if not frames or not joint_names:
        return 1.0
    missing = 0
    total = len(frames) * len(joint_names)
    for frame in frames:
        for name in joint_names:
            if not _visible(frame.keypoints.get(name), threshold):
                missing += 1
    return float(missing / total)


def _interpolated_keypoint_fraction(
    raw_frames: Sequence[PoseFrame],
    smoothed_frames: Sequence[PoseFrame] | None,
    joint_names: Sequence[str],
    threshold: float,
) -> float | None:
    """Fraction of joint-slots present in smoothed but missing/low-conf in raw."""
    if smoothed_frames is None or not raw_frames or not joint_names:
        return None
    raw_by = {f.frame_index: f for f in raw_frames}
    total = 0
    interpolated = 0
    for sm in smoothed_frames:
        raw = raw_by.get(sm.frame_index)
        if raw is None:
            continue
        for name in joint_names:
            total += 1
            raw_ok = _visible(raw.keypoints.get(name), threshold)
            sm_ok = name in sm.keypoints and math.isfinite(sm.keypoints[name].x)
            if sm_ok and not raw_ok:
                interpolated += 1
    if total == 0:
        return None
    return float(interpolated / total)


def _coverage_fraction(
    frames: Sequence[PoseFrame],
    required: Sequence[str],
    threshold: float,
) -> float:
    if not frames or not required:
        return 0.0
    good = 0
    for frame in frames:
        if all(_visible(frame.keypoints.get(name), threshold) for name in required):
            good += 1
    return float(good / len(frames))


def _frame_bbox(
    frame: PoseFrame,
    threshold: float,
) -> tuple[float, float, float, float] | None:
    xs: list[float] = []
    ys: list[float] = []
    for kp in frame.keypoints.values():
        if not _visible(kp, threshold):
            continue
        if not (math.isfinite(kp.x) and math.isfinite(kp.y)):
            continue
        xs.append(kp.x)
        ys.append(kp.y)
    if len(xs) < 2:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def _mean_player_size_ratio(
    frames: Sequence[PoseFrame],
    threshold: float,
) -> float | None:
    """Mean normalized bbox height (keypoints already in [0, 1] image space)."""
    heights: list[float] = []
    for frame in frames:
        box = _frame_bbox(frame, threshold)
        if box is None:
            continue
        _x0, y0, _x1, y1 = box
        h = y1 - y0
        if h > 0 and math.isfinite(h):
            heights.append(float(h))
    if not heights:
        return None
    return float(sum(heights) / len(heights))


def _torso_center(
    frame: PoseFrame,
    threshold: float,
) -> tuple[float, float] | None:
    names = ("left_hip", "right_hip", "left_shoulder", "right_shoulder")
    xs: list[float] = []
    ys: list[float] = []
    for name in names:
        kp = frame.keypoints.get(name)
        if not _visible(kp, threshold):
            continue
        assert kp is not None
        xs.append(kp.x)
        ys.append(kp.y)
    if len(xs) < 2:
        return None
    return sum(xs) / len(xs), sum(ys) / len(ys)


def _camera_stability(
    frames: Sequence[PoseFrame],
    threshold: float,
) -> float | None:
    """Heuristic stability from torso-center jitter (1 = stable, 0 = shaky)."""
    centers: list[tuple[float, float]] = []
    for frame in frames:
        center = _torso_center(frame, threshold)
        if center is not None:
            centers.append(center)
    if len(centers) < 3:
        return None
    steps: list[float] = []
    for i in range(1, len(centers)):
        dx = centers[i][0] - centers[i - 1][0]
        dy = centers[i][1] - centers[i - 1][1]
        steps.append(math.hypot(dx, dy))
    mean_step = sum(steps) / len(steps)
    # ~0.02 normalized units/frame ≈ mild motion; larger → lower score.
    score = 1.0 / (1.0 + (mean_step / 0.02))
    return float(max(0.0, min(1.0, score)))


def _collect_warnings(metrics: VideoQualityMetrics) -> list[str]:
    warnings: list[str] = []
    if metrics.fps is not None and metrics.fps < _MIN_FPS:
        warnings.append("LOW_FPS")
    if metrics.width is not None and metrics.height is not None:
        if metrics.width < _MIN_WIDTH or metrics.height < _MIN_HEIGHT:
            warnings.append("LOW_RESOLUTION")
    if (
        metrics.mean_pose_confidence is not None
        and metrics.mean_pose_confidence < _MIN_MEAN_CONFIDENCE
    ):
        warnings.append("LOW_POSE_CONFIDENCE")
    if (
        metrics.full_body_coverage is not None
        and metrics.full_body_coverage < _MIN_FULL_BODY_COVERAGE
    ):
        warnings.append("LOW_FULL_BODY_COVERAGE")
    if (
        metrics.racket_arm_visibility is not None
        and metrics.racket_arm_visibility < _MIN_RACKET_ARM_VISIBILITY
    ):
        warnings.append("LOW_RACKET_ARM_VISIBILITY")
    if (
        metrics.missing_keypoint_fraction is not None
        and metrics.missing_keypoint_fraction > _MAX_MISSING_FRACTION
    ):
        warnings.append("HIGH_MISSING_KEYPOINTS")
    if (
        metrics.interpolated_keypoint_fraction is not None
        and metrics.interpolated_keypoint_fraction > _MAX_INTERPOLATED_FRACTION
    ):
        warnings.append("HIGH_INTERPOLATED_KEYPOINTS")
    if (
        metrics.player_size_ratio is not None
        and metrics.player_size_ratio < _MIN_PLAYER_SIZE_RATIO
    ):
        warnings.append("SMALL_PLAYER_SIZE")
    if (
        metrics.camera_stability is not None
        and metrics.camera_stability < _MIN_CAMERA_STABILITY
    ):
        warnings.append("UNSTABLE_CAMERA")
    return warnings


def _analysis_confidence(
    metrics: VideoQualityMetrics,
    warnings: Sequence[str],
) -> float:
    """Soft confidence in [0.05, 0.98]; warnings reduce score but do not zero it."""
    parts: list[float] = []
    if metrics.mean_pose_confidence is not None:
        parts.append(max(0.0, min(1.0, metrics.mean_pose_confidence)))
    if metrics.full_body_coverage is not None:
        parts.append(max(0.0, min(1.0, metrics.full_body_coverage)))
    if metrics.racket_arm_visibility is not None:
        parts.append(max(0.0, min(1.0, metrics.racket_arm_visibility)))
    if metrics.missing_keypoint_fraction is not None:
        parts.append(max(0.0, min(1.0, 1.0 - metrics.missing_keypoint_fraction)))
    if metrics.player_size_ratio is not None:
        # Ideal around 0.4–0.8 of frame height; squash extremes lightly.
        size = metrics.player_size_ratio
        parts.append(max(0.0, min(1.0, size / 0.45)))
    if metrics.camera_stability is not None:
        parts.append(max(0.0, min(1.0, metrics.camera_stability)))
    if metrics.fps is not None:
        parts.append(max(0.0, min(1.0, metrics.fps / 30.0)))

    base = sum(parts) / len(parts) if parts else 0.2
    penalty = 0.06 * len(warnings)
    score = base * (1.0 - min(0.55, penalty))
    return float(max(0.05, min(0.98, score)))
