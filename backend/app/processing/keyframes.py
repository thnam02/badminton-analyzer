"""Select and save representative smash keyframes from existing analysis.

Uses PhaseSequence + PoseSequence indices only. Frame extraction reads the
source video sequentially so indices stay aligned with pose/phase data.
No new pose/CV inference and no OpenAI.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.schemas.final_analysis import FinalAnalysisState
from app.schemas.keyframes import (
    CONTACT_MINUS_2,
    CONTACT_PLUS_2,
    Keyframe,
    KeyframeSet,
)
from app.schemas.phases import PhaseSegment, PhaseSequence, SmashPhase
from app.schemas.pose import PoseSequence

_PHASE_ORDER: tuple[SmashPhase, ...] = (
    SmashPhase.PREPARATION,
    SmashPhase.BACKSWING,
    SmashPhase.ACCELERATION,
    SmashPhase.ESTIMATED_CONTACT,
    SmashPhase.FOLLOW_THROUGH,
)


@dataclass(frozen=True, slots=True)
class KeyframeSelection:
    """Chosen frame before disk I/O (unit-testable)."""

    phase: str
    frame_index: int
    timestamp: float
    confidence: float


def select_keyframe_indices(
    phases: PhaseSequence,
    pose: PoseSequence,
    *,
    include_contact_neighbors: bool = True,
) -> list[KeyframeSelection]:
    """Pick representative frame indices aligned to ``pose`` / ``phases``.

    Phase keyframes use the midpoint of each segment (contact uses the
    estimated-contact anchor). Optional ``CONTACT_MINUS_2`` / ``CONTACT_PLUS_2``
    are included when those indices exist in the pose sequence.
    """
    pose_by_index = {frame.frame_index: frame for frame in pose.frames}
    if not pose_by_index:
        return []

    selections: list[KeyframeSelection] = []

    for phase in _PHASE_ORDER:
        if phase is SmashPhase.ESTIMATED_CONTACT:
            idx = phases.estimated_contact_frame_index
            if idx is None or idx not in pose_by_index:
                continue
            conf = _contact_confidence(phases)
            ts = (
                phases.estimated_contact_timestamp
                if phases.estimated_contact_timestamp is not None
                else pose_by_index[idx].timestamp
            )
            selections.append(
                KeyframeSelection(
                    phase=phase.value,
                    frame_index=idx,
                    timestamp=float(ts),
                    confidence=conf,
                )
            )
            continue

        segment = _first_segment(phases, phase)
        if segment is None:
            continue
        idx = _representative_index(segment)
        if idx not in pose_by_index:
            # Snap to nearest pose frame inside the segment window.
            idx = _nearest_pose_index_in_window(
                pose_by_index,
                segment.start_frame_index,
                segment.end_frame_index,
                preferred=idx,
            )
        if idx is None:
            continue
        selections.append(
            KeyframeSelection(
                phase=phase.value,
                frame_index=idx,
                timestamp=pose_by_index[idx].timestamp,
                confidence=float(max(0.0, min(1.0, segment.confidence))),
            )
        )

    if include_contact_neighbors:
        contact = phases.estimated_contact_frame_index
        if contact is not None:
            conf = _contact_confidence(phases)
            for label, offset in (
                (CONTACT_MINUS_2, -2),
                (CONTACT_PLUS_2, 2),
            ):
                neighbor = contact + offset
                if neighbor not in pose_by_index:
                    continue
                selections.append(
                    KeyframeSelection(
                        phase=label,
                        frame_index=neighbor,
                        timestamp=pose_by_index[neighbor].timestamp,
                        confidence=conf,
                    )
                )

    selections.sort(key=lambda s: (s.frame_index, s.phase))
    return selections


def extract_keyframes_from_final(
    state: FinalAnalysisState,
    output_dir: Path,
    *,
    include_contact_neighbors: bool = True,
) -> KeyframeSet:
    """Extract keyframes using only final phases / pose from ``FinalAnalysisState``."""
    return extract_keyframes(
        state.input_path,
        state.phases,
        state.smoothed_pose,
        output_dir,
        include_contact_neighbors=include_contact_neighbors,
    )


def extract_keyframes(
    video_path: Path,
    phases: PhaseSequence,
    pose: PoseSequence,
    output_dir: Path,
    *,
    include_contact_neighbors: bool = True,
) -> KeyframeSet:
    """Select keyframes, save raw BGR frames as JPEGs, return ``KeyframeSet``.

    Prefer ``extract_keyframes_from_final`` after contact resolution.
    """
    selections = select_keyframe_indices(
        phases,
        pose,
        include_contact_neighbors=include_contact_neighbors,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if not selections:
        return KeyframeSet(video=phases.video or pose.video, output_dir=str(output_dir))

    needed = {s.frame_index for s in selections}
    frames_by_index = _read_frames_by_index(video_path, needed)

    keyframes: list[Keyframe] = []
    for selection in selections:
        image = frames_by_index.get(selection.frame_index)
        if image is None:
            continue
        filename = f"{selection.phase}_f{selection.frame_index:06d}.jpg"
        file_path = output_dir / filename
        ok = cv2.imwrite(str(file_path), image)
        if not ok:
            continue
        keyframes.append(
            Keyframe(
                phase=selection.phase,
                frame_index=selection.frame_index,
                timestamp=selection.timestamp,
                file_path=str(file_path),
                confidence=selection.confidence,
            )
        )

    return KeyframeSet(
        video=phases.video or pose.video,
        output_dir=str(output_dir),
        keyframes=keyframes,
    )


def _first_segment(phases: PhaseSequence, phase: SmashPhase) -> PhaseSegment | None:
    for segment in phases.segments:
        if segment.phase is phase:
            return segment
    return None


def _representative_index(segment: PhaseSegment) -> int:
    return (segment.start_frame_index + segment.end_frame_index) // 2


def _contact_confidence(phases: PhaseSequence) -> float:
    for segment in phases.segments:
        if segment.phase is SmashPhase.ESTIMATED_CONTACT:
            return float(max(0.0, min(1.0, segment.confidence)))
    return float(max(0.0, min(1.0, phases.confidence)))


def _nearest_pose_index_in_window(
    pose_by_index: dict[int, object],
    start: int,
    end: int,
    *,
    preferred: int,
) -> int | None:
    candidates = [idx for idx in pose_by_index if start <= idx <= end]
    if not candidates:
        return None
    return min(candidates, key=lambda idx: (abs(idx - preferred), idx))


def _read_frames_by_index(
    video_path: Path,
    frame_indices: set[int],
) -> dict[int, np.ndarray]:
    """Sequential decode — same 0-based indexing as pose collection."""
    if not frame_indices:
        return {}
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video for keyframes: {video_path}")

    wanted = set(frame_indices)
    max_needed = max(wanted)
    out: dict[int, np.ndarray] = {}
    frame_index = 0
    try:
        while frame_index <= max_needed and wanted:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index in wanted:
                out[frame_index] = frame.copy()
                wanted.discard(frame_index)
            frame_index += 1
    finally:
        capture.release()
    return out
