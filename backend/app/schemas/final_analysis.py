"""Canonical post-contact analysis state for all downstream consumers.

``FinalAnalysisState`` is created once after ContactResolver + phase re-snap.
It is frozen: contact and phase boundaries must not be replaced or treated as
mutable by StrokeMetrics, technique, keyframes, evidence, coaching, overlay,
or dataset export. Pre-resolution kinematics live only on
``IntermediateAnalysisSnapshot`` for debugging.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.schemas.angles import AngleSequence
from app.schemas.contact import ContactEvent
from app.schemas.motion import MotionSequence
from app.schemas.phases import PhaseSequence, SmashPhase
from app.schemas.pose import PoseSequence
from app.schemas.video_quality import VideoQualityReport


class FinalAnalysisStateError(ValueError):
    """Raised when a FinalAnalysisState would violate analysis invariants."""


@dataclass(frozen=True, slots=True)
class IntermediateAnalysisSnapshot:
    """Debug-only pre-resolution kinematics (not for coaching / export / overlay)."""

    raw_pose: PoseSequence
    initial_phases: PhaseSequence
    kinematic_contact: ContactEvent


@dataclass(frozen=True, slots=True)
class FinalAnalysisState:
    """Single canonical analysis object after contact resolution.

    Downstream stages must read pose / angles / motion / quality / phases /
    contact (and path metadata) from this object — or from values derived
    directly from it in the same finalize pass — never from separate stale
    copies of contact or phases.
    """

    smoothed_pose: PoseSequence
    angles: AngleSequence
    motion: MotionSequence
    video_quality: VideoQualityReport
    phases: PhaseSequence
    contact: ContactEvent
    # Immutable stroke / clip metadata required by downstream stages.
    video: str
    video_fps: float
    video_width: int
    video_height: int
    input_path: Path
    output_path: Path
    intermediate: IntermediateAnalysisSnapshot | None = None

    @property
    def frame_indices(self) -> tuple[int, ...]:
        return tuple(sorted(f.frame_index for f in self.smoothed_pose.frames))

    @property
    def contact_frame_index(self) -> int:
        return int(self.contact.frame_index)

    @property
    def phase_contact_frame_index(self) -> int | None:
        return self.phases.estimated_contact_frame_index

    def to_debug_dict(self) -> dict[str, Any]:
        """Compact summary for logging (not a persistence schema)."""
        return {
            "video": self.video,
            "frame_count": len(self.frame_indices),
            "contact_type": self.contact.contact_type,
            "contact_frame_index": self.contact.frame_index,
            "phase_contact_frame_index": self.phases.estimated_contact_frame_index,
            "phase_segment_count": len(self.phases.segments),
            "has_intermediate": self.intermediate is not None,
        }


def build_final_analysis_state(
    *,
    smoothed_pose: PoseSequence,
    angles: AngleSequence,
    motion: MotionSequence,
    video_quality: VideoQualityReport,
    phases: PhaseSequence,
    contact: ContactEvent,
    video_fps: float,
    video_width: int,
    video_height: int,
    input_path: Path,
    output_path: Path,
    intermediate: IntermediateAnalysisSnapshot | None = None,
    require_contact_in_timeline: bool = True,
) -> FinalAnalysisState:
    """Copy inputs into a frozen FinalAnalysisState after invariant checks."""
    pose_copy = _copy_pose_sequence(smoothed_pose)
    angles_copy = _copy_angle_sequence(angles)
    motion_copy = _copy_motion_sequence(motion)
    phases_copy = _copy_phase_sequence(phases)
    contact_copy = _copy_contact_event(contact)
    video = (
        pose_copy.video
        or angles_copy.video
        or motion_copy.video
        or phases_copy.video
        or output_path.name
    )
    state = FinalAnalysisState(
        smoothed_pose=pose_copy,
        angles=angles_copy,
        motion=motion_copy,
        video_quality=video_quality,
        phases=phases_copy,
        contact=contact_copy,
        video=video,
        video_fps=float(video_fps),
        video_width=int(video_width),
        video_height=int(video_height),
        input_path=Path(input_path),
        output_path=Path(output_path),
        intermediate=intermediate,
    )
    validate_final_analysis_state(
        state,
        require_contact_in_timeline=require_contact_in_timeline,
    )
    return state


def validate_final_analysis_state(
    state: FinalAnalysisState,
    *,
    require_contact_in_timeline: bool = True,
) -> None:
    """Raise ``FinalAnalysisStateError`` if canonical invariants fail."""
    frames = state.frame_indices
    if not frames:
        raise FinalAnalysisStateError(
            "FinalAnalysisState requires a non-empty smoothed pose sequence."
        )

    min_f, max_f = frames[0], frames[-1]
    contact_f = state.contact.frame_index
    if contact_f < min_f or contact_f > max_f:
        raise FinalAnalysisStateError(
            f"Contact frame {contact_f} is outside analyzed frame range "
            f"[{min_f}, {max_f}]."
        )

    _validate_phase_ordering(state.phases)

    phase_contact = state.phases.estimated_contact_frame_index
    if require_contact_in_timeline and not _is_placeholder_contact(state.contact):
        if phase_contact is None:
            raise FinalAnalysisStateError(
                "Final phases are missing estimated_contact_frame_index."
            )
        if int(phase_contact) != int(contact_f):
            raise FinalAnalysisStateError(
                f"Contact frame {contact_f} disagrees with final phase "
                f"estimated_contact_frame_index={phase_contact}."
            )
        phase_at = state.phases.phase_at(contact_f)
        if phase_at is not SmashPhase.ESTIMATED_CONTACT:
            raise FinalAnalysisStateError(
                f"Contact frame {contact_f} is not labeled ESTIMATED_CONTACT "
                f"in the final phase timeline (got {phase_at!r})."
            )
        if phase_contact < min_f or phase_contact > max_f:
            raise FinalAnalysisStateError(
                f"Phase contact frame {phase_contact} is outside analyzed "
                f"frame range [{min_f}, {max_f}]."
            )

    # Pose / angles / motion frame index sets should cover the contact frame.
    for name, seq in (
        ("angles", state.angles),
        ("motion", state.motion),
    ):
        indices = {f.frame_index for f in seq.frames}
        if indices and contact_f not in indices and not _is_placeholder_contact(
            state.contact
        ):
            raise FinalAnalysisStateError(
                f"Contact frame {contact_f} is missing from {name} sequence."
            )


def _is_placeholder_contact(contact: ContactEvent) -> bool:
    """True for the empty kinematic placeholder when no smash contact exists."""
    return (
        contact.kinematic_frame_index is None
        and contact.confidence <= 0.0
        and "No kinematic contact" in (contact.notes or "")
    )


def _validate_phase_ordering(phases: PhaseSequence) -> None:
    if not phases.segments:
        return
    prev_end: int | None = None
    for i, seg in enumerate(phases.segments):
        if seg.start_frame_index > seg.end_frame_index:
            raise FinalAnalysisStateError(
                f"Phase segment {i} ({seg.phase.value}) has start "
                f"{seg.start_frame_index} > end {seg.end_frame_index}."
            )
        # Allow abutting / shared boundaries; forbid going backwards (overlap).
        if prev_end is not None and seg.start_frame_index < prev_end:
            raise FinalAnalysisStateError(
                f"Phase segments are not monotonic: segment {i} "
                f"({seg.phase.value}) starts at {seg.start_frame_index} before "
                f"previous end {prev_end}."
            )
        prev_end = seg.end_frame_index

    # frame_phases is a dict (unique keys); sorted indices must be ascending.
    if phases.frame_phases:
        ordered = sorted(phases.frame_phases.keys())
        for a, b in zip(ordered, ordered[1:]):
            if b < a:
                raise FinalAnalysisStateError(
                    "Final phase frame_phases indices are not monotonic."
                )


def _copy_pose_sequence(pose: PoseSequence) -> PoseSequence:
    return PoseSequence(video=pose.video, frames=list(pose.frames))


def _copy_angle_sequence(angles: AngleSequence) -> AngleSequence:
    return AngleSequence(video=angles.video, frames=list(angles.frames))


def _copy_motion_sequence(motion: MotionSequence) -> MotionSequence:
    return MotionSequence(
        video=motion.video,
        frames=list(motion.frames),
        peaks=dict(motion.peaks) if motion.peaks else {},
    )


def _copy_phase_sequence(phases: PhaseSequence) -> PhaseSequence:
    return PhaseSequence(
        video=phases.video,
        segments=list(phases.segments),
        frame_phases=dict(phases.frame_phases),
        estimated_contact_frame_index=phases.estimated_contact_frame_index,
        estimated_contact_timestamp=phases.estimated_contact_timestamp,
        confidence=phases.confidence,
        notes=phases.notes,
    )


def _copy_contact_event(contact: ContactEvent) -> ContactEvent:
    return ContactEvent(
        contact_type=contact.contact_type,
        frame_index=int(contact.frame_index),
        timestamp=float(contact.timestamp),
        confidence=float(contact.confidence),
        evidence=list(contact.evidence),
        kinematic_frame_index=contact.kinematic_frame_index,
        kinematic_timestamp=contact.kinematic_timestamp,
        notes=contact.notes,
    )
