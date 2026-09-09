"""Pose context helpers for racket ↔ hitting-hand association.

Consumes already-computed PoseSequence JSON/objects only — never runs MMPose.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.schemas.pose import Keypoint, PoseFrame, PoseSequence

_HAND_JOINTS = {
    "RIGHT": ("right_wrist", "right_elbow", "right_shoulder"),
    "LEFT": ("left_wrist", "left_elbow", "left_shoulder"),
}


def load_pose_sequence(path: Path) -> PoseSequence:
    """Load a pose JSON artifact (raw or smoothed) into PoseSequence."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    frames: list[PoseFrame] = []
    for raw in data.get("frames", []):
        kps: dict[str, Keypoint] = {}
        for name, kp in (raw.get("keypoints") or {}).items():
            kps[name] = Keypoint(
                x=float(kp["x"]),
                y=float(kp["y"]),
                confidence=float(kp.get("confidence", 0.0)),
            )
        frames.append(
            PoseFrame(
                frame_index=int(raw["frame_index"]),
                timestamp=float(raw["timestamp"]),
                keypoints=kps,
            )
        )
    return PoseSequence(video=str(data.get("video", path.name)), frames=frames)


def infer_hitting_hand(
    pose: PoseSequence,
    *,
    preferred: str | None = None,
    confidence_threshold: float = 0.3,
) -> str:
    """Choose LEFT/RIGHT from optional preference or mean wrist confidence.

    Pipeline motion metrics are right-biased; default to RIGHT when tied/unknown.
    """
    if preferred in ("LEFT", "RIGHT"):
        return preferred

    totals = {"LEFT": 0.0, "RIGHT": 0.0}
    counts = {"LEFT": 0, "RIGHT": 0}
    for frame in pose.frames:
        for hand, (wrist_name, _elbow, _shoulder) in _HAND_JOINTS.items():
            wrist = frame.keypoints.get(wrist_name)
            if wrist is None or wrist.confidence < confidence_threshold:
                continue
            totals[hand] += wrist.confidence
            counts[hand] += 1
    if counts["LEFT"] == 0 and counts["RIGHT"] == 0:
        return "RIGHT"
    mean_l = totals["LEFT"] / max(1, counts["LEFT"])
    mean_r = totals["RIGHT"] / max(1, counts["RIGHT"])
    # Prefer the side with more visible wrist samples; break ties toward RIGHT.
    if counts["LEFT"] > counts["RIGHT"] * 1.15 and mean_l >= mean_r:
        return "LEFT"
    if counts["RIGHT"] > counts["LEFT"] * 1.15 and mean_r >= mean_l:
        return "RIGHT"
    return "RIGHT" if mean_r >= mean_l else "LEFT"


def arm_keypoints(
    frame: PoseFrame,
    hand: str,
) -> tuple[Keypoint | None, Keypoint | None, Keypoint | None]:
    """Return (wrist, elbow, shoulder) for the given hand."""
    wrist_n, elbow_n, shoulder_n = _HAND_JOINTS[hand]
    return (
        frame.keypoints.get(wrist_n),
        frame.keypoints.get(elbow_n),
        frame.keypoints.get(shoulder_n),
    )


def pose_frame_map(pose: PoseSequence) -> dict[int, PoseFrame]:
    return {f.frame_index: f for f in pose.frames}
