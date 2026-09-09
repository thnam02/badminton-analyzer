"""Independent racket detection / lightweight tracking service."""

from __future__ import annotations

import logging
from pathlib import Path

from app.config import settings
from app.cv.racket.convert import detections_to_trajectory
from app.cv.racket.debug_render import render_racket_debug_video
from app.cv.racket.factory import get_racket_detector
from app.cv.racket.pose_context import infer_hitting_hand, load_pose_sequence
from app.cv.racket.track import apply_lightweight_tracking
from app.schemas.pose import PoseSequence
from app.schemas.racket import RacketTrajectory
from app.services.video_service import (
    probe_video_metadata,
    racket_debug_video_path_for,
    racket_json_path_for,
)

logger = logging.getLogger(__name__)


class RacketService:
    """Detect/track racket; export racket.json + debug overlay video."""

    def track_video(
        self,
        video_path: Path,
        output_stem: Path,
        *,
        pose: PoseSequence | None = None,
        pose_json_path: Path | None = None,
        backend: str | None = None,
        hitting_hand: str | None = None,
        apply_tracking: bool | None = None,
    ) -> tuple[Path, Path, RacketTrajectory]:
        """Run racket detection on ``video_path``.

        Pose is optional context loaded from an existing smoothed pose JSON
        (or passed in-memory). This service never invokes MMPose or shuttle.
        """
        video_path = Path(video_path)
        if not video_path.is_file():
            raise FileNotFoundError(f"Video not found: {video_path}")

        detector = get_racket_detector(backend)
        if not detector.is_available():
            raise RuntimeError(
                f"Racket backend '{detector.name}' is not available. "
                "Use RACKET_BACKEND=pose_guided with smoothed pose JSON, "
                "or configure RACKET_YOLO_WEIGHTS for YOLO."
            )

        pose_seq = pose
        if pose_seq is None and pose_json_path is not None:
            pose_seq = load_pose_sequence(Path(pose_json_path))

        hand = hitting_hand or settings.racket_hitting_hand or None
        if hand not in ("LEFT", "RIGHT") and pose_seq is not None:
            hand = infer_hitting_hand(pose_seq, preferred=None)

        fps, width, height = probe_video_metadata(video_path)
        logger.info(
            "Racket detection (%s) on %s hand=%s (%dx%d @ %.2f fps)",
            detector.name,
            video_path.name,
            hand,
            width,
            height,
            fps,
        )
        raw = detector.detect(video_path, pose=pose_seq, hitting_hand=hand)
        trajectory = detections_to_trajectory(
            raw,
            video_path=video_path,
            backend=detector.name,
            fps=fps,
            width=width,
            height=height,
            hitting_hand=hand,
        )

        do_track = (
            settings.racket_track_enabled
            if apply_tracking is None
            else apply_tracking
        )
        if do_track:
            trajectory = apply_lightweight_tracking(
                trajectory,
                max_gap=settings.racket_track_max_gap,
                max_jump=settings.racket_track_max_jump,
            )

        json_path = racket_json_path_for(output_stem)
        debug_path = racket_debug_video_path_for(output_stem)
        trajectory.save_json(json_path)
        render_racket_debug_video(
            video_path,
            trajectory,
            debug_path,
            trail_length=settings.racket_debug_trail_length,
        )
        return json_path, debug_path, trajectory


racket_service = RacketService()
