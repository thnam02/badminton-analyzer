"""Independent shuttlecock tracking service (does not touch pose / contact)."""

from __future__ import annotations

import logging
from pathlib import Path

from app.config import settings
from app.cv.shuttle.convert import detections_to_trajectory
from app.cv.shuttle.debug_render import render_shuttle_debug_video
from app.cv.shuttle.factory import get_shuttle_tracker
from app.processing.shuttle_smooth import interpolate_short_gaps
from app.schemas.shuttle import ShuttleTrajectory
from app.services.video_service import (
    probe_video_metadata,
    shuttle_debug_video_path_for,
    shuttle_json_path_for,
)

logger = logging.getLogger(__name__)


class ShuttleService:
    """Run a shuttle tracker adapter and export shuttle.json + debug video."""

    def track_video(
        self,
        video_path: Path,
        output_stem: Path,
        *,
        backend: str | None = None,
        interpolate: bool | None = None,
    ) -> tuple[Path, Path, ShuttleTrajectory]:
        """Track shuttle in ``video_path``; name artifacts from ``output_stem``.

        ``output_stem`` is typically the pose overlay path
        (``{uuid}_pose.mp4``) so artifacts become ``{uuid}_shuttle.json`` and
        ``{uuid}_shuttle_debug.mp4``.
        """
        video_path = Path(video_path)
        if not video_path.is_file():
            raise FileNotFoundError(f"Video not found: {video_path}")

        tracker = get_shuttle_tracker(backend)
        if not tracker.is_available():
            raise RuntimeError(
                f"Shuttle backend '{tracker.name}' is not available. "
                "Configure SHUTTLE_TRACKNET_ROOT / SHUTTLE_TRACKNET_WEIGHTS "
                "for TrackNetV3, or set SHUTTLE_BACKEND=heuristic for smoke tests."
            )

        fps, width, height = probe_video_metadata(video_path)
        logger.info(
            "Shuttle tracking (%s) on %s (%dx%d @ %.2f fps)",
            tracker.name,
            video_path.name,
            width,
            height,
            fps,
        )
        raw = tracker.track(video_path)
        trajectory = detections_to_trajectory(
            raw,
            video_path=video_path,
            backend=tracker.name,
            fps=fps,
            width=width,
            height=height,
        )

        do_interp = (
            settings.shuttle_interp_max_gap > 0
            if interpolate is None
            else interpolate
        )
        if do_interp:
            trajectory = interpolate_short_gaps(
                trajectory,
                max_gap=settings.shuttle_interp_max_gap,
            )

        json_path = shuttle_json_path_for(output_stem)
        debug_path = shuttle_debug_video_path_for(output_stem)
        trajectory.save_json(json_path)
        render_shuttle_debug_video(
            video_path,
            trajectory,
            debug_path,
            trail_length=settings.shuttle_debug_trail_length,
        )
        return json_path, debug_path, trajectory


shuttle_service = ShuttleService()
