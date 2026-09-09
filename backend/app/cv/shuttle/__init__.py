"""Shuttlecock tracking package (independent of pose)."""

from app.cv.shuttle.convert import detections_to_trajectory
from app.cv.shuttle.factory import get_shuttle_tracker

__all__ = ["detections_to_trajectory", "get_shuttle_tracker"]
