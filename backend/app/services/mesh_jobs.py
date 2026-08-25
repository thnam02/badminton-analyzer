"""Background WHAM mesh jobs so /analyze can return before CPU feature extraction finishes."""

from __future__ import annotations

import json
import logging
import threading
import traceback
from pathlib import Path
from typing import Any

from app.config import settings
from app.cv.mesh.pipeline import (
    probe_video_size,
    recover_mesh_sequence,
    render_mesh_debug_video,
)
from app.schemas.pose import PoseSequence
from app.services.video_service import mesh_json_path_for, mesh_video_path_for

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_jobs: dict[str, dict[str, Any]] = {}


def mesh_status_path_for(pose_output: Path) -> Path:
    stem = pose_output.name.replace("_pose.mp4", "")
    return settings.output_dir / f"{stem}_mesh.status.json"


def job_id_for(pose_output: Path) -> str:
    return pose_output.name.replace("_pose.mp4", "")


def write_status(pose_output: Path, payload: dict[str, Any]) -> Path:
    path = mesh_status_path_for(pose_output)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    job_id = job_id_for(pose_output)
    with _lock:
        _jobs[job_id] = payload
    return path


def read_status(job_id: str) -> dict[str, Any] | None:
    with _lock:
        cached = _jobs.get(job_id)
    if cached is not None:
        return cached
    path = settings.output_dir / f"{job_id}_mesh.status.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def start_mesh_job(
    *,
    video_path: Path,
    pose_output: Path,
    pose_sequence: PoseSequence,
) -> dict[str, Any]:
    """Kick off mesh recovery in a daemon thread; return pending status payload."""
    job_id = job_id_for(pose_output)
    mesh_video = mesh_video_path_for(pose_output)
    mesh_json = mesh_json_path_for(pose_output)
    status: dict[str, Any] = {
        "job_id": job_id,
        "status": "pending",
        "mesh_video_url": f"/outputs/{mesh_video.name}",
        "mesh_json_url": f"/outputs/{mesh_json.name}",
        "error": None,
    }
    write_status(pose_output, status)

    # Persist a copy of the source clip; /analyze deletes the upload afterward.
    source_copy = settings.output_dir / f"{job_id}_mesh_source{video_path.suffix.lower()}"
    if video_path.resolve() != source_copy.resolve():
        source_copy.write_bytes(video_path.read_bytes())

    thread = threading.Thread(
        target=_run_mesh_job,
        name=f"mesh-{job_id}",
        kwargs={
            "job_id": job_id,
            "pose_output": pose_output,
            "video_path": source_copy,
            "pose_sequence": pose_sequence,
            "mesh_video": mesh_video,
            "mesh_json": mesh_json,
        },
        daemon=True,
    )
    thread.start()
    return status


def _run_mesh_job(
    *,
    job_id: str,
    pose_output: Path,
    video_path: Path,
    pose_sequence: PoseSequence,
    mesh_video: Path,
    mesh_json: Path,
) -> None:
    write_status(
        pose_output,
        {
            "job_id": job_id,
            "status": "running",
            "mesh_video_url": f"/outputs/{mesh_video.name}",
            "mesh_json_url": f"/outputs/{mesh_json.name}",
            "error": None,
        },
    )
    try:
        logger.info("Mesh job %s started (%s)", job_id, video_path.name)
        width, height = probe_video_size(video_path)
        mesh_sequence = recover_mesh_sequence(
            video_path,
            pose_sequence=pose_sequence,
            image_width=width,
            image_height=height,
        )
        render_mesh_debug_video(
            video_path,
            mesh_video,
            mesh_sequence,
            pose_sequence=pose_sequence,
        )
        mesh_sequence.save_summary_json(mesh_json)
        write_status(
            pose_output,
            {
                "job_id": job_id,
                "status": "done",
                "mesh_video_url": f"/outputs/{mesh_video.name}",
                "mesh_json_url": f"/outputs/{mesh_json.name}",
                "error": None,
            },
        )
        logger.info("Mesh job %s done → %s", job_id, mesh_video.name)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Mesh job %s failed", job_id)
        write_status(
            pose_output,
            {
                "job_id": job_id,
                "status": "error",
                "mesh_video_url": f"/outputs/{mesh_video.name}",
                "mesh_json_url": f"/outputs/{mesh_json.name}",
                "error": str(exc),
                "traceback": traceback.format_exc()[-2000:],
            },
        )
    finally:
        try:
            if "_mesh_source" in video_path.name:
                video_path.unlink(missing_ok=True)
        except OSError:
            pass
