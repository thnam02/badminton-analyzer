from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from app.config import settings
from app.services.mesh_jobs import read_status
from app.services.pose_service import pose_service
from app.services.video_service import new_output_path, new_upload_path

router = APIRouter(tags=["analyze"])

ALLOWED_EXTENSIONS = {".mp4", ".mov"}


@router.get("/mesh-status/{job_id}")
def mesh_status(job_id: str) -> dict:
    """Poll background WHAM mesh job started by /analyze?mesh_overlay=true."""
    status = read_status(job_id)
    if status is None:
        # Fallback: mesh already on disk from an older sync run.
        mesh_video = settings.output_dir / f"{job_id}_mesh.mp4"
        if mesh_video.is_file():
            return {
                "job_id": job_id,
                "status": "done",
                "mesh_video_url": f"/outputs/{mesh_video.name}",
                "mesh_json_url": f"/outputs/{job_id}_mesh.json",
                "error": None,
            }
        raise HTTPException(status_code=404, detail=f"Unknown mesh job '{job_id}'")
    return status


@router.post("/analyze")
async def analyze(
    video: UploadFile = File(...),
    muscle_overlay: bool | None = Query(default=None),
    mesh_overlay: bool | None = Query(default=None),
) -> dict[str, str]:
    filename = video.filename or "upload.mp4"
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: .mp4, .mov",
        )

    upload_path = new_upload_path(filename)
    output_path = new_output_path()
    video_path: Path | None = None
    raw_json_path: Path | None = None
    smoothed_json_path: Path | None = None
    angles_json_path: Path | None = None
    motion_json_path: Path | None = None
    phases_json_path: Path | None = None
    metrics_json_path: Path | None = None
    technique_json_path: Path | None = None
    mesh_video_path: Path | None = None
    mesh_json_path: Path | None = None
    mesh_status_payload: dict | None = None
    # DensePose muscle path disabled; query param ignored.
    show_muscles = False
    del muscle_overlay
    run_mesh = settings.mesh_enabled if mesh_overlay is None else mesh_overlay

    try:
        contents = await video.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty upload")
        upload_path.write_bytes(contents)

        (
            video_path,
            raw_json_path,
            smoothed_json_path,
            angles_json_path,
            motion_json_path,
            phases_json_path,
            metrics_json_path,
            technique_json_path,
            mesh_video_path,
            mesh_json_path,
            mesh_status_payload,
            _raw_sequence,
            _smoothed_sequence,
            _angle_sequence,
            _motion_sequence,
            _phase_sequence,
            _stroke_metrics,
            _technique_evaluation,
        ) = pose_service.analyze_video(
            upload_path,
            output_path,
            muscle_overlay=False,
            mesh_overlay=run_mesh,
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        detail = str(exc)
        if run_mesh and ("WHAM" in detail or "Mesh" in detail or "mesh" in detail):
            detail = (
                f"{detail} — WHAM needs checkpoints under vendor/WHAM/checkpoints "
                "and body models under vendor/WHAM/dataset/body_models. "
                "Soft deps: pip install -r requirements.txt. "
                "Analyze with mesh enabled reuses RTMPose tracks (no ViTPose)."
            )
        raise HTTPException(status_code=500, detail=detail) from exc
    finally:
        if upload_path.exists():
            upload_path.unlink(missing_ok=True)

    if video_path is None or not video_path.exists():
        raise HTTPException(status_code=500, detail="Processing produced no output")
    if raw_json_path is None or not raw_json_path.exists():
        raise HTTPException(status_code=500, detail="Processing produced no pose JSON")
    if smoothed_json_path is None or not smoothed_json_path.exists():
        raise HTTPException(
            status_code=500, detail="Processing produced no smoothed pose JSON"
        )
    if angles_json_path is None or not angles_json_path.exists():
        raise HTTPException(
            status_code=500, detail="Processing produced no angles JSON"
        )
    if motion_json_path is None or not motion_json_path.exists():
        raise HTTPException(
            status_code=500, detail="Processing produced no motion JSON"
        )
    if phases_json_path is None or not phases_json_path.exists():
        raise HTTPException(
            status_code=500, detail="Processing produced no phases JSON"
        )
    if metrics_json_path is None or not metrics_json_path.exists():
        raise HTTPException(
            status_code=500, detail="Processing produced no stroke metrics JSON"
        )
    if technique_json_path is None or not technique_json_path.exists():
        raise HTTPException(
            status_code=500, detail="Processing produced no technique JSON"
        )

    payload: dict[str, str] = {
        "output_path": str(video_path),
        "video_url": f"/outputs/{video_path.name}",
        "pose_json_path": str(raw_json_path),
        "pose_json_url": f"/outputs/{raw_json_path.name}",
        "smoothed_pose_json_path": str(smoothed_json_path),
        "smoothed_pose_json_url": f"/outputs/{smoothed_json_path.name}",
        "angles_json_path": str(angles_json_path),
        "angles_json_url": f"/outputs/{angles_json_path.name}",
        "motion_json_path": str(motion_json_path),
        "motion_json_url": f"/outputs/{motion_json_path.name}",
        "phases_json_path": str(phases_json_path),
        "phases_json_url": f"/outputs/{phases_json_path.name}",
        "stroke_metrics_json_path": str(metrics_json_path),
        "stroke_metrics_json_url": f"/outputs/{metrics_json_path.name}",
        "technique_json_path": str(technique_json_path),
        "technique_json_url": f"/outputs/{technique_json_path.name}",
        "muscle_overlay": str(show_muscles).lower(),
        "mesh_overlay": str(run_mesh).lower(),
    }
    if mesh_status_payload is not None:
        payload["mesh_status"] = str(mesh_status_payload.get("status", "pending"))
        payload["mesh_job_id"] = str(mesh_status_payload.get("job_id", ""))
        payload["mesh_status_url"] = f"/mesh-status/{mesh_status_payload.get('job_id', '')}"
        if mesh_status_payload.get("mesh_video_url"):
            payload["mesh_video_url"] = str(mesh_status_payload["mesh_video_url"])
        if mesh_status_payload.get("mesh_json_url"):
            payload["mesh_json_url"] = str(mesh_status_payload["mesh_json_url"])
    elif mesh_video_path is not None:
        payload["mesh_video_path"] = str(mesh_video_path)
        payload["mesh_video_url"] = f"/outputs/{mesh_video_path.name}"
        payload["mesh_status"] = "done"
    if mesh_json_path is not None and "mesh_json_url" not in payload:
        payload["mesh_json_path"] = str(mesh_json_path)
        payload["mesh_json_url"] = f"/outputs/{mesh_json_path.name}"
    return payload
