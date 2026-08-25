from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from app.config import settings
from app.services.pose_service import pose_service
from app.services.video_service import new_output_path, new_upload_path

router = APIRouter(tags=["analyze"])

ALLOWED_EXTENSIONS = {".mp4", ".mov"}


@router.post("/analyze")
async def analyze(
    video: UploadFile = File(...),
    muscle_overlay: bool | None = Query(default=None),
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
    show_muscles = (
        settings.overlay_muscle_enabled
        if muscle_overlay is None
        else muscle_overlay
    )

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
            muscle_overlay=show_muscles,
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        detail = str(exc)
        if show_muscles and "DensePose" in detail:
            detail = (
                f"{detail} — Install DensePose with "
                "backend/scripts/bootstrap_densepose.sh or disable muscle overlay."
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

    return {
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
    }
