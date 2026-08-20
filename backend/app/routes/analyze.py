from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.services.pose_service import pose_service
from app.services.video_service import new_output_path, new_upload_path

router = APIRouter(tags=["analyze"])

ALLOWED_EXTENSIONS = {".mp4", ".mov"}


@router.post("/analyze")
async def analyze(video: UploadFile = File(...)) -> dict[str, str]:
    filename = video.filename or "upload.mp4"
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: .mp4, .mov",
        )

    upload_path = new_upload_path(filename)
    output_path = new_output_path()

    try:
        contents = await video.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty upload")
        upload_path.write_bytes(contents)

        pose_service.analyze_video(upload_path, output_path)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 — surface CV/runtime errors to client
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if upload_path.exists():
            upload_path.unlink(missing_ok=True)

    if not output_path.exists():
        raise HTTPException(status_code=500, detail="Processing produced no output")

    relative = output_path.name
    return {
        "output_path": str(output_path),
        "video_url": f"/outputs/{relative}",
    }
