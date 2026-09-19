"""File-backed analysis history index (Phase D — no auth/DB)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import settings

MANIFEST_FILENAME = "analyses_index.json"


@dataclass(slots=True)
class AnalysisIndexEntry:
    analysis_id: str
    created_at: str
    stroke_type: str = "forehand_smash"
    handedness: str | None = None
    camera_view: str | None = None
    analysis_confidence: float | None = None
    main_issue: str | None = None
    pose_video_url: str = ""
    snapshot_id: str = ""
    reference_profile_id: str = ""
    analysis_status: str = "COMPLETE"
    coaching_status: str = "UNKNOWN"
    mesh_status: str = "NONE"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def manifest_path(output_dir: Path | None = None) -> Path:
    root = Path(output_dir) if output_dir is not None else Path(settings.output_dir)
    return root / MANIFEST_FILENAME


def load_index(output_dir: Path | None = None) -> list[AnalysisIndexEntry]:
    path = manifest_path(output_dir)
    if not path.is_file():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    entries_raw = raw.get("analyses") if isinstance(raw, dict) else raw
    if not isinstance(entries_raw, list):
        return []
    out: list[AnalysisIndexEntry] = []
    for item in entries_raw:
        if not isinstance(item, dict) or not item.get("analysis_id"):
            continue
        out.append(
            AnalysisIndexEntry(
                analysis_id=str(item["analysis_id"]),
                created_at=str(item.get("created_at") or ""),
                stroke_type=str(item.get("stroke_type") or "forehand_smash"),
                handedness=item.get("handedness"),
                camera_view=item.get("camera_view"),
                analysis_confidence=(
                    float(item["analysis_confidence"])
                    if item.get("analysis_confidence") is not None
                    else None
                ),
                main_issue=item.get("main_issue"),
                pose_video_url=str(item.get("pose_video_url") or ""),
                snapshot_id=str(item.get("snapshot_id") or ""),
                reference_profile_id=str(item.get("reference_profile_id") or ""),
                analysis_status=str(item.get("analysis_status") or "COMPLETE"),
                coaching_status=str(item.get("coaching_status") or "UNKNOWN"),
                mesh_status=str(item.get("mesh_status") or "NONE"),
            )
        )
    return out


def save_index(
    entries: list[AnalysisIndexEntry], *, output_dir: Path | None = None
) -> Path:
    path = manifest_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0.0",
        "updated_at": _utc_now(),
        "analyses": [e.to_dict() for e in entries],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def upsert_index_entry(
    entry: AnalysisIndexEntry, *, output_dir: Path | None = None
) -> AnalysisIndexEntry:
    entries = load_index(output_dir)
    by_id = {e.analysis_id: e for e in entries}
    by_id[entry.analysis_id] = entry
    ordered = sorted(
        by_id.values(),
        key=lambda e: e.created_at or "",
        reverse=True,
    )
    save_index(ordered, output_dir=output_dir)
    return entry


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
