"""Assemble a versioned dataset export from existing analysis artifacts.

Does not alter pose, shuttle, racket, contact, or coaching computation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from app.schemas.annotation import CoachAnnotationSet
from app.schemas.dataset import (
    DATASET_EXPORT_VERSION,
    DatasetExport,
    blank_annotation_template,
    utc_now_iso,
)
from app.schemas.final_analysis import FinalAnalysisState
from app.schemas.keyframes import KeyframeSet
from app.schemas.provenance import (
    AnalysisSnapshot,
    apply_provenance,
    validate_artifact_provenance,
    validate_object_provenance,
)
from app.schemas.stroke_metrics import StrokeMetrics
from app.schemas.technique import TechniqueEvaluation
from app.services.video_service import (
    _artifact_base_stem,
    annotation_template_json_path_for,
    dataset_export_json_path_for,
)

logger = logging.getLogger(__name__)


class DatasetExporter:
    """Build label-ready exports from on-disk (or in-memory) analysis JSON."""

    def export_from_final(
        self,
        state: FinalAnalysisState,
        *,
        metrics: StrokeMetrics,
        technique: TechniqueEvaluation,
        keyframes: KeyframeSet,
        snapshot: AnalysisSnapshot,
        phases_json_path: Path | None = None,
        metrics_json_path: Path | None = None,
        contact_json_path: Path | None = None,
        technique_json_path: Path | None = None,
        keyframes_json_path: Path | None = None,
        quality_json_path: Path | None = None,
        evidence_json_path: Path | None = None,
        pose_json_path: Path | None = None,
        smoothed_pose_json_path: Path | None = None,
        shuttle_json_path: Path | None = None,
        racket_json_path: Path | None = None,
        stroke_type: str = "SMASH",
    ) -> tuple[Path, Path, DatasetExport]:
        """Export using contact / phases / quality from ``FinalAnalysisState``."""
        validate_object_provenance(metrics, snapshot)
        validate_object_provenance(technique, snapshot)
        validate_object_provenance(keyframes, snapshot)
        validate_artifact_provenance(state.phases.to_dict(), snapshot)
        validate_artifact_provenance(state.contact.to_dict(), snapshot)

        dataset_path, template_path, export = self.export_analysis(
            output_stem=state.output_path,
            video_metadata={
                "video": state.video,
                "fps": state.video_fps,
                "width": state.video_width,
                "height": state.video_height,
                "usable": state.video_quality.usable,
                "analysis_confidence": state.video_quality.analysis_confidence,
            },
            phases=state.phases.to_dict(),
            metrics=metrics.to_dict(),
            contact=state.contact.to_dict(),
            technique=technique.to_dict(),
            keyframes=keyframes.to_dict(),
            video_quality=state.video_quality.to_dict(),
            phases_json_path=phases_json_path,
            metrics_json_path=metrics_json_path,
            contact_json_path=contact_json_path,
            technique_json_path=technique_json_path,
            keyframes_json_path=keyframes_json_path,
            quality_json_path=quality_json_path,
            evidence_json_path=evidence_json_path,
            pose_json_path=pose_json_path,
            smoothed_pose_json_path=smoothed_pose_json_path,
            shuttle_json_path=shuttle_json_path,
            racket_json_path=racket_json_path,
            stroke_type=stroke_type,
            snapshot=snapshot,
        )
        return dataset_path, template_path, export


    def export_analysis(
        self,
        *,
        output_stem: Path,
        video_metadata: dict[str, Any] | None = None,
        phases: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        contact: dict[str, Any] | None = None,
        technique: dict[str, Any] | None = None,
        keyframes: dict[str, Any] | None = None,
        video_quality: dict[str, Any] | None = None,
        phases_json_path: Path | None = None,
        metrics_json_path: Path | None = None,
        contact_json_path: Path | None = None,
        technique_json_path: Path | None = None,
        keyframes_json_path: Path | None = None,
        quality_json_path: Path | None = None,
        evidence_json_path: Path | None = None,
        pose_json_path: Path | None = None,
        smoothed_pose_json_path: Path | None = None,
        shuttle_json_path: Path | None = None,
        racket_json_path: Path | None = None,
        stroke_type: str = "SMASH",
        snapshot: AnalysisSnapshot | None = None,
    ) -> tuple[Path, Path, DatasetExport]:
        """Write ``{id}_dataset.json`` + annotation template; return paths + object."""
        analysis_id = _artifact_base_stem(output_stem)
        if snapshot is not None and snapshot.analysis_id != analysis_id:
            raise ValueError(
                f"snapshot.analysis_id={snapshot.analysis_id!r} does not match "
                f"output stem analysis_id={analysis_id!r}."
            )
        phases_data = phases or _load_json(phases_json_path) or {}
        metrics_data = metrics or _load_json(metrics_json_path) or {}
        contact_data = contact or _load_json(contact_json_path) or {}
        technique_data = technique or _load_json(technique_json_path) or {}
        keyframes_data = keyframes or _load_json(keyframes_json_path) or {}
        quality_data = video_quality or _load_json(quality_json_path)

        meta = video_metadata or {}
        if not meta and isinstance(quality_data, dict):
            q_metrics = quality_data.get("metrics") or {}
            meta = {
                "video": quality_data.get("video")
                or metrics_data.get("video")
                or phases_data.get("video"),
                "fps": q_metrics.get("fps"),
                "width": q_metrics.get("width"),
                "height": q_metrics.get("height"),
                "usable": quality_data.get("usable"),
                "analysis_confidence": quality_data.get("analysis_confidence"),
            }
        if "video" not in meta or not meta.get("video"):
            meta = {
                **meta,
                "video": metrics_data.get("video")
                or phases_data.get("video")
                or output_stem.name,
            }

        issues = technique_data.get("issues")
        if issues is None:
            issues = technique_data.get("technique_issues") or []

        kf_list = keyframes_data.get("keyframes")
        if kf_list is None and isinstance(keyframes_data, list):
            kf_list = keyframes_data
        if kf_list is None:
            kf_list = []

        annotation_set = CoachAnnotationSet(analysis_id=analysis_id)
        template = blank_annotation_template(analysis_id)

        export = DatasetExport(
            dataset_export_version=DATASET_EXPORT_VERSION,
            analysis_id=analysis_id,
            created_at=utc_now_iso(),
            stroke_type=stroke_type,
            video_metadata=meta,
            pose_metrics=metrics_data,
            phases=phases_data,
            contact_event=contact_data,
            technique_issues=[dict(i) for i in issues],
            keyframes=[dict(k) for k in kf_list],
            video_quality=quality_data if isinstance(quality_data, dict) else None,
            artifact_refs={
                "pose_json": _name(pose_json_path),
                "smoothed_pose_json": _name(smoothed_pose_json_path),
                "phases_json": _name(phases_json_path),
                "stroke_metrics_json": _name(metrics_json_path),
                "contact_json": _name(contact_json_path),
                "technique_json": _name(technique_json_path),
                "keyframes_json": _name(keyframes_json_path),
                "video_quality_json": _name(quality_json_path),
                "evidence_json": _name(evidence_json_path),
                "shuttle_json": _name(shuttle_json_path),
                "racket_json": _name(racket_json_path),
                "overlay_video": output_stem.name if output_stem.suffix else None,
            },
            coach_annotations=annotation_set,
            annotation_template=template,
        )

        if snapshot is not None:
            apply_provenance(
                export,
                snapshot,
                artifact_schema_version=DATASET_EXPORT_VERSION,
            )
            validate_object_provenance(export, snapshot)

        dataset_path = dataset_export_json_path_for(output_stem)
        template_path = annotation_template_json_path_for(output_stem)
        export.save_json(dataset_path)
        template_path.write_text(
            json.dumps(template, indent=2),
            encoding="utf-8",
        )
        logger.info(
            "Dataset export written analysis_id=%s snapshot_id=%s path=%s",
            analysis_id,
            getattr(export, "snapshot_id", "") or None,
            dataset_path.name,
        )
        return dataset_path, template_path, export


def _load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not Path(path).is_file():
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _name(path: Path | None) -> str | None:
    return Path(path).name if path is not None else None


dataset_exporter = DatasetExporter()
