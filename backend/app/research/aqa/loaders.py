"""Dataset loaders for research AQA (consume exports + coach labels).

No training loops — only materialize ``AQASample`` instances.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterator

from app.research.aqa.schemas import (
    AQASample,
    DeterministicEvidenceBundle,
    PhaseNormalizedMotionFeatures,
    RGBKeyframeFeatures,
    ReferenceProfileContext,
)
from app.schemas.annotation import CoachAnnotation


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def discover_coach_annotation_paths(
    outputs_dir: Path,
    analysis_id: str,
) -> list[Path]:
    """Find ``{analysis_id}_annotation_{coach_id}.json`` beside dataset exports."""
    pattern = re.compile(
        rf"^{re.escape(analysis_id)}_annotation_(?!template$).+\.json$"
    )
    paths = [
        p
        for p in Path(outputs_dir).glob(f"{analysis_id}_annotation_*.json")
        if pattern.match(p.name) and "template" not in p.name
    ]
    return sorted(paths)


def load_coach_labels(paths: list[Path]) -> list[dict[str, Any]]:
    labels: list[dict[str, Any]] = []
    for path in paths:
        data = load_json(path)
        # Validate shape lightly via CoachAnnotation.from_dict.
        labels.append(CoachAnnotation.from_dict(data).to_dict())
    return labels


def build_phase_normalized_motion(
    *,
    phases: dict[str, Any],
    metrics: dict[str, Any],
) -> PhaseNormalizedMotionFeatures:
    """Lightweight placeholder features from phase windows + metrics.

    Not a production feature extractor — scaffolds tensors for future models.
    """
    feature_names = [
        "window_length_frames",
        "mean_wrist_speed_proxy",
        "contact_elbow_angle_deg",
        "peak_wrist_speed",
    ]
    phases_out: dict[str, list[float]] = {}
    segments = phases.get("segments") or []
    for seg in segments:
        name = str(seg.get("phase", "UNKNOWN"))
        start = seg.get("start_frame_index")
        end = seg.get("end_frame_index")
        length = 0.0
        if isinstance(start, int) and isinstance(end, int):
            length = float(max(0, end - start + 1))
        phases_out[name] = [
            length,
            float(metrics.get("peak_wrist_speed") or 0.0),
            float(metrics.get("contact_elbow_angle_deg") or 0.0),
            float(metrics.get("peak_wrist_speed") or 0.0),
        ]
    return PhaseNormalizedMotionFeatures(
        phases=phases_out,
        contact_frame_index=phases.get("estimated_contact_frame_index"),
        feature_names=feature_names,
    )


def build_rgb_keyframe_features(keyframes: list[dict[str, Any]]) -> RGBKeyframeFeatures:
    paths = []
    for kf in keyframes:
        fp = kf.get("file_path") or kf.get("path")
        if fp:
            paths.append(str(fp))
    return RGBKeyframeFeatures(keyframe_paths=paths, backend="path_refs_only")


def build_reference_profile_context(
    technique: dict[str, Any] | None = None,
    *,
    profile: dict[str, Any] | None = None,
) -> ReferenceProfileContext:
    if profile is not None:
        bands = {}
        for key, band in (profile.get("metrics") or {}).items():
            if isinstance(band, dict):
                bands[str(key)] = dict(band)
        return ReferenceProfileContext(
            profile_id=profile.get("profile_id") or profile.get("id"),
            stroke_type=profile.get("stroke_type"),
            handedness=profile.get("handedness"),
            camera_view=profile.get("camera_view"),
            metric_bands=bands,
            provisional=bool(profile.get("provisional", True)),
        )
    technique = technique or {}
    return ReferenceProfileContext(
        profile_id=technique.get("reference_profile_id"),
        stroke_type="SMASH",
        provisional=True,
    )


def sample_from_dataset_export(
    dataset_path: Path,
    *,
    outputs_dir: Path | None = None,
    pose_json_path: Path | None = None,
    include_pose_frames: bool = True,
    reference_profile: dict[str, Any] | None = None,
) -> AQASample:
    """Build an ``AQASample`` from ``{id}_dataset.json`` (+ optional pose/labels)."""
    dataset_path = Path(dataset_path)
    data = load_json(dataset_path)
    analysis_id = str(data.get("analysis_id") or dataset_path.stem.replace("_dataset", ""))
    out_dir = Path(outputs_dir) if outputs_dir is not None else dataset_path.parent

    refs = data.get("artifact_refs") or {}
    pose_frames: list[dict[str, Any]] = []
    if include_pose_frames:
        pose_path = pose_json_path
        if pose_path is None and refs.get("smoothed_pose_json"):
            pose_path = out_dir / str(refs["smoothed_pose_json"])
        if pose_path is None and refs.get("pose_json"):
            pose_path = out_dir / str(refs["pose_json"])
        if pose_path is not None and Path(pose_path).is_file():
            pose_data = load_json(Path(pose_path))
            pose_frames = list(pose_data.get("frames") or [])

    metrics = data.get("pose_metrics") or {}
    phases = data.get("phases") or {}
    keyframes = data.get("keyframes") or []
    technique_issues = data.get("technique_issues") or []

    coach_paths = discover_coach_annotation_paths(out_dir, analysis_id)
    # Prefer embedded annotations if somehow populated; else load side files.
    embedded = (data.get("coach_annotations") or {}).get("annotations") or []
    coach_labels = list(embedded) if embedded else load_coach_labels(coach_paths)

    evidence = DeterministicEvidenceBundle(
        pose_metrics=dict(metrics),
        phases=dict(phases),
        contact_event=dict(data.get("contact_event") or {}),
        technique_issues=[dict(i) for i in technique_issues],
        video_quality=data.get("video_quality"),
    )

    return AQASample(
        analysis_id=analysis_id,
        stroke_type=str(data.get("stroke_type") or "SMASH"),
        pose_frames=pose_frames,
        phase_normalized_motion=build_phase_normalized_motion(
            phases=phases, metrics=metrics
        ),
        rgb_keyframe_features=build_rgb_keyframe_features(keyframes),
        reference_profile=build_reference_profile_context(
            {"reference_profile_id": None, **({"issues": technique_issues})},
            profile=reference_profile,
        ),
        coach_labels=coach_labels,
        deterministic_evidence=evidence,
        artifact_refs={k: (str(v) if v is not None else None) for k, v in refs.items()},
    )


def iter_aqa_samples(
    outputs_dir: Path,
    *,
    include_pose_frames: bool = False,
) -> Iterator[AQASample]:
    """Yield samples for every ``*_dataset.json`` under an outputs directory."""
    outputs_dir = Path(outputs_dir)
    for dataset_path in sorted(outputs_dir.glob("*_dataset.json")):
        yield sample_from_dataset_export(
            dataset_path,
            outputs_dir=outputs_dir,
            include_pose_frames=include_pose_frames,
        )


class AQADataset:
    """Thin dataset wrapper for research iteration (no torch Dataset required)."""

    def __init__(
        self,
        outputs_dir: Path,
        *,
        include_pose_frames: bool = False,
    ) -> None:
        self.outputs_dir = Path(outputs_dir)
        self.include_pose_frames = include_pose_frames
        self._paths = sorted(self.outputs_dir.glob("*_dataset.json"))

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, index: int) -> AQASample:
        return sample_from_dataset_export(
            self._paths[index],
            outputs_dir=self.outputs_dir,
            include_pose_frames=self.include_pose_frames,
        )

    def __iter__(self) -> Iterator[AQASample]:
        return iter_aqa_samples(
            self.outputs_dir,
            include_pose_frames=self.include_pose_frames,
        )
