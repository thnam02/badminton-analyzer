"""AngleValidator: compare pipeline angles to manual / derived ground truth."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from app.processing.angles import compute_joint_angle
from app.schemas.angles import AngleFrame, AngleSequence
from app.schemas.phases import PhaseSequence
from app.schemas.pose import Keypoint, PoseFrame, PoseSequence
from app.validation.angle_annotations import (
    AnnotatedAngleFrame,
    AngleValidationAnnotationSet,
)
from app.validation.angle_joints import (
    PIPELINE_ANGLE_NAMES,
    TRIPLET_BY_ANGLE,
    VALIDATED_ANGLE_NAMES,
)
from app.validation.angle_metrics import (
    invalid_rate,
    mean_absolute_error,
    median_absolute_error,
    percentile_absolute_error,
)
from app.validation.angle_report import (
    AggregateAngleStats,
    AngleErrorDetail,
    AngleValidationReport,
    ConfidenceBinBreakdown,
    FrameAngleValidationResult,
    PhaseAngleBreakdown,
)
from app.validation.annotations import annotated_to_keypoint

# Default bins for confidence-stratified error (half-open [lo, hi)).
DEFAULT_CONFIDENCE_BINS: tuple[tuple[str, float, float], ...] = (
    ("low_<0.5", 0.0, 0.5),
    ("mid_0.5_0.75", 0.5, 0.75),
    ("high_>=0.75", 0.75, 1.0001),
)


class AngleValidator:
    """Offline comparison of predicted angles vs annotated / derived GT.

    Completely separate from ``PoseService`` / ``/analyze``.
    """

    def __init__(
        self,
        *,
        confidence_threshold: float = 0.5,
        confidence_bins: Sequence[tuple[str, float, float]] = DEFAULT_CONFIDENCE_BINS,
    ) -> None:
        self.confidence_threshold = float(confidence_threshold)
        self.confidence_bins = tuple(confidence_bins)

    def validate(
        self,
        predictions: AngleSequence,
        annotations: AngleValidationAnnotationSet,
        *,
        smoothed_pose: PoseSequence | None = None,
        phases: PhaseSequence | None = None,
    ) -> AngleValidationReport:
        """Compare pipeline angles (and optional pose-derived left angles) to GT.

        ``smoothed_pose`` is used only to:
        - derive predicted left-side angles with the production formula
        - estimate per-angle pose confidence for confidence-binned reports
        """
        pred_by = {f.frame_index: f for f in predictions.frames}
        pose_by = (
            {f.frame_index: f for f in smoothed_pose.frames}
            if smoothed_pose is not None
            else {}
        )
        frame_results: list[FrameAngleValidationResult] = []

        for ann in annotations.frames:
            gt_angles = resolve_ground_truth_angles(
                ann, confidence_threshold=self.confidence_threshold
            )
            pred_frame = pred_by.get(ann.frame_index)
            pose_frame = pose_by.get(ann.frame_index)
            phase = _resolve_phase(ann, phases)
            frame_results.append(
                self._validate_frame(
                    ann,
                    gt_angles=gt_angles,
                    pred_frame=pred_frame,
                    pose_frame=pose_frame,
                    phase=phase,
                )
            )

        overall = self._aggregate("overall", frame_results, per_angle=True)
        return AngleValidationReport(
            video=annotations.video or predictions.video,
            annotation_version=annotations.annotation_version,
            annotated_frame_count=len(annotations.frames),
            matched_frame_count=sum(
                1 for f in frame_results if f.annotated_angle_count > 0
            ),
            overall=overall,
            by_phase=self._by_phase(frame_results),
            by_confidence=self._by_confidence(frame_results),
            frames=frame_results,
        )

    def _validate_frame(
        self,
        ann: AnnotatedAngleFrame,
        *,
        gt_angles: dict[str, float],
        pred_frame: AngleFrame | None,
        pose_frame: PoseFrame | None,
        phase: str | None,
    ) -> FrameAngleValidationResult:
        details: list[AngleErrorDetail] = []
        abs_errors: list[float] = []
        annotated = 0
        valid = 0
        frame_conf = ann.pose_confidence

        for angle_name in VALIDATED_ANGLE_NAMES:
            if angle_name not in gt_angles:
                continue
            annotated += 1
            gt = float(gt_angles[angle_name])
            pred = _predicted_angle(
                angle_name,
                pred_frame=pred_frame,
                pose_frame=pose_frame,
                confidence_threshold=self.confidence_threshold,
            )
            pose_conf = _angle_pose_confidence(angle_name, pose_frame)
            if frame_conf is None and pose_conf is not None:
                # Prefer annotation-level confidence when set; else mean of joints.
                pass
            sample_conf = frame_conf if frame_conf is not None else pose_conf

            if pred is None:
                details.append(
                    AngleErrorDetail(
                        angle_name=angle_name,
                        gt_degrees=gt,
                        pred_degrees=None,
                        absolute_error=None,
                        predicted=False,
                        pose_confidence=sample_conf,
                    )
                )
                continue

            valid += 1
            err = abs(pred - gt)
            abs_errors.append(err)
            details.append(
                AngleErrorDetail(
                    angle_name=angle_name,
                    gt_degrees=gt,
                    pred_degrees=float(pred),
                    absolute_error=float(err),
                    predicted=True,
                    pose_confidence=sample_conf,
                )
            )

        return FrameAngleValidationResult(
            frame_index=ann.frame_index,
            phase=phase,
            pose_confidence=frame_conf
            if frame_conf is not None
            else _mean_detail_confidence(details),
            angle_errors=details,
            mean_absolute_error=mean_absolute_error(abs_errors),
            annotated_angle_count=annotated,
            valid_prediction_count=valid,
            invalid_rate=invalid_rate(annotated=annotated, valid_predictions=valid),
        )

    def _aggregate(
        self,
        name: str,
        frames: Sequence[FrameAngleValidationResult],
        *,
        per_angle: bool,
        angle_filter: set[str] | None = None,
    ) -> AggregateAngleStats:
        errors: list[float] = []
        ann = 0
        valid = 0
        by_angle_err: dict[str, list[float]] = defaultdict(list)
        by_angle_ann: dict[str, int] = defaultdict(int)
        by_angle_valid: dict[str, int] = defaultdict(int)

        for frame in frames:
            for detail in frame.angle_errors:
                if angle_filter is not None and detail.angle_name not in angle_filter:
                    continue
                by_angle_ann[detail.angle_name] += 1
                ann += 1
                if detail.predicted and detail.absolute_error is not None:
                    by_angle_valid[detail.angle_name] += 1
                    valid += 1
                    by_angle_err[detail.angle_name].append(detail.absolute_error)
                    errors.append(detail.absolute_error)

        children: list[AggregateAngleStats] = []
        if per_angle:
            for angle_name in VALIDATED_ANGLE_NAMES:
                if by_angle_ann.get(angle_name, 0) == 0:
                    continue
                errs = by_angle_err.get(angle_name, [])
                children.append(
                    AggregateAngleStats(
                        name=angle_name,
                        sample_count=by_angle_ann[angle_name],
                        mae=mean_absolute_error(errs),
                        median_absolute_error=median_absolute_error(errs),
                        p90_absolute_error=percentile_absolute_error(
                            errs, percentile=90.0
                        ),
                        invalid_rate=invalid_rate(
                            annotated=by_angle_ann[angle_name],
                            valid_predictions=by_angle_valid.get(angle_name, 0),
                        ),
                    )
                )

        return AggregateAngleStats(
            name=name,
            sample_count=ann,
            mae=mean_absolute_error(errors),
            median_absolute_error=median_absolute_error(errors),
            p90_absolute_error=percentile_absolute_error(errors, percentile=90.0),
            invalid_rate=invalid_rate(annotated=ann, valid_predictions=valid),
            per_angle=children,
        )

    def _by_phase(
        self, frames: Sequence[FrameAngleValidationResult]
    ) -> list[PhaseAngleBreakdown]:
        groups: dict[str, list[FrameAngleValidationResult]] = defaultdict(list)
        for frame in frames:
            groups[frame.phase or "UNKNOWN"].append(frame)

        out: list[PhaseAngleBreakdown] = []
        # Prefer smash phase order when present.
        preferred = (
            "PREPARATION",
            "BACKSWING",
            "ACCELERATION",
            "ESTIMATED_CONTACT",
            "FOLLOW_THROUGH",
            "UNKNOWN",
        )
        ordered = [p for p in preferred if p in groups]
        ordered.extend(sorted(p for p in groups if p not in preferred))

        for phase in ordered:
            group = groups[phase]
            errors: list[float] = []
            ann = 0
            valid = 0
            for frame in group:
                ann += frame.annotated_angle_count
                valid += frame.valid_prediction_count
                for detail in frame.angle_errors:
                    if detail.absolute_error is not None:
                        errors.append(detail.absolute_error)
            out.append(
                PhaseAngleBreakdown(
                    phase=phase,
                    frame_count=len(group),
                    sample_count=ann,
                    mae=mean_absolute_error(errors),
                    median_absolute_error=median_absolute_error(errors),
                    p90_absolute_error=percentile_absolute_error(
                        errors, percentile=90.0
                    ),
                    invalid_rate=invalid_rate(
                        annotated=ann, valid_predictions=valid
                    ),
                )
            )
        return out

    def _by_confidence(
        self, frames: Sequence[FrameAngleValidationResult]
    ) -> list[ConfidenceBinBreakdown]:
        # Collect (confidence, absolute_error | None for invalid) per sample.
        samples: list[tuple[float, float | None]] = []
        for frame in frames:
            for detail in frame.angle_errors:
                conf = detail.pose_confidence
                if conf is None:
                    continue
                samples.append(
                    (
                        float(conf),
                        float(detail.absolute_error)
                        if detail.absolute_error is not None
                        else None,
                    )
                )

        out: list[ConfidenceBinBreakdown] = []
        for label, lo, hi in self.confidence_bins:
            errs: list[float] = []
            ann = 0
            valid = 0
            for conf, err in samples:
                if not (lo <= conf < hi):
                    continue
                ann += 1
                if err is not None:
                    valid += 1
                    errs.append(err)
            out.append(
                ConfidenceBinBreakdown(
                    label=label,
                    confidence_min=lo,
                    confidence_max=hi,
                    sample_count=ann,
                    mae=mean_absolute_error(errs),
                    median_absolute_error=median_absolute_error(errs),
                    p90_absolute_error=percentile_absolute_error(
                        errs, percentile=90.0
                    ),
                    invalid_rate=invalid_rate(
                        annotated=ann, valid_predictions=valid
                    ),
                )
            )
        return out


def validate_angles(
    predictions: AngleSequence,
    annotations: AngleValidationAnnotationSet,
    *,
    smoothed_pose: PoseSequence | None = None,
    phases: PhaseSequence | None = None,
    confidence_threshold: float = 0.5,
) -> AngleValidationReport:
    return AngleValidator(confidence_threshold=confidence_threshold).validate(
        predictions,
        annotations,
        smoothed_pose=smoothed_pose,
        phases=phases,
    )


def export_angle_validation_report(
    report: AngleValidationReport,
    output_path: Path,
) -> Path:
    """Write versioned ``angle_validation.json`` (summary + per-frame records)."""
    return report.save_json(Path(output_path))


def resolve_ground_truth_angles(
    ann: AnnotatedAngleFrame,
    *,
    confidence_threshold: float = 0.0,
) -> dict[str, float]:
    """Prefer direct annotated angles; fill gaps from annotated keypoints."""
    out = dict(ann.angles)
    if not ann.keypoints:
        return out

    kps = {
        name: annotated_to_keypoint(kp, confidence=1.0)
        for name, kp in ann.keypoints.items()
    }
    for angle_name, (proximal, vertex, distal) in TRIPLET_BY_ANGLE.items():
        if angle_name in out:
            continue
        derived = compute_joint_angle(
            kps,
            proximal,
            vertex,
            distal,
            confidence_threshold=confidence_threshold,
        )
        if derived is not None:
            out[angle_name] = float(derived)
    return out


def _predicted_angle(
    angle_name: str,
    *,
    pred_frame: AngleFrame | None,
    pose_frame: PoseFrame | None,
    confidence_threshold: float,
) -> float | None:
    if angle_name in PIPELINE_ANGLE_NAMES and pred_frame is not None:
        value = getattr(pred_frame, angle_name, None)
        if value is not None:
            return float(value)
    # Left-side (or missing pipeline) angles: derive from smoothed pose offline.
    if pose_frame is None or angle_name not in TRIPLET_BY_ANGLE:
        return None
    proximal, vertex, distal = TRIPLET_BY_ANGLE[angle_name]
    return compute_joint_angle(
        pose_frame.keypoints,
        proximal,
        vertex,
        distal,
        confidence_threshold=confidence_threshold,
    )


def _angle_pose_confidence(
    angle_name: str,
    pose_frame: PoseFrame | None,
) -> float | None:
    if pose_frame is None or angle_name not in TRIPLET_BY_ANGLE:
        return None
    proximal, vertex, distal = TRIPLET_BY_ANGLE[angle_name]
    confs: list[float] = []
    for name in (proximal, vertex, distal):
        kp = pose_frame.keypoints.get(name)
        if kp is None:
            return None
        confs.append(float(kp.confidence))
    return float(sum(confs) / len(confs))


def _mean_detail_confidence(details: Sequence[AngleErrorDetail]) -> float | None:
    vals = [d.pose_confidence for d in details if d.pose_confidence is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _resolve_phase(
    ann: AnnotatedAngleFrame,
    phases: PhaseSequence | None,
) -> str | None:
    if ann.phase:
        return str(ann.phase)
    if phases is None:
        return None
    phase = phases.phase_at(ann.frame_index)
    return phase.value if phase is not None else None
