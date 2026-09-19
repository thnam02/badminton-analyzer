"""PoseValidator: compare RTMPose predictions to manual COCO-17 annotations."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from app.schemas.phases import PhaseSequence
from app.schemas.pose import PoseFrame, PoseSequence
from app.validation.annotations import (
    AnnotatedPoseFrame,
    PoseValidationAnnotationSet,
)
from app.validation.joints import BADMINTON_CRITICAL_JOINTS, COCO17_JOINTS
from app.validation.metrics import (
    mean_or_none,
    missing_rate,
    normalized_pixel_error,
    pck_score,
)
from app.validation.report import (
    AggregateGroupStats,
    AggregateJointStats,
    FrameValidationResult,
    JointErrorDetail,
    PhaseBreakdown,
    PoseValidationReport,
)

DEFAULT_PCK_THRESHOLDS: tuple[float, ...] = (0.05, 0.10, 0.20)


class PoseValidator:
    """Offline comparison of predicted PoseSequence vs manual annotations.

    Completely separate from ``PoseService`` / ``/analyze`` — call this from
    research scripts or tests only.
    """

    def __init__(
        self,
        *,
        pck_thresholds: Sequence[float] = DEFAULT_PCK_THRESHOLDS,
        confidence_threshold: float = 0.0,
    ) -> None:
        self.pck_thresholds = tuple(float(t) for t in pck_thresholds)
        self.confidence_threshold = float(confidence_threshold)

    def validate(
        self,
        predictions: PoseSequence,
        annotations: PoseValidationAnnotationSet,
        *,
        phases: PhaseSequence | None = None,
    ) -> PoseValidationReport:
        pred_by_index = {f.frame_index: f for f in predictions.frames}
        frame_results: list[FrameValidationResult] = []

        for ann_frame in annotations.frames:
            pred_frame = pred_by_index.get(ann_frame.frame_index)
            phase = _resolve_phase(ann_frame, phases)
            frame_results.append(
                self._validate_frame(ann_frame, pred_frame, phase=phase)
            )

        overall = self._aggregate_group(
            "overall",
            frame_results,
            joint_filter=set(COCO17_JOINTS),
        )
        badminton = self._aggregate_group(
            "badminton_critical",
            frame_results,
            joint_filter=set(BADMINTON_CRITICAL_JOINTS),
        )
        by_phase = self._aggregate_by_phase(frame_results)

        return PoseValidationReport(
            video=annotations.video or predictions.video,
            annotation_version=annotations.annotation_version,
            pck_thresholds=list(self.pck_thresholds),
            confidence_threshold=self.confidence_threshold,
            annotated_frame_count=len(annotations.frames),
            matched_frame_count=sum(
                1 for f in frame_results if f.annotated_joint_count > 0
            ),
            overall=overall,
            badminton_critical=badminton,
            by_phase=by_phase,
            frames=frame_results,
        )

    def _validate_frame(
        self,
        ann: AnnotatedPoseFrame,
        pred: PoseFrame | None,
        *,
        phase: str | None,
    ) -> FrameValidationResult:
        details: list[JointErrorDetail] = []
        errors: list[float] = []
        confidences: list[float] = []
        annotated = 0
        detected = 0

        for joint, gt in sorted(ann.keypoints.items()):
            annotated += 1
            pred_kp = None if pred is None else pred.keypoints.get(joint)
            if (
                pred_kp is None
                or pred_kp.confidence < self.confidence_threshold
            ):
                details.append(
                    JointErrorDetail(
                        joint=joint,
                        error=None,
                        predicted=False,
                        confidence=None if pred_kp is None else float(pred_kp.confidence),
                        gt_x=gt.x,
                        gt_y=gt.y,
                    )
                )
                continue

            detected += 1
            err = normalized_pixel_error(pred_kp.x, pred_kp.y, gt.x, gt.y)
            errors.append(err)
            confidences.append(float(pred_kp.confidence))
            details.append(
                JointErrorDetail(
                    joint=joint,
                    error=err,
                    predicted=True,
                    confidence=float(pred_kp.confidence),
                    gt_x=gt.x,
                    gt_y=gt.y,
                    pred_x=float(pred_kp.x),
                    pred_y=float(pred_kp.y),
                )
            )

        return FrameValidationResult(
            frame_index=ann.frame_index,
            phase=phase,
            joint_errors=details,
            mean_error=mean_or_none(errors),
            mean_confidence=mean_or_none(confidences),
            missing_detection_rate=missing_rate(
                annotated=annotated, detected=detected
            ),
            annotated_joint_count=annotated,
            detected_joint_count=detected,
        )

    def _aggregate_group(
        self,
        name: str,
        frames: Sequence[FrameValidationResult],
        *,
        joint_filter: set[str],
    ) -> AggregateGroupStats:
        per_joint_errors: dict[str, list[float]] = defaultdict(list)
        per_joint_conf: dict[str, list[float]] = defaultdict(list)
        per_joint_ann: dict[str, int] = defaultdict(int)
        per_joint_det: dict[str, int] = defaultdict(int)
        all_errors: list[float] = []
        all_conf: list[float] = []
        total_ann = 0
        total_det = 0

        for frame in frames:
            for detail in frame.joint_errors:
                if detail.joint not in joint_filter:
                    continue
                per_joint_ann[detail.joint] += 1
                total_ann += 1
                if detail.predicted and detail.error is not None:
                    per_joint_det[detail.joint] += 1
                    total_det += 1
                    per_joint_errors[detail.joint].append(detail.error)
                    all_errors.append(detail.error)
                    if detail.confidence is not None:
                        per_joint_conf[detail.joint].append(detail.confidence)
                        all_conf.append(detail.confidence)

        joint_stats: list[AggregateJointStats] = []
        for joint in sorted(joint_filter):
            if per_joint_ann.get(joint, 0) == 0:
                continue
            errs = per_joint_errors.get(joint, [])
            joint_stats.append(
                AggregateJointStats(
                    joint=joint,
                    sample_count=per_joint_ann[joint],
                    mean_error=mean_or_none(errs),
                    mean_confidence=mean_or_none(per_joint_conf.get(joint, [])),
                    missing_detection_rate=missing_rate(
                        annotated=per_joint_ann[joint],
                        detected=per_joint_det.get(joint, 0),
                    ),
                    pck={
                        _pck_key(t): pck_score(errs, threshold=t)
                        for t in self.pck_thresholds
                    },
                )
            )

        return AggregateGroupStats(
            name=name,
            sample_count=total_ann,
            mean_error=mean_or_none(all_errors),
            mean_confidence=mean_or_none(all_conf),
            missing_detection_rate=missing_rate(
                annotated=total_ann, detected=total_det
            ),
            pck={
                _pck_key(t): pck_score(all_errors, threshold=t)
                for t in self.pck_thresholds
            },
            per_joint=joint_stats,
        )

    def _aggregate_by_phase(
        self,
        frames: Sequence[FrameValidationResult],
    ) -> list[PhaseBreakdown]:
        by_phase: dict[str, list[FrameValidationResult]] = defaultdict(list)
        for frame in frames:
            label = frame.phase or "UNKNOWN"
            by_phase[label].append(frame)

        breakdowns: list[PhaseBreakdown] = []
        for phase in sorted(by_phase.keys()):
            group = by_phase[phase]
            errors: list[float] = []
            confs: list[float] = []
            ann = 0
            det = 0
            for frame in group:
                ann += frame.annotated_joint_count
                det += frame.detected_joint_count
                for detail in frame.joint_errors:
                    if detail.predicted and detail.error is not None:
                        errors.append(detail.error)
                        if detail.confidence is not None:
                            confs.append(detail.confidence)
            breakdowns.append(
                PhaseBreakdown(
                    phase=phase,
                    frame_count=len(group),
                    mean_error=mean_or_none(errors),
                    mean_confidence=mean_or_none(confs),
                    missing_detection_rate=missing_rate(
                        annotated=ann, detected=det
                    ),
                    pck={
                        _pck_key(t): pck_score(errors, threshold=t)
                        for t in self.pck_thresholds
                    },
                )
            )
        return breakdowns


def validate_pose(
    predictions: PoseSequence,
    annotations: PoseValidationAnnotationSet,
    *,
    phases: PhaseSequence | None = None,
    pck_thresholds: Sequence[float] = DEFAULT_PCK_THRESHOLDS,
    confidence_threshold: float = 0.0,
) -> PoseValidationReport:
    """Module-level convenience wrapper around ``PoseValidator``."""
    return PoseValidator(
        pck_thresholds=pck_thresholds,
        confidence_threshold=confidence_threshold,
    ).validate(predictions, annotations, phases=phases)


def export_pose_validation_report(
    report: PoseValidationReport,
    output_path: Path,
) -> Path:
    """Write versioned ``pose_validation.json`` (summary + per-frame errors)."""
    return report.save_json(Path(output_path))


def _resolve_phase(
    ann: AnnotatedPoseFrame,
    phases: PhaseSequence | None,
) -> str | None:
    if ann.phase:
        return str(ann.phase)
    if phases is None:
        return None
    phase = phases.phase_at(ann.frame_index)
    return phase.value if phase is not None else None


def _pck_key(threshold: float) -> str:
    # Stable JSON key, e.g. 0.05 -> "0.05"
    text = f"{threshold:.4f}".rstrip("0").rstrip(".")
    return text
