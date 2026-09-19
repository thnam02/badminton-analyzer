"""PhaseContactValidator: compare predicted phases/contact to manual timing GT."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

from app.schemas.contact import (
    CONTACT_TYPE_KINEMATIC,
    CONTACT_TYPE_TRACKED,
    ContactEvent,
)
from app.schemas.phases import PhaseSegment, PhaseSequence, SmashPhase
from app.validation.phase_contact_annotations import (
    PhaseContactAnnotationSet,
    VideoPhaseContactAnnotation,
)
from app.validation.phase_contact_boundaries import (
    BOUNDARY_CONTACT,
    BOUNDARY_NAMES,
    BOUNDARY_TO_SEGMENT,
    FRAME_TOLERANCES,
    TIME_TOLERANCES_MS,
)
from app.validation.phase_contact_metrics import (
    absolute_frame_error,
    absolute_time_error_ms,
    frame_error_to_ms,
    mean_or_none,
    median_or_none,
    missing_rate,
    percentile_or_none,
    within_tolerance_rate,
)
from app.validation.phase_contact_report import (
    BoundaryTimingError,
    GroupTimingBreakdown,
    PhaseContactValidationReport,
    TimingAggregateStats,
    VideoTimingResult,
)

# Analysis-confidence bins for stratified reporting [lo, hi).
DEFAULT_CONFIDENCE_BINS: tuple[tuple[str, float, float], ...] = (
    ("low_<0.5", 0.0, 0.5),
    ("mid_0.5_0.75", 0.5, 0.75),
    ("high_>=0.75", 0.75, 1.0001),
)


class PhaseContactValidator:
    """Offline timing comparison — not used by production ``/analyze``."""

    def __init__(
        self,
        *,
        frame_tolerances: Sequence[int] = FRAME_TOLERANCES,
        time_tolerances_ms: Sequence[int] = TIME_TOLERANCES_MS,
        confidence_bins: Sequence[tuple[str, float, float]] = DEFAULT_CONFIDENCE_BINS,
    ) -> None:
        self.frame_tolerances = tuple(int(t) for t in frame_tolerances)
        self.time_tolerances_ms = tuple(int(t) for t in time_tolerances_ms)
        self.confidence_bins = tuple(confidence_bins)

    def validate_video(
        self,
        annotation: VideoPhaseContactAnnotation,
        phases: PhaseSequence,
        contact: ContactEvent | None,
    ) -> VideoTimingResult:
        predicted = extract_predicted_boundaries(phases, contact)
        boundary_errors: list[BoundaryTimingError] = []
        contact_frame_err: int | None = None
        contact_ms_err: float | None = None
        contact_annotated = BOUNDARY_CONTACT in annotation.boundaries
        contact_predicted = predicted.get(BOUNDARY_CONTACT) is not None

        for name in BOUNDARY_NAMES:
            gt = annotation.boundaries.get(name)
            if gt is None:
                continue
            gt_ts = annotation.resolved_timestamp(name)
            pred = predicted.get(name)
            if pred is None:
                boundary_errors.append(
                    BoundaryTimingError(
                        boundary=name,
                        gt_frame_index=gt.frame_index,
                        gt_timestamp=gt_ts,
                        pred_frame_index=None,
                        pred_timestamp=None,
                        absolute_frame_error=None,
                        absolute_time_error_ms=None,
                        predicted=False,
                    )
                )
                continue

            pred_frame, pred_ts = pred
            frame_err = absolute_frame_error(pred_frame, gt.frame_index)
            if gt_ts is not None and pred_ts is not None:
                ms_err = absolute_time_error_ms(pred_ts, gt_ts)
            else:
                ms_err = frame_error_to_ms(frame_err, fps=annotation.fps)

            boundary_errors.append(
                BoundaryTimingError(
                    boundary=name,
                    gt_frame_index=gt.frame_index,
                    gt_timestamp=gt_ts,
                    pred_frame_index=int(pred_frame),
                    pred_timestamp=float(pred_ts) if pred_ts is not None else None,
                    absolute_frame_error=frame_err,
                    absolute_time_error_ms=ms_err,
                    predicted=True,
                )
            )
            if name == BOUNDARY_CONTACT:
                contact_frame_err = frame_err
                contact_ms_err = ms_err

        contact_type = None
        if contact is not None and not _is_placeholder_contact(contact):
            contact_type = contact.contact_type

        return VideoTimingResult(
            video=annotation.video,
            fps=float(annotation.fps),
            analysis_confidence=annotation.analysis_confidence,
            contact_type=contact_type,
            contact_annotated=contact_annotated,
            contact_predicted=contact_predicted
            and contact_type is not None,
            boundary_errors=boundary_errors,
            contact_frame_error=contact_frame_err,
            contact_time_error_ms=contact_ms_err,
        )

    def validate(
        self,
        annotations: PhaseContactAnnotationSet,
        predictions: Mapping[str, tuple[PhaseSequence, ContactEvent | None]],
    ) -> PhaseContactValidationReport:
        """Validate all annotated videos present in ``predictions`` keyed by video id."""
        results: list[VideoTimingResult] = []
        for ann in annotations.videos:
            key = ann.video
            if key not in predictions:
                # Still record missing predictions for contact failure rate.
                results.append(
                    VideoTimingResult(
                        video=key,
                        fps=float(ann.fps),
                        analysis_confidence=ann.analysis_confidence,
                        contact_type=None,
                        contact_annotated=BOUNDARY_CONTACT in ann.boundaries,
                        contact_predicted=False,
                        boundary_errors=[
                            BoundaryTimingError(
                                boundary=name,
                                gt_frame_index=b.frame_index,
                                gt_timestamp=ann.resolved_timestamp(name),
                                pred_frame_index=None,
                                pred_timestamp=None,
                                absolute_frame_error=None,
                                absolute_time_error_ms=None,
                                predicted=False,
                            )
                            for name, b in ann.boundaries.items()
                        ],
                    )
                )
                continue
            phases, contact = predictions[key]
            results.append(self.validate_video(ann, phases, contact))

        contact_ann = sum(1 for r in results if r.contact_annotated)
        contact_ok = sum(
            1
            for r in results
            if r.contact_annotated and r.contact_predicted and r.contact_frame_error is not None
        )

        return PhaseContactValidationReport(
            annotation_version=annotations.annotation_version,
            video_count=len(results),
            overall_contact=self._aggregate_contact(results, name="overall_contact"),
            overall_boundaries={
                name: self._aggregate_boundary(results, name)
                for name in BOUNDARY_NAMES
                if any(b.boundary == name for r in results for b in r.boundary_errors)
            },
            by_contact_type=self._by_contact_type(results),
            by_fps=self._by_fps(results),
            by_analysis_confidence=self._by_confidence(results),
            contact_missing_rate=missing_rate(
                annotated=contact_ann, predicted=contact_ok
            ),
            videos=results,
        )

    def _aggregate_contact(
        self,
        results: Sequence[VideoTimingResult],
        *,
        name: str,
        filter_fn=None,
    ) -> TimingAggregateStats:
        frame_errs: list[float] = []
        ms_errs: list[float] = []
        annotated = 0
        predicted = 0
        for result in results:
            if filter_fn is not None and not filter_fn(result):
                continue
            if not result.contact_annotated:
                continue
            annotated += 1
            if result.contact_frame_error is not None:
                predicted += 1
                frame_errs.append(float(result.contact_frame_error))
                if result.contact_time_error_ms is not None:
                    ms_errs.append(float(result.contact_time_error_ms))
        return self._stats_from_errors(
            name,
            frame_errs,
            ms_errs,
            annotated=annotated,
            predicted=predicted,
        )

    def _aggregate_boundary(
        self,
        results: Sequence[VideoTimingResult],
        boundary: str,
        *,
        filter_fn=None,
    ) -> TimingAggregateStats:
        frame_errs: list[float] = []
        ms_errs: list[float] = []
        annotated = 0
        predicted = 0
        for result in results:
            if filter_fn is not None and not filter_fn(result):
                continue
            for err in result.boundary_errors:
                if err.boundary != boundary:
                    continue
                annotated += 1
                if err.predicted and err.absolute_frame_error is not None:
                    predicted += 1
                    frame_errs.append(float(err.absolute_frame_error))
                    if err.absolute_time_error_ms is not None:
                        ms_errs.append(float(err.absolute_time_error_ms))
        return self._stats_from_errors(
            boundary,
            frame_errs,
            ms_errs,
            annotated=annotated,
            predicted=predicted,
        )

    def _stats_from_errors(
        self,
        name: str,
        frame_errs: Sequence[float],
        ms_errs: Sequence[float],
        *,
        annotated: int,
        predicted: int,
    ) -> TimingAggregateStats:
        return TimingAggregateStats(
            name=name,
            sample_count=annotated,
            mae_frames=mean_or_none(frame_errs),
            median_frames=median_or_none(frame_errs),
            p90_frames=percentile_or_none(frame_errs, percentile=90.0),
            mae_ms=mean_or_none(ms_errs),
            median_ms=median_or_none(ms_errs),
            p90_ms=percentile_or_none(ms_errs, percentile=90.0),
            within_frame_tol={
                f"±{t}": within_tolerance_rate(frame_errs, tolerance=float(t))
                for t in self.frame_tolerances
            },
            within_time_tol_ms={
                f"±{t}ms": within_tolerance_rate(ms_errs, tolerance=float(t))
                for t in self.time_tolerances_ms
            },
            missing_rate=missing_rate(annotated=annotated, predicted=predicted),
        )

    def _by_contact_type(
        self, results: Sequence[VideoTimingResult]
    ) -> list[GroupTimingBreakdown]:
        groups: list[GroupTimingBreakdown] = []
        for contact_type in (CONTACT_TYPE_KINEMATIC, CONTACT_TYPE_TRACKED):
            subset = [r for r in results if r.contact_type == contact_type]
            if not subset:
                continue
            groups.append(
                GroupTimingBreakdown(
                    key=contact_type,
                    sample_count=len(subset),
                    contact=self._aggregate_contact(
                        subset, name=f"contact_{contact_type}"
                    ),
                    boundaries={
                        name: self._aggregate_boundary(subset, name)
                        for name in BOUNDARY_NAMES
                        if any(
                            b.boundary == name
                            for r in subset
                            for b in r.boundary_errors
                        )
                    },
                )
            )
        return groups

    def _by_fps(
        self, results: Sequence[VideoTimingResult]
    ) -> list[GroupTimingBreakdown]:
        buckets: dict[str, list[VideoTimingResult]] = defaultdict(list)
        for result in results:
            # Bucket to nearest common FPS label.
            fps_key = _fps_bucket(result.fps)
            buckets[fps_key].append(result)
        out: list[GroupTimingBreakdown] = []
        for key in sorted(buckets.keys()):
            subset = buckets[key]
            out.append(
                GroupTimingBreakdown(
                    key=key,
                    sample_count=len(subset),
                    contact=self._aggregate_contact(subset, name=f"contact_{key}"),
                    boundaries={
                        name: self._aggregate_boundary(subset, name)
                        for name in BOUNDARY_NAMES
                        if any(
                            b.boundary == name
                            for r in subset
                            for b in r.boundary_errors
                        )
                    },
                )
            )
        return out

    def _by_confidence(
        self, results: Sequence[VideoTimingResult]
    ) -> list[GroupTimingBreakdown]:
        out: list[GroupTimingBreakdown] = []
        for label, lo, hi in self.confidence_bins:
            subset = [
                r
                for r in results
                if r.analysis_confidence is not None and lo <= r.analysis_confidence < hi
            ]
            out.append(
                GroupTimingBreakdown(
                    key=label,
                    sample_count=len(subset),
                    contact=self._aggregate_contact(subset, name=f"contact_{label}"),
                    boundaries={
                        name: self._aggregate_boundary(subset, name)
                        for name in BOUNDARY_NAMES
                        if any(
                            b.boundary == name
                            for r in subset
                            for b in r.boundary_errors
                        )
                    },
                )
            )
        return out


def validate_phase_contact(
    annotations: PhaseContactAnnotationSet,
    predictions: Mapping[str, tuple[PhaseSequence, ContactEvent | None]],
) -> PhaseContactValidationReport:
    return PhaseContactValidator().validate(annotations, predictions)


def export_phase_contact_validation_report(
    report: PhaseContactValidationReport,
    output_path: Path,
) -> Path:
    """Write versioned ``phase_contact_validation.json``."""
    return report.save_json(Path(output_path))


def extract_predicted_boundaries(
    phases: PhaseSequence,
    contact: ContactEvent | None,
) -> dict[str, tuple[int, float | None]]:
    """Map predicted PhaseSequence + ContactEvent to annotation boundary keys."""
    out: dict[str, tuple[int, float | None]] = {}
    by_phase = _segments_by_phase(phases.segments)

    for boundary, (phase, which) in BOUNDARY_TO_SEGMENT.items():
        seg = by_phase.get(phase)
        if seg is None:
            continue
        if which == "start":
            out[boundary] = (int(seg.start_frame_index), float(seg.start_timestamp))
        else:
            out[boundary] = (int(seg.end_frame_index), float(seg.end_timestamp))

    # Contact: prefer resolved ContactEvent when real; else phase estimate.
    if contact is not None and not _is_placeholder_contact(contact):
        out[BOUNDARY_CONTACT] = (int(contact.frame_index), float(contact.timestamp))
    elif phases.estimated_contact_frame_index is not None:
        out[BOUNDARY_CONTACT] = (
            int(phases.estimated_contact_frame_index),
            float(phases.estimated_contact_timestamp)
            if phases.estimated_contact_timestamp is not None
            else None,
        )
    return out


def _segments_by_phase(
    segments: Sequence[PhaseSegment],
) -> dict[SmashPhase, PhaseSegment]:
    # Last segment wins if duplicates (should not happen in normal detection).
    out: dict[SmashPhase, PhaseSegment] = {}
    for seg in segments:
        out[seg.phase] = seg
    return out


def _is_placeholder_contact(contact: ContactEvent) -> bool:
    return (
        contact.kinematic_frame_index is None
        and contact.confidence <= 0.0
        and "No kinematic contact" in (contact.notes or "")
    )


def _fps_bucket(fps: float) -> str:
    if fps <= 0:
        return "fps_unknown"
    # Round to nearest integer FPS for grouping (24/25/30/60 etc.).
    return f"fps_{int(round(fps))}"
