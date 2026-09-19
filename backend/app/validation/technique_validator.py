"""TechniqueIssueValidator: compare TechniqueIssue[] to coach ground truth."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

from app.schemas.technique import TechniqueEvaluation, TechniqueIssue
from app.validation.technique_annotations import (
    StrokeTechniqueAnnotation,
    TechniqueValidationAnnotationSet,
)
from app.validation.technique_issues import (
    LABEL_PRESENT,
    LABEL_UNCERTAIN,
    SUPPORTED_TECHNIQUE_ISSUE_CODES,
)
from app.validation.technique_metrics import (
    ConfusionCounts,
    metrics_from_confusion,
)
from app.validation.technique_report import (
    GroupTechniqueBreakdown,
    IssueClassificationMetrics,
    IssuePredictionOutcome,
    StrokeTechniqueResult,
    TechniqueValidationReport,
)

# Analysis-confidence bins for stratified reporting [lo, hi).
DEFAULT_CONFIDENCE_BINS: tuple[tuple[str, float, float], ...] = (
    ("low_<0.5", 0.0, 0.5),
    ("mid_0.5_0.75", 0.5, 0.75),
    ("high_>=0.75", 0.75, 1.0001),
)


class TechniqueIssueValidator:
    """Offline coach-label comparison — not used by production ``/analyze``."""

    def __init__(
        self,
        *,
        issue_codes: Sequence[str] = SUPPORTED_TECHNIQUE_ISSUE_CODES,
        confidence_bins: Sequence[tuple[str, float, float]] = DEFAULT_CONFIDENCE_BINS,
        exclude_uncertain: bool = True,
    ) -> None:
        self.issue_codes = tuple(issue_codes)
        self.confidence_bins = tuple(confidence_bins)
        self.exclude_uncertain = bool(exclude_uncertain)

    def validate_stroke(
        self,
        annotation: StrokeTechniqueAnnotation,
        predicted_issues: Sequence[TechniqueIssue] | TechniqueEvaluation,
    ) -> StrokeTechniqueResult:
        predicted_codes = _predicted_codes(predicted_issues)
        outcomes: list[IssuePredictionOutcome] = []
        uncertain_count = 0

        for code in self.issue_codes:
            coach = annotation.issues.get(code)
            if coach is None:
                continue
            pred_present = code in predicted_codes
            if coach.is_uncertain:
                uncertain_count += 1
                # Uncertain coach labels are never folded into the 2x2 matrix;
                # they are reported only under ``uncertain_summary``.
                outcomes.append(
                    IssuePredictionOutcome(
                        issue_code=code,
                        gt_label=LABEL_UNCERTAIN,
                        predicted_present=pred_present,
                        included_in_metrics=False,
                        outcome=None,
                    )
                )
                continue

            gt_present = coach.is_present
            outcomes.append(
                IssuePredictionOutcome(
                    issue_code=code,
                    gt_label=coach.label,
                    predicted_present=pred_present,
                    included_in_metrics=True,
                    outcome=_outcome(gt_present=gt_present, pred_present=pred_present),
                )
            )

        return StrokeTechniqueResult(
            stroke_id=annotation.stroke_id,
            video=annotation.video or annotation.stroke_id,
            analysis_confidence=annotation.analysis_confidence,
            camera_quality=annotation.camera_quality,
            predicted_issue_codes=sorted(predicted_codes),
            outcomes=outcomes,
            uncertain_label_count=uncertain_count,
        )

    def validate(
        self,
        annotations: TechniqueValidationAnnotationSet,
        predictions: Mapping[
            str, Sequence[TechniqueIssue] | TechniqueEvaluation
        ],
    ) -> TechniqueValidationReport:
        """Validate strokes keyed by ``stroke_id`` (or video id fallback)."""
        results: list[StrokeTechniqueResult] = []
        for ann in annotations.strokes:
            key = ann.stroke_id
            pred = predictions.get(key)
            if pred is None and ann.video:
                pred = predictions.get(ann.video)
            if pred is None:
                results.append(
                    StrokeTechniqueResult(
                        stroke_id=ann.stroke_id,
                        video=ann.video or ann.stroke_id,
                        analysis_confidence=ann.analysis_confidence,
                        camera_quality=ann.camera_quality,
                        predicted_issue_codes=[],
                        outcomes=[
                            IssuePredictionOutcome(
                                issue_code=code,
                                gt_label=label.label,
                                predicted_present=False,
                                included_in_metrics=label.is_certain
                                or not self.exclude_uncertain,
                                outcome=(
                                    _outcome(
                                        gt_present=label.is_present,
                                        pred_present=False,
                                    )
                                    if label.is_certain
                                    else None
                                ),
                            )
                            for code, label in ann.issues.items()
                            if code in self.issue_codes
                        ],
                        uncertain_label_count=sum(
                            1
                            for label in ann.issues.values()
                            if label.is_uncertain
                        ),
                    )
                )
                continue
            results.append(self.validate_stroke(ann, pred))

        per_issue = {
            code: self._metrics_for_issue(results, code)
            for code in self.issue_codes
            if any(o.issue_code == code for r in results for o in r.outcomes)
        }
        return TechniqueValidationReport(
            annotation_version=annotations.annotation_version,
            stroke_count=len(results),
            per_issue=per_issue,
            micro_average=self._micro_average(results),
            by_analysis_confidence=self._by_confidence(results),
            by_camera_quality=self._by_camera_quality(results),
            uncertain_summary=self._uncertain_summary(results),
            strokes=results,
        )

    def _metrics_for_issue(
        self,
        results: Sequence[StrokeTechniqueResult],
        issue_code: str,
        *,
        filter_fn=None,
    ) -> IssueClassificationMetrics:
        counts = ConfusionCounts()
        uncertain = 0
        for result in results:
            if filter_fn is not None and not filter_fn(result):
                continue
            for outcome in result.outcomes:
                if outcome.issue_code != issue_code:
                    continue
                if outcome.gt_label == LABEL_UNCERTAIN:
                    uncertain += 1
                    if not outcome.included_in_metrics:
                        continue
                if not outcome.included_in_metrics or outcome.outcome is None:
                    continue
                counts.add(
                    gt_present=outcome.gt_label == LABEL_PRESENT,
                    pred_present=outcome.predicted_present,
                )
        return _metrics_obj(issue_code, counts, uncertain_excluded=uncertain)

    def _micro_average(
        self,
        results: Sequence[StrokeTechniqueResult],
        *,
        filter_fn=None,
        name: str = "micro_average",
    ) -> IssueClassificationMetrics:
        counts = ConfusionCounts()
        uncertain = 0
        for result in results:
            if filter_fn is not None and not filter_fn(result):
                continue
            for outcome in result.outcomes:
                if outcome.gt_label == LABEL_UNCERTAIN:
                    uncertain += 1
                    if not outcome.included_in_metrics:
                        continue
                if not outcome.included_in_metrics or outcome.outcome is None:
                    continue
                counts.add(
                    gt_present=outcome.gt_label == LABEL_PRESENT,
                    pred_present=outcome.predicted_present,
                )
        return _metrics_obj(name, counts, uncertain_excluded=uncertain)

    def _by_confidence(
        self, results: Sequence[StrokeTechniqueResult]
    ) -> list[GroupTechniqueBreakdown]:
        out: list[GroupTechniqueBreakdown] = []
        for label, lo, hi in self.confidence_bins:
            subset = [
                r
                for r in results
                if r.analysis_confidence is not None
                and lo <= r.analysis_confidence < hi
            ]
            out.append(self._group_breakdown(label, subset))
        return out

    def _by_camera_quality(
        self, results: Sequence[StrokeTechniqueResult]
    ) -> list[GroupTechniqueBreakdown]:
        buckets: dict[str, list[StrokeTechniqueResult]] = defaultdict(list)
        for result in results:
            if result.camera_quality is None:
                continue
            key = str(result.camera_quality).strip().lower() or "unknown"
            buckets[key].append(result)
        return [
            self._group_breakdown(key, buckets[key])
            for key in sorted(buckets.keys())
        ]

    def _group_breakdown(
        self,
        key: str,
        subset: Sequence[StrokeTechniqueResult],
    ) -> GroupTechniqueBreakdown:
        per_issue = {
            code: self._metrics_for_issue(subset, code)
            for code in self.issue_codes
            if any(o.issue_code == code for r in subset for o in r.outcomes)
        }
        return GroupTechniqueBreakdown(
            key=key,
            sample_count=len(subset),
            per_issue=per_issue,
            micro_average=self._micro_average(subset, name=f"micro_{key}")
            if subset
            else None,
        )

    def _uncertain_summary(
        self, results: Sequence[StrokeTechniqueResult]
    ) -> dict[str, object]:
        """Separate report for uncertain coach labels (excluded from primary metrics)."""
        by_issue: dict[str, dict[str, int]] = defaultdict(
            lambda: {"uncertain_count": 0, "predicted_present": 0, "predicted_absent": 0}
        )
        total = 0
        for result in results:
            for outcome in result.outcomes:
                if outcome.gt_label != LABEL_UNCERTAIN:
                    continue
                total += 1
                bucket = by_issue[outcome.issue_code]
                bucket["uncertain_count"] += 1
                if outcome.predicted_present:
                    bucket["predicted_present"] += 1
                else:
                    bucket["predicted_absent"] += 1
        return {
            "excluded_from_primary_metrics": self.exclude_uncertain,
            "total_uncertain_labels": total,
            "by_issue": {k: dict(v) for k, v in sorted(by_issue.items())},
        }


def validate_technique_issues(
    annotations: TechniqueValidationAnnotationSet,
    predictions: Mapping[str, Sequence[TechniqueIssue] | TechniqueEvaluation],
) -> TechniqueValidationReport:
    return TechniqueIssueValidator().validate(annotations, predictions)


def export_technique_validation_report(
    report: TechniqueValidationReport,
    output_path: Path,
) -> Path:
    """Write versioned ``technique_validation.json``."""
    return report.save_json(Path(output_path))


def _predicted_codes(
    predicted: Sequence[TechniqueIssue] | TechniqueEvaluation,
) -> set[str]:
    """Codes counted as positive predictions (excludes insufficient evidence)."""
    from app.schemas.technique_calibration import IssueStatus

    if isinstance(predicted, TechniqueEvaluation):
        issues = predicted.issues
    else:
        issues = predicted
    out: set[str] = set()
    for issue in issues:
        if not issue.code:
            continue
        status = getattr(issue, "status", "") or ""
        if status == IssueStatus.INSUFFICIENT_EVIDENCE.value:
            continue
        if status == IssueStatus.NO_ISSUE.value:
            continue
        out.add(str(issue.code))
    return out


def _outcome(*, gt_present: bool, pred_present: bool) -> str:
    if gt_present and pred_present:
        return "tp"
    if (not gt_present) and pred_present:
        return "fp"
    if gt_present and (not pred_present):
        return "fn"
    return "tn"


def _metrics_obj(
    issue_code: str,
    counts: ConfusionCounts,
    *,
    uncertain_excluded: int,
) -> IssueClassificationMetrics:
    raw = metrics_from_confusion(counts)
    return IssueClassificationMetrics(
        issue_code=issue_code,
        precision=raw["precision"],
        recall=raw["recall"],
        f1=raw["f1"],
        specificity=raw["specificity"],
        false_positive_rate=raw["false_positive_rate"],
        false_negative_rate=raw["false_negative_rate"],
        support=int(raw["support"]),
        sample_count=int(raw["sample_count"]),
        confusion_matrix=dict(raw["confusion_matrix"]),
        uncertain_excluded_count=int(uncertain_excluded),
    )
