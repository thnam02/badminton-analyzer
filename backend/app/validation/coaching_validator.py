"""CoachingValidator: faithfulness + unsupported-claim checks for CoachingReports."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from app.schemas.coaching import CoachingReport
from app.schemas.evidence import EvidencePackage
from app.validation.coaching_annotations import (
    COACH_RATING_DIMENSIONS,
    CoachReviewAnnotation,
    CoachingValidationAnnotationSet,
)
from app.validation.coaching_faithfulness import (
    build_evidence_index,
    check_structured_faithfulness,
    check_text_faithfulness,
)
from app.validation.coaching_unsupported import detect_unsupported_claims
from app.validation.coaching_validation_report import (
    AnalysisCoachingValidation,
    CoachingValidationReport,
    ConfidenceBinCoachingStats,
)
from app.validation.technique_issues import SUPPORTED_TECHNIQUE_ISSUE_CODES

DEFAULT_CONFIDENCE_BINS: tuple[tuple[str, float, float], ...] = (
    ("low_<0.5", 0.0, 0.5),
    ("mid_0.5_0.75", 0.5, 0.75),
    ("high_>=0.75", 0.75, 1.0001),
)


class CoachingValidator:
    """Offline coaching-report validation — not used by production ``/analyze``."""

    def __init__(
        self,
        *,
        confidence_bins: Sequence[tuple[str, float, float]] = DEFAULT_CONFIDENCE_BINS,
        known_issue_codes: Sequence[str] = SUPPORTED_TECHNIQUE_ISSUE_CODES,
    ) -> None:
        self.confidence_bins = tuple(confidence_bins)
        self.known_issue_codes = set(known_issue_codes)

    def validate_analysis(
        self,
        *,
        analysis_id: str,
        report: CoachingReport,
        evidence: EvidencePackage,
        coach_review: CoachReviewAnnotation | None = None,
    ) -> AnalysisCoachingValidation:
        index = build_evidence_index(evidence)
        texts = _report_texts(report)

        findings = []
        findings.extend(
            check_text_faithfulness(
                texts,
                index,
                known_issue_codes=self.known_issue_codes | index.issue_codes,
            )
        )
        findings.extend(
            check_structured_faithfulness(
                prioritized_issue_codes=[
                    i.issue_code for i in report.prioritized_issues
                ],
                metric_hints=[
                    hint
                    for item in report.prioritized_issues
                    for hint in item.related_metric_hints
                ],
                drill_issue_codes=[
                    code
                    for drill in report.drills
                    for code in drill.targets_issue_codes
                ],
                index=index,
            )
        )

        unsupported = detect_unsupported_claims(texts)
        unsupported_dicts = [u.to_dict() for u in unsupported]
        finding_dicts = [f.to_dict() for f in findings]

        claim_count = len(findings)
        supported_count = sum(1 for f in findings if f.supported)
        mismatched = sum(1 for f in findings if not f.supported)
        hall_count = mismatched + len(unsupported)

        priority_codes = [i.issue_code for i in report.prioritized_issues]
        priority_rate = _priority_agreement_rate(
            priority_codes,
            evidence_codes=index.issue_codes,
            expected=(
                coach_review.expected_priority_issue_codes if coach_review else None
            ),
        )

        conf = evidence.analysis_confidence
        if coach_review is not None and coach_review.analysis_confidence is not None:
            conf = coach_review.analysis_confidence

        return AnalysisCoachingValidation(
            analysis_id=analysis_id,
            video=report.video or evidence.video,
            analysis_confidence=float(conf) if conf is not None else None,
            faithfulness_findings=finding_dicts,
            unsupported_claims=unsupported_dicts,
            faithfulness_claim_count=claim_count,
            faithfulness_supported_count=supported_count,
            faithfulness_rate=(
                float(supported_count) / float(claim_count) if claim_count else None
            ),
            hallucination_or_unsupported_count=hall_count,
            has_hallucination_or_unsupported=hall_count > 0,
            priority_issue_codes=priority_codes,
            priority_agreement_rate=priority_rate,
            coach_ratings=coach_review.ratings() if coach_review else {},
            coach_comments=coach_review.comments if coach_review else "",
        )

    def validate(
        self,
        cases: Sequence[tuple[str, CoachingReport, EvidencePackage]],
        *,
        coach_reviews: CoachingValidationAnnotationSet | Mapping[str, CoachReviewAnnotation] | None = None,
    ) -> CoachingValidationReport:
        review_map: dict[str, CoachReviewAnnotation] = {}
        annotation_version = ""
        if isinstance(coach_reviews, CoachingValidationAnnotationSet):
            review_map = coach_reviews.by_analysis_id()
            annotation_version = coach_reviews.annotation_version
        elif coach_reviews is not None:
            review_map = dict(coach_reviews)

        records = [
            self.validate_analysis(
                analysis_id=analysis_id,
                report=report,
                evidence=evidence,
                coach_review=review_map.get(analysis_id)
                or review_map.get(report.video)
                or review_map.get(evidence.video),
            )
            for analysis_id, report, evidence in cases
        ]
        return CoachingValidationReport(
            annotation_version=annotation_version,
            analysis_count=len(records),
            evidence_faithfulness_rate=_mean(
                [r.faithfulness_rate for r in records if r.faithfulness_rate is not None]
            ),
            hallucination_unsupported_rate=_rate_true(
                [r.has_hallucination_or_unsupported for r in records]
            ),
            mean_priority_agreement_rate=_mean(
                [
                    r.priority_agreement_rate
                    for r in records
                    if r.priority_agreement_rate is not None
                ]
            ),
            mean_coach_ratings=_mean_ratings(records),
            by_analysis_confidence=self._by_confidence(records),
            analyses=list(records),
        )

    def _by_confidence(
        self, records: Sequence[AnalysisCoachingValidation]
    ) -> list[ConfidenceBinCoachingStats]:
        out: list[ConfidenceBinCoachingStats] = []
        for label, lo, hi in self.confidence_bins:
            subset = [
                r
                for r in records
                if r.analysis_confidence is not None
                and lo <= r.analysis_confidence < hi
            ]
            out.append(
                ConfidenceBinCoachingStats(
                    key=label,
                    sample_count=len(subset),
                    evidence_faithfulness_rate=_mean(
                        [
                            r.faithfulness_rate
                            for r in subset
                            if r.faithfulness_rate is not None
                        ]
                    ),
                    hallucination_unsupported_rate=_rate_true(
                        [r.has_hallucination_or_unsupported for r in subset]
                    ),
                    mean_priority_agreement_rate=_mean(
                        [
                            r.priority_agreement_rate
                            for r in subset
                            if r.priority_agreement_rate is not None
                        ]
                    ),
                    mean_coach_ratings=_mean_ratings(subset),
                )
            )
        return out


def validate_coaching_reports(
    cases: Sequence[tuple[str, CoachingReport, EvidencePackage]],
    *,
    coach_reviews: CoachingValidationAnnotationSet
    | Mapping[str, CoachReviewAnnotation]
    | None = None,
) -> CoachingValidationReport:
    return CoachingValidator().validate(cases, coach_reviews=coach_reviews)


def export_coaching_validation_report(
    report: CoachingValidationReport,
    output_path: Path,
) -> Path:
    """Write versioned ``coaching_validation.json``."""
    return report.save_json(Path(output_path))


def _report_texts(report: CoachingReport) -> list[tuple[str, str]]:
    texts: list[tuple[str, str]] = [("summary", report.summary)]
    for i, item in enumerate(report.prioritized_issues):
        texts.append((f"prioritized_issues[{i}].explanation", item.explanation))
        texts.append((f"prioritized_issues[{i}].issue_code", item.issue_code))
    for i, strength in enumerate(report.strengths):
        texts.append((f"strengths[{i}].description", strength.description))
    for i, drill in enumerate(report.drills):
        texts.append((f"drills[{i}].name", drill.name))
        texts.append((f"drills[{i}].description", drill.description))
    for i, caveat in enumerate(report.caveats):
        texts.append((f"caveats[{i}]", caveat))
    return texts


def _priority_agreement_rate(
    predicted: Sequence[str],
    *,
    evidence_codes: set[str],
    expected: Sequence[str] | None,
) -> float | None:
    if expected:
        if not predicted and not expected:
            return 1.0
        if not predicted or not expected:
            return 0.0
        pred_set = set(predicted)
        exp_set = set(expected)
        overlap = len(pred_set & exp_set)
        return float(overlap) / float(len(exp_set))
    if not predicted:
        return 1.0 if not evidence_codes else 0.0
    hits = sum(1 for code in predicted if code in evidence_codes)
    return float(hits) / float(len(predicted))


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _rate_true(flags: Sequence[bool]) -> float | None:
    if not flags:
        return None
    return float(sum(1 for f in flags if f) / len(flags))


def _mean_ratings(
    records: Sequence[AnalysisCoachingValidation],
) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for name in COACH_RATING_DIMENSIONS:
        vals = [
            float(r.coach_ratings[name])
            for r in records
            if name in r.coach_ratings
        ]
        out[name] = _mean(vals)
    return out
