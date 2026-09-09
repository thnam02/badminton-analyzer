"""Mock ActionQualityModel — deterministic heuristics for interface smoke tests.

Not trained. Never sets ``affects_production_feedback``.
"""

from __future__ import annotations

from statistics import mean

from app.research.aqa.schemas import (
    QUALITY_DIMENSION_NAMES,
    AQASample,
    ActionQualityPrediction,
    IssueProbability,
    QualityDimensionPrediction,
    UncertaintyEstimate,
)
from app.schemas.annotation import QUALITY_RATING_TO_SCORE, QualityRating

_RATING_ORDER = [
    QualityRating.POOR.value,
    QualityRating.FAIR.value,
    QualityRating.GOOD.value,
    QualityRating.EXCELLENT.value,
]


class MockActionQualityModel:
    """Research scaffold mock: weak heuristics + high uncertainty.

    Prefer coach labels when present; otherwise derive soft scores from
    deterministic evidence. Output is always ``research_only=True``.
    """

    model_id = "mock_aqa_v0"

    def is_available(self) -> bool:
        return True

    def predict(self, sample: AQASample) -> ActionQualityPrediction:
        dims: list[QualityDimensionPrediction] = []
        for name in QUALITY_DIMENSION_NAMES:
            score, probs, unc = self._dimension_from_sample(sample, name)
            dims.append(
                QualityDimensionPrediction(
                    name=name,
                    score_mean=score,
                    rating_probabilities=probs,
                    uncertainty=unc,
                )
            )

        issues = self._issue_probabilities(sample)
        overall = mean(d.score_mean for d in dims) if dims else None
        overall_unc = UncertaintyEstimate(
            std=0.35,
            confidence=0.4,
            method="mock_uniform_prior",
        )
        return ActionQualityPrediction(
            model_id=self.model_id,
            analysis_id=sample.analysis_id,
            quality_dimensions=dims,
            issue_probabilities=issues,
            overall_quality_mean=overall,
            overall_uncertainty=overall_unc,
            research_only=True,
            affects_production_feedback=False,
            deterministic_evidence=sample.deterministic_evidence,
            notes=(
                "Mock AQA prediction for interface testing only. "
                "Do not use for athlete feedback."
            ),
        )

    def _dimension_from_sample(
        self,
        sample: AQASample,
        name: str,
    ) -> tuple[float, dict[str, float], UncertaintyEstimate]:
        # Aggregate coach scores when available.
        coach_scores: list[float] = []
        for label in sample.coach_labels:
            facet = label.get(name) or {}
            rating = facet.get("rating")
            score = facet.get("score")
            if score is None and rating is not None:
                score = QUALITY_RATING_TO_SCORE.get(str(rating))
            if isinstance(score, (int, float)) and score > 0:
                coach_scores.append(float(score))

        if coach_scores:
            score = sum(coach_scores) / len(coach_scores)
            unc = UncertaintyEstimate(
                std=0.25 if len(coach_scores) == 1 else 0.15,
                confidence=0.55 if len(coach_scores) == 1 else 0.7,
                method="mock_coach_label_mean",
            )
            return score, _soft_probs_from_score(score), unc

        # Fallback: map a few deterministic metrics to a mid prior.
        metrics = sample.deterministic_evidence.pose_metrics
        prior = 2.5
        if name == "contact_quality":
            elbow = metrics.get("contact_elbow_angle_deg")
            if isinstance(elbow, (int, float)):
                prior = 2.0 + min(2.0, max(0.0, (float(elbow) - 140.0) / 20.0))
        elif name == "kinetic_chain_timing":
            offset = metrics.get("peak_elbow_omega_offset_frames")
            if isinstance(offset, (int, float)):
                prior = 2.5 - min(1.0, abs(float(offset)) / 8.0)
        elif name == "follow_through_quality":
            ratio = metrics.get("follow_through_speed_ratio")
            if isinstance(ratio, (int, float)):
                prior = 1.5 + min(2.5, max(0.0, float(ratio) * 3.0))
        elif name == "preparation_quality":
            knee = metrics.get("knee_contribution_deg")
            if isinstance(knee, (int, float)):
                prior = 1.5 + min(2.5, max(0.0, float(knee) / 20.0))

        unc = UncertaintyEstimate(
            std=0.5,
            confidence=0.3,
            method="mock_deterministic_prior",
        )
        return float(prior), _soft_probs_from_score(prior), unc

    def _issue_probabilities(self, sample: AQASample) -> list[IssueProbability]:
        out: list[IssueProbability] = []
        seen: set[str] = set()
        for issue in sample.deterministic_evidence.technique_issues:
            code = str(issue.get("code") or "")
            if not code or code in seen:
                continue
            seen.add(code)
            # Soften rule hits into uncertain probabilities.
            sev = str(issue.get("severity") or "").lower()
            base = 0.55
            if sev == "high":
                base = 0.75
            elif sev == "low":
                base = 0.4
            out.append(
                IssueProbability(
                    code=code,
                    probability=base,
                    uncertainty=UncertaintyEstimate(
                        std=0.2,
                        confidence=0.45,
                        method="mock_from_deterministic_issue",
                    ),
                    notes="Derived from deterministic technique issue (mock).",
                )
            )
        for label in sample.coach_labels:
            for code in label.get("overall_issue_labels") or []:
                code_s = str(code)
                if code_s in seen:
                    continue
                seen.add(code_s)
                out.append(
                    IssueProbability(
                        code=code_s,
                        probability=0.6,
                        uncertainty=UncertaintyEstimate(
                            std=0.25,
                            confidence=0.5,
                            method="mock_from_coach_label",
                        ),
                    )
                )
        return out


def _soft_probs_from_score(score: float) -> dict[str, float]:
    """Map a 1–4 score to a soft categorical distribution over ratings."""
    # Distance-based softmax-ish weights.
    weights: dict[str, float] = {}
    for i, rating in enumerate(_RATING_ORDER, start=1):
        weights[rating] = max(1e-3, 1.0 / (1.0 + abs(score - float(i))))
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}
