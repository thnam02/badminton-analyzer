# Reference-based technique evaluation (Part C)

This document describes how smash technique issues are decided from
**measurements + reference distributions + calibrated confidence**, and how
those decisions are validated against coach labels. It does **not** invent
scientifically validated population norms: provisional profiles remain clearly
marked until a coach-validated configuration is explicitly promoted.

| Concern | Module |
|---------|--------|
| Protocol / corpus catalog | `docs/reference-data-protocol.md`, `schemas/reference_dataset.py` |
| Statistical profiles (C2) | `processing/reference_profile_builder.py`, `schemas/reference.py` |
| Profile selection (C3/C6) | `processing/reference_profile_selector.py`, `schemas/reference_profile_registry.py` |
| Calibrated evaluator (C4) | `processing/technique.py`, `schemas/technique_calibration.py` |
| Legacy A/B path (C5) | `processing/technique_legacy.py` |
| Coach validation (C5) | `validation/technique_validator.py`, `validation/technique_ab_comparison.py` |

---

## End-to-end decision path

```text
FinalAnalysisState
        ↓
ReferenceProfileSelector  (or explicit profile_id)
        ↓
exact immutable ReferenceProfile  (never "latest")
        ↓
StrokeMetrics
        ↓
distribution-aware TechniqueEvaluator
        ↓
EvaluationConfidence + SeverityCalibrationConfig
        ↓
TechniqueIssue
  NO_ISSUE | MINOR | MODERATE | MAJOR | INSUFFICIENT_EVIDENCE
```

Every emitted issue is explainable as:

```text
measurement
+ reference distribution (median / P10–P90 / robust z)
+ reference percentile position
+ measurement / phase / pose / video-quality / reference confidence
+ rule_version + severity_calibration_version
= technique decision
```

---

## C4 — Severity and confidence calibration

`SeverityCalibrationConfig` (versioned, default `severity_calibration_version=1.0.0`)
maps **percentile position** and optional **robust z** to status bands:

| Region (higher_is_better, conceptual) | Status |
|---------------------------------------|--------|
| Central reference band (e.g. P20–P80) | `NO_ISSUE` (no issue object) |
| Slightly outside | `MINOR` |
| Clear deviation | `MODERATE` |
| Extreme deviation | `MAJOR` |

Thresholds (`major_percentile_max`, `moderate_percentile_max`, `minor_percentile_max`,
`central_percentile_*`, `*_abs_z`) are **configurable and versioned** — do not
treat a single percentile cross as an automatic strong coaching claim.

`EvaluationConfidence` propagates:

- `measurement_confidence`
- `phase_confidence`
- `video_quality_confidence`
- `reference_confidence`
- `pose_confidence`
- `combined_confidence`

Combination method is configurable: `weighted_mean` (default), `min`, or
`geometric_mean`, with `ConfidenceWeights`.

If confidence or reference sample count is too low for an **adverse**
measurement, the evaluator returns:

```json
{
  "code": "INSUFFICIENT_ELBOW_EXTENSION",
  "status": "INSUFFICIENT_EVIDENCE",
  "status_reason": "Combined confidence below calibrated minimum",
  "combined_confidence": 0.41
}
```

rather than a confident `MAJOR` / `MODERATE` coaching issue.

Backward-compatible `severity` (`LOW`/`MEDIUM`/`HIGH`) is derived from status
via `status_to_severity`; **`status` is authoritative**.

---

## C5 — Coach-label validation (legacy vs reference)

Keep `evaluate_technique_legacy` for A/B comparison. Offline tooling:

```python
from app.validation import compare_legacy_vs_reference, build_validation_decision
```

Both evaluators run on the **same** coach-labelled strokes. Per issue:

- precision, recall, F1, specificity, FPR, FNR, support, confusion matrix

Also:

- `INSUFFICIENT_EVIDENCE` rate
- coverage rate (judgements vs refusals)

Stratify (when enough samples) by camera view, skill level, analysis /
pose / contact confidence, and reference profile.

Severity agreement (when coach severity is labelled):

- exact agreement
- within-one-level agreement

**Promotion is explicit.** `ValidationDecisionArtifact` records:

```json
{
  "evaluator_version": "reference_v1",
  "validated": false,
  "insufficient_data": true,
  "validation_dataset_version": "coach_smash_v1",
  "notes": "...",
  "metrics": {}
}
```

Production must not default to an unvalidated reference configuration solely
because the code path exists.

---

## C6 — Provenance, freezing, versioning

`VersionedReferenceProfile` statuses: `DRAFT` → `VALIDATING` → `VALIDATED` →
`DEPRECATED`.

A `VALIDATED` profile is **immutable**. Metric or definition changes create
`v2` via `bump_profile_version` — never rewrite `v1`.

Dataset fingerprints (`deterministic_dataset_fingerprint`) hash canonical
sample IDs + metric values only (no timestamps or temp paths).

Manifests record included/excluded samples and exclusion reasons.

`ReferenceProfileSelector` / `select_latest_validated_compatible` may *request*
the latest validated compatible profile, but analysis artifacts store the
**resolved exact `profile_id`**. Literal `"latest"` is rejected.

Historical analyses that recorded `…_v1` must keep reproducing with `v1` even
after `v2` exists.

---

## Production defaults

- Prefer an explicitly selected / settings `profile_id` that points at a
  concrete versioned profile.
- When using the registry, resolve `latest_validated_compatible` **before**
  analysis and pass the concrete ID into `evaluate_technique`.
- If no compatible profile exists (`match_level=none`), use clearly marked
  provisional hard-coded fallbacks — never mix settings thresholds into a
  distribution decision.
- Legacy rules remain available for validation comparison only.
