# Dataset export & coach annotations

This document describes the **label-ready dataset export** produced for every
analysis. It does **not** change pose, shuttle, racket, contact resolution, or
coaching behavior. No ML model is trained here — the goal is reproducible
artifacts coaches can validate and that future training jobs can consume.

## Artifacts

For an analysis whose overlay video is `outputs/{analysis_id}_pose.mp4`:

| File | Purpose |
|------|---------|
| `{analysis_id}_dataset.json` | Versioned export bundling metadata, metrics, phases, contact, technique issues, keyframes, artifact refs, and empty coach-annotation placeholders |
| `{analysis_id}_annotation_template.json` | Blank `CoachAnnotation` example to copy |
| `{analysis_id}_annotation_{coach_id}.json` | **One file per coach** (submitted independently; not written by `/analyze`) |

`dataset_export_version` and `annotation_version` are independent semver strings
(currently both `1.0.0`). Bump the export version when the bundled analysis
fields change; bump the annotation version when coach rating fields change.

## Export contents (`*_dataset.json`)

- `video_metadata` — fps/width/height/usable confidence when available
- `pose_metrics` — stroke metrics JSON (contact-snapped when contact was resolved)
- `phases` — phase sequence including estimated/tracked contact frame
- `contact_event` — `ContactEvent` (`TRACKED_CONTACT` or `KINEMATIC_ESTIMATE`)
- `technique_issues` — rule-based issues from the technique evaluator
- `keyframes` — selected frame list (+ paths when extracted)
- `video_quality` — optional quality report
- `artifact_refs` — filenames of related JSON/video for reproducibility
- `coach_annotations` — `{ annotations: [] }` placeholder set
- `annotation_template` — embedded blank coach form

Analysis code paths are unchanged; this file is an assembly of already-written
outputs.

## Coach annotation JSON (`CoachAnnotation`)

Internal schema (`annotation_version: "1.0.0"`):

```json
{
  "annotation_version": "1.0.0",
  "analysis_id": "abc123def",
  "coach_id": "coach_jane",
  "annotated_at": "2026-09-09T09:00:00+00:00",
  "preparation_quality": { "rating": "GOOD", "score": 3, "notes": "" },
  "kinetic_chain_timing": { "rating": "FAIR", "score": 2, "notes": "Late hip drive" },
  "contact_quality": { "rating": "EXCELLENT", "score": 4, "notes": "" },
  "follow_through_quality": { "rating": "GOOD", "score": 3, "notes": "" },
  "overall_issue_labels": ["LOW_KNEE_CONTRIBUTION", "late_contact"],
  "free_text_notes": "Open stance smash; good racket speed.",
  "agrees_with_system_contact": true,
  "corrected_contact_frame_index": null,
  "corrected_contact_timestamp": null
}
```

### Rating fields

| Field | Meaning |
|-------|---------|
| `preparation_quality` | Ready position / loading before backswing |
| `kinetic_chain_timing` | Proximal-to-distal sequencing into contact |
| `contact_quality` | Contact height, timing, and solid contact feel |
| `follow_through_quality` | Deceleration and finish after contact |
| `overall_issue_labels` | Free list — may reuse system codes (e.g. `INSUFFICIENT_ELBOW_EXTENSION`) or custom tags |
| `free_text_notes` | Unstructured coach commentary |

`rating` ∈ `NOT_RATED` | `POOR` | `FAIR` | `GOOD` | `EXCELLENT`.  
Optional integer `score` maps POOR→1 … EXCELLENT→4 (omit or null for `NOT_RATED`).

Optional contact checks: set `agrees_with_system_contact` and, if disagreeing,
`corrected_contact_frame_index` / `corrected_contact_timestamp`.

## How multiple coaches label the same stroke

1. Run `/analyze` as usual → receive `dataset_json_url` and
   `annotation_template_json_url`.
2. Each coach copies the template to  
   `outputs/{analysis_id}_annotation_{coach_id}.json`  
   (use a stable unique `coach_id`, e.g. email local-part or staff id).
3. Coaches fill ratings **independently**. Do **not** edit another coach’s file.
4. Do **not** write annotations into `{analysis_id}_dataset.json` during labeling;
   keep the export immutable as the system snapshot. A future merge job can
   collect `*_annotation_*.json` beside the dataset file.
5. Disagreement is expected and valuable — store all labels; do not average
   in-place during collection.

Suggested layout after three coaches label one stroke:

```text
outputs/
  abc123_pose.mp4
  abc123_dataset.json
  abc123_annotation_template.json
  abc123_annotation_coach_jane.json
  abc123_annotation_coach_lee.json
  abc123_annotation_coach_sam.json
```

## Reproducibility

- `analysis_id` is the UUID stem shared by all artifacts for that upload.
- `artifact_refs` points at the sibling JSON/video names under `outputs/`.
- Re-running analysis creates a **new** `analysis_id`; previous labels stay
  tied to the old id.

## Out of scope (this stage)

- No annotation UI
- No automatic inter-rater agreement metrics
- No model training / fine-tuning pipeline
