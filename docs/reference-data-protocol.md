# Reference-data protocol (forehand smash)

Versioned collection protocol and dataset **metadata** schema for building a
badminton smash reference corpus. This document describes **how to record** and
**how to catalog** strokes. It does **not** compute technique reference ranges
and does **not** change production technique rules, thresholds, contact logic,
OpenAI prompts, or WHAM.

| Version field | Current | Bump when… |
|---------------|---------|------------|
| `reference_data_protocol_version` | `1.0.0` | Camera placement, FPS/resolution floors, or acceptance gates change |
| `reference_dataset_schema_version` | `1.0.0` | Player / session / stroke / label field shapes change |

Typed schemas live in `backend/app/schemas/reference_dataset.py`.

---

## Scope (initial collection)

- **Stroke:** forehand smash only (`FOREHAND_SMASH`)
- **Camera protocol:** one controlled view per session — `rear_45` or `side_45`
- **Players:** approximately **5–10 per skill group**
- **Skill groups:** `beginner`, `intermediate`, `advanced_expert` (aliases: `advanced`, `expert`)
- **Volume:** approximately **15–30 valid smashes per player**
- **Session repeats:** plan **≥15** smashes per session (**~25 preferred**) so enough takes survive acceptance

Handedness (`LEFT` / `RIGHT`) must be recorded for every player and stroke.

---

## Controlled recording specification

### Camera placement / view

| View ID | Intent |
|---------|--------|
| `rear_45` | Behind the player, ~45° off the sagittal plane (racket arm visible) |
| `side_45` | Side-on, ~45° toward the court (full kinetic chain visible) |

Use a **stable** tripod. Avoid panning/zooming during the smash. Keep the full
body (ankles through head) and racket arm in frame for the entire stroke.

### Capture settings

| Setting | Minimum | Preferred |
|---------|---------|-----------|
| FPS | **30** | **60** |
| Resolution | **1280×720** | **1920×1080** |
| Full-body visibility | Required | Required |
| Handedness | Recorded | Recorded |
| Skill group | Recorded | Recorded |
| Repeated smashes / session | ≥15 | ~25 |

Optional metadata (recommended): approximate camera height (m) and distance to
player (m) on `CameraSetup`.

---

## Dataset metadata schemas

| Type | Role |
|------|------|
| `Player` | Anonymized player ID, skill group, handedness |
| `CameraSetup` | View, FPS, resolution, optional height/distance |
| `RecordingSession` | Session ID, player, camera setup, planned smash count |
| `StrokeSample` | One take: stroke type, take number, capture metadata, acceptance, analysis artifact refs |
| `ReferenceLabel` | Optional human labels (issue tags, quality tags) — **no percentile ranges** |
| `ReferenceDatasetManifest` | Versioned bundle of the above + embedded protocol snapshot |

`StrokeSample.artifact_refs` may point at existing pipeline outputs (e.g.
`{analysis_id}_dataset.json`, pose/phases/contact/technique JSON, overlay
video) without re-running analysis inside this schema.

Example manifest: `backend/data/reference_dataset/example_reference_dataset.json`.

---

## Acceptance criteria (valid stroke)

A stroke is **accepted** into the reference corpus only if all of the following
hold (see `evaluate_stroke_acceptance`):

1. **Full body visible** — ankles through head in frame for the smash window  
2. **No major occlusion** — racket arm / torso not heavily blocked at contact  
3. **Stable camera** — no large shake or intentional pan/zoom during the stroke  
4. **Pose coverage above threshold** — default **≥ 0.75** fraction of expected body joints confidently observed  
5. **Contact event available or estimable** — tracked contact **or** kinematic estimate present after analysis  
6. **Protocol capture floors** — camera view ∈ `{rear_45, side_45}`, FPS ≥ 30, resolution ≥ 1280×720  

Failed criteria are stored on `StrokeSample.acceptance_failures`. Rejected takes
remain in the manifest for audit but should not feed future reference-range
fitting.

---

## Explicit non-goals (this version)

- Do **not** calculate metric medians / percentiles / `ReferenceProfile` bands  
- Do **not** modify technique rules, severity thresholds, or contact resolution  
- Do **not** replace the per-analysis `*_dataset.json` export (coach validation);
  this protocol catalogs **collection** metadata across players/sessions  

Future work can derive provisional reference ranges **from accepted strokes
only**, under a separate versioned fitting step.

## Building empirical reference profiles

Offline builder: `backend/app/processing/reference_profile_builder.py`.

- Inputs: `ReferenceDatasetManifest` (accepted `StrokeSample`s) + finalized
  analysis metrics (`pose_metrics` / `*_stroke_metrics.json`, optional motion peaks)
- Groups by `stroke_type`, `handedness`, `camera_view`, `skill_level`
- Emits versioned `BuiltReferenceProfile` / `BuiltReferenceProfileSet`
  (`reference_profile_build_version`) with per-metric median/mean/std,
  P10/P25/P75/P90, IQR, missing-rate, and quality summary
- Optionally drops low-quality analyses via `VideoQualityReport`
- Production evaluation path (C3–C6): see
  [`docs/reference-based-technique-evaluation.md`](reference-based-technique-evaluation.md)
  — selector → calibrated evaluator → coach A/B validation → immutable
  versioned profiles. Provisional catalogs remain marked until explicitly
  validated.
