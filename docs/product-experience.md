# Phase D — Product experience

Product UX for the existing deterministic technique pipeline. No biomechanics,
reference, or coaching-model changes.

## Routes

| Route | Purpose |
|-------|---------|
| `/` | Landing |
| `/analyze` | Guided upload + processing stages |
| `/analysis/[id]` | Full analysis result |
| `/history` | Local file-backed analysis list |
| `/compare?left=&right=` | Session comparison |

## Backend API

| Endpoint | Role |
|----------|------|
| `GET /analyses` | Recent summaries (manifest + disk discovery) |
| `GET /analyses/{id}` | Product aggregate (phases, contact, issues, coaching, confidence) |
| `GET /compare?left=&right=` | Compatible metric comparison |
| `POST /analyze` | Unchanged pipeline; now returns `analysis_id`, soft coaching, writes history index |

Aggregation reads finalized artifacts only — never recomputes analysis.

History index: `outputs/analyses_index.json`.

## Frontend structure

```text
frontend/src/
├── app/
│   ├── page.tsx
│   ├── analyze/page.tsx
│   ├── analysis/[id]/page.tsx
│   ├── history/page.tsx
│   └── compare/page.tsx
├── components/analysis/
│   ├── AnalysisHeader.tsx
│   ├── AnalysisResultView.tsx
│   ├── VideoPlayer.tsx
│   ├── PhaseTimeline.tsx
│   ├── TechniqueIssueCard.tsx
│   ├── ReferenceComparison.tsx
│   ├── CoachingFocus.tsx
│   ├── DrillCard.tsx
│   ├── ConfidenceSummary.tsx
│   ├── MetricsPanel.tsx
│   └── OverlayControls.tsx
└── lib/
    ├── api.ts
    ├── types.ts
    └── format.ts
```

## Product contracts

Normalized by `GET /analyses/{id}`:

- `AnalysisResult` with separate `analysis_status` / `coaching_status` / `mesh_status`
- Findings vs `INSUFFICIENT_EVIDENCE`
- Coaching unavailable does not fail the page
- Confidence as High / Moderate / Low (not a fake %)
- No overall technique score

## Verification notes

Use an existing analysis id with phases + technique artifacts, e.g. from
`outputs/{id}_pose_phases.json`. Older artifacts without coaching still load;
coaching section shows the unavailable state.
