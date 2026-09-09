# Research Action Quality Assessment (AQA)

This document describes a **research-only** scaffold for a future learned
badminton Action Quality Assessment model. It does **not** replace the
deterministic analysis pipeline (pose → phases → metrics → technique rules →
evidence → optional LLM coaching).

**Status:** interfaces, dataset loaders, and a **mock** model only.  
**No training** is performed in-repo in this milestone.  
**Production feedback must not** consume AQA predictions until the evaluation
gates below are satisfied.

## Package layout

```text
backend/app/research/aqa/
  protocol.py      # ActionQualityModel Protocol
  schemas.py       # AQASample, ActionQualityPrediction, uncertainty types
  loaders.py       # consume *_dataset.json, pose, coach labels
  mock_model.py    # MockActionQualityModel
  factory.py       # get_aqa_model("mock")
```

`analyze` / `PoseService` / coaching **do not** import this package.

## Model I/O contract

### Inputs (`AQASample`)

| Field | Source |
|-------|--------|
| Pose sequence | smoothed / raw pose JSON frames |
| Phase-normalized motion | placeholder features from phase windows + stroke metrics |
| RGB / keyframe features | keyframe file paths (embeddings empty until a vision encoder exists) |
| Reference-profile context | profile id / metric bands (provisional OK) |
| Coach labels | `{id}_annotation_{coach_id}.json` or embedded placeholders |
| Deterministic evidence | metrics, phases, contact event, technique issues (explainable) |

### Outputs (`ActionQualityPrediction`)

- **Quality dimensions** (aligned with coach facets): preparation, kinetic-chain
  timing, contact, follow-through — each with `score_mean`, rating
  probabilities, and `UncertaintyEstimate`
- **Issue probabilities** with uncertainty (soft labels, not hard rule fires)
- Flags: `research_only=True`, `affects_production_feedback=False`
- Deterministic evidence attached for explanation — biomechanics stay primary

## Candidate modeling approaches

These are **candidates**, not commitments. Implement behind `ActionQualityModel`
without changing production routing.

### 1. Temporal skeleton Transformer / ST-GCN

- Input: COCO-17 (or richer) joints over time, optionally phase-tokenized
- Graph/Transformer encodes proximal→distal coordination
- Heads: dimension regression/classification + issue multi-label
- Strengths: strong for kinematic-chain timing; interpretable joint attention
- Risks: camera view / occlusion sensitivity; needs view normalization

### 2. Siamese / expert–reference comparison

- Twin encoders: athlete clip vs reference-profile or elite exemplar sequence
- Distance / similarity features feed quality heads
- Strengths: matches “compare to reference bands” product language
- Risks: reference scarcity; provisional profiles must not be treated as ground truth

### 3. Multimodal pose + RGB fusion

- Pose stream + keyframe/RGB encoder (e.g. on contact ±N frames)
- Late or cross-attention fusion
- Strengths: contact quality / racket–shuttle cues when tracks are noisy
- Risks: domain shift, compute cost, privacy; RGB alone is brittle indoors

Hybrid stacks (skeleton backbone + optional RGB at contact) are preferred over
RGB-only for explainability.

## Evaluation gates before production feedback

A learned model may **influence** athlete-facing feedback only after documented
pass criteria on a held-out, multi-coach labeled set. Suggested minimum bar
(tune thresholds with stakeholders; do not ship on mock metrics):

| Metric | Role | Gate idea |
|--------|------|-----------|
| **Spearman / Pearson** vs mean coach quality scores (per dimension) | Ranking / calibration of quality | ρ ≥ agreed floor on held-out set |
| **QWK (quadratic weighted kappa)** on ordinal ratings | Agreement with coaches | ≥ human–human baseline − ε |
| **AUROC / AUPRC** per issue code | Issue detection quality | ≥ deterministic rule baseline where labels exist |
| **ECE / reliability diagrams** | Uncertainty calibration | ECE below agreed max; abstain when confidence low |
| **Inter-rater ICC** of coaches | Label noise context | Report; model should not exceed noisy labels’ reliability |
| **Ablation vs deterministic-only** | Added value | Show lift on dimensions rules miss (e.g. aesthetic timing) |
| **Slice metrics** | Fairness / robustness | By camera view, handedness, resolution, FPS |
| **Failure audit** | Safety | Manual review of high-confidence disagreements |

### Process gates

1. Multi-coach labels on the same `analysis_id` (see
   [dataset-annotations.md](./dataset-annotations.md)).
2. Train / val / test splits by **athlete or session**, never by frame within the
   same smash.
3. Mock / untrained backends remain `affects_production_feedback=False`.
4. Production path keeps deterministic technique issues as **explainable
   evidence**; learned scores are additive and must cite uncertainty.
5. Kill switch: config flag to disable learned heads without redeploying CV.

## Using the scaffold (research)

```python
from pathlib import Path
from app.research.aqa import get_aqa_model, AQADataset

dataset = AQADataset(Path("outputs"), include_pose_frames=False)
model = get_aqa_model("mock")
assert model.is_available()
for sample in dataset:
    pred = model.predict(sample)
    assert pred.research_only and not pred.affects_production_feedback
```

## Out of scope (this task)

- Training loops, checkpoints, GPU training code
- Wiring AQA into `/analyze`, evidence packaging, or LLM coaching prompts
- Replacing rule-based `evaluate_technique`
