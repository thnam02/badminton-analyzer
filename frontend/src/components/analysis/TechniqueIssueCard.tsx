"use client";

import { useState } from "react";
import { formatPercentile, confidenceLabel } from "@/lib/format";
import {
  coachingCopyForIssue,
  findingSeverityLabel,
  formatConfidencePct,
  formatEvidenceValue,
  formatPlayerMeasurement,
  phaseChipLabel,
  playerIssueTitle,
} from "@/lib/techniqueCopy";
import type { TechniqueIssueView } from "@/lib/types";

type Variant = "main" | "compact" | "insufficient";

type Props = {
  issue: TechniqueIssueView;
  variant?: Variant;
  active?: boolean;
  fps?: number | null;
  onShowMoment?: (issue: TechniqueIssueView) => void;
};

export function TechniqueIssueCard({
  issue,
  variant,
  active,
  fps,
  onShowMoment,
}: Props) {
  const mode: Variant =
    variant ||
    (issue.is_insufficient_evidence ? "insufficient" : "compact");

  if (mode === "insufficient" || issue.is_insufficient_evidence) {
    return <InsufficientCard issue={issue} active={active} />;
  }

  if (mode === "main") {
    return (
      <MainFocusCard
        issue={issue}
        active={active}
        fps={fps}
        onShowMoment={onShowMoment}
      />
    );
  }

  return (
    <CompactFindingCard
      issue={issue}
      active={active}
      fps={fps}
      onShowMoment={onShowMoment}
    />
  );
}

function InsufficientCard({
  issue,
  active,
}: {
  issue: TechniqueIssueView;
  active?: boolean;
}) {
  return (
    <article
      className={`rounded-lg border border-dashed border-[var(--border)] bg-[var(--bg)] p-4 ${
        active ? "ring-2 ring-[var(--accent)]" : ""
      }`}
      aria-label="Unable to assess reliably"
    >
      <p className="text-xs font-semibold uppercase tracking-wide text-[var(--muted)]">
        Unable to assess reliably
      </p>
      <h3 className="mt-1 font-display text-lg font-semibold text-[var(--fg)]">
        {playerIssueTitle(issue.code)}
      </h3>
      <p className="mt-2 text-sm leading-relaxed text-[var(--muted)]">
        {issue.status_reason ||
          "We could not measure this part of the stroke with enough confidence to give coaching advice."}
      </p>
      <p className="mt-3 text-xs text-[var(--muted)]">
        {phaseChipLabel(issue.phase_label)} · not counted as a technique fault
      </p>
    </article>
  );
}

function MainFocusCard({
  issue,
  active,
  fps,
  onShowMoment,
}: {
  issue: TechniqueIssueView;
  active?: boolean;
  fps?: number | null;
  onShowMoment?: (issue: TechniqueIssueView) => void;
}) {
  const [evidenceOpen, setEvidenceOpen] = useState(false);
  const copy = coachingCopyForIssue(issue);
  const plainMeasure = formatPlayerMeasurement(issue, fps);

  return (
    <article
      className={`rounded-xl border border-[var(--accent)]/30 bg-[var(--bg-elevated)] p-5 sm:p-6 ${
        active ? "ring-2 ring-[var(--accent)]" : ""
      }`}
    >
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0 space-y-2">
          <p className="text-xs font-semibold uppercase tracking-[0.14em] text-[var(--accent)]">
            Main focus
          </p>
          <h3 className="font-display text-2xl font-semibold tracking-tight text-[var(--fg)] sm:text-3xl">
            {copy.title}
          </h3>
          <p className="text-sm text-[var(--muted)]">
            {phaseChipLabel(issue.phase_label)} · {findingSeverityLabel(issue)}
          </p>
        </div>
        <ActionRow
          issue={issue}
          evidenceOpen={evidenceOpen}
          onToggleEvidence={() => setEvidenceOpen((v) => !v)}
          onShowMoment={onShowMoment}
        />
      </div>

      <div className="mt-5 space-y-4">
        <CopyBlock label="What happened" body={copy.whatHappened} />
        {plainMeasure !== "—" && (
          <p className="text-sm text-[var(--muted)]">
            In plain terms: <span className="text-[var(--fg)]">{plainMeasure}</span>
          </p>
        )}
        <CopyBlock label="Why it matters" body={copy.whyItMatters} />
        <div className="rounded-lg bg-[var(--accent-soft)] px-4 py-3">
          <p className="text-xs font-semibold uppercase tracking-wide text-[var(--accent)]">
            What to do next
          </p>
          <p className="mt-1 text-base font-medium leading-snug text-[var(--fg)]">
            {copy.coachingFocus}
          </p>
        </div>
      </div>

      {evidenceOpen && <EvidencePanel issue={issue} />}
    </article>
  );
}

function CompactFindingCard({
  issue,
  active,
  fps,
  onShowMoment,
}: {
  issue: TechniqueIssueView;
  active?: boolean;
  fps?: number | null;
  onShowMoment?: (issue: TechniqueIssueView) => void;
}) {
  const [open, setOpen] = useState(false);
  const [evidenceOpen, setEvidenceOpen] = useState(false);
  const copy = coachingCopyForIssue(issue);
  const plainMeasure = formatPlayerMeasurement(issue, fps);

  return (
    <article
      className={`rounded-lg border border-[var(--border)] bg-[var(--bg-elevated)] ${
        active ? "ring-2 ring-[var(--accent)]" : ""
      }`}
    >
      <button
        type="button"
        className="flex w-full items-start justify-between gap-3 px-4 py-3 text-left"
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
      >
        <div className="min-w-0">
          <h3 className="text-base font-semibold text-[var(--fg)]">{copy.title}</h3>
          <p className="mt-1 text-xs text-[var(--muted)]">
            {phaseChipLabel(issue.phase_label)} · {findingSeverityLabel(issue)}
          </p>
        </div>
        <span className="mt-0.5 shrink-0 text-xs font-medium text-[var(--accent)]">
          {open ? "Hide" : "Details"}
        </span>
      </button>

      {open && (
        <div className="space-y-3 border-t border-[var(--border)] px-4 py-3">
          <CopyBlock label="What happened" body={copy.whatHappened} compact />
          {plainMeasure !== "—" && (
            <p className="text-sm text-[var(--muted)]">
              In plain terms:{" "}
              <span className="text-[var(--fg)]">{plainMeasure}</span>
            </p>
          )}
          <CopyBlock label="Why it matters" body={copy.whyItMatters} compact />
          <div className="rounded-md bg-[var(--bg)] px-3 py-2">
            <p className="text-xs font-semibold uppercase tracking-wide text-[var(--muted)]">
              What to do next
            </p>
            <p className="mt-1 text-sm font-medium text-[var(--fg)]">
              {copy.coachingFocus}
            </p>
          </div>
          <ActionRow
            issue={issue}
            evidenceOpen={evidenceOpen}
            onToggleEvidence={() => setEvidenceOpen((v) => !v)}
            onShowMoment={onShowMoment}
          />
          {evidenceOpen && <EvidencePanel issue={issue} />}
        </div>
      )}
    </article>
  );
}

function CopyBlock({
  label,
  body,
  compact,
}: {
  label: string;
  body: string;
  compact?: boolean;
}) {
  return (
    <div>
      <p
        className={`font-semibold uppercase tracking-wide text-[var(--muted)] ${
          compact ? "text-[0.65rem]" : "text-xs"
        }`}
      >
        {label}
      </p>
      <p
        className={`mt-1 leading-relaxed text-[var(--fg)] ${
          compact ? "text-sm" : "text-sm sm:text-base"
        }`}
      >
        {body}
      </p>
    </div>
  );
}

function ActionRow({
  issue,
  evidenceOpen,
  onToggleEvidence,
  onShowMoment,
}: {
  issue: TechniqueIssueView;
  evidenceOpen: boolean;
  onToggleEvidence: () => void;
  onShowMoment?: (issue: TechniqueIssueView) => void;
}) {
  return (
    <div className="flex flex-wrap items-center gap-2">
      {issue.seek_timestamp != null && onShowMoment && (
        <button
          type="button"
          onClick={() => onShowMoment(issue)}
          className="rounded-md border border-[var(--border)] bg-[var(--bg)] px-3 py-2 text-sm font-medium text-[var(--accent)] hover:border-[var(--accent)]"
        >
          Show this moment
        </button>
      )}
      <button
        type="button"
        className="rounded-md px-3 py-2 text-sm font-medium text-[var(--muted)] underline-offset-2 hover:text-[var(--accent)] hover:underline"
        aria-expanded={evidenceOpen}
        onClick={onToggleEvidence}
      >
        {evidenceOpen ? "Hide evidence" : "View evidence"}
      </button>
    </div>
  );
}

function EvidencePanel({ issue }: { issue: TechniqueIssueView }) {
  const combined =
    issue.combined_confidence ?? issue.confidence ?? null;

  return (
    <div className="mt-3 rounded-md border border-[var(--border)] bg-[var(--bg)] p-3 text-xs text-[var(--muted)]">
      <p className="font-semibold uppercase tracking-wide text-[var(--muted)]">
        Measurement evidence
      </p>
      <dl className="mt-2 grid gap-2 sm:grid-cols-2">
        <div>
          <dt>Measured value</dt>
          <dd className="font-mono text-[var(--fg)]">
            {formatEvidenceValue(issue.measured_value, issue.unit)}
          </dd>
        </div>
        <div>
          <dt>Reference median</dt>
          <dd className="font-mono text-[var(--fg)]">
            {formatEvidenceValue(issue.reference_median, issue.unit)}
          </dd>
        </div>
        <div>
          <dt>Reference range</dt>
          <dd className="font-mono text-[var(--fg)]">
            {formatEvidenceValue(issue.reference_low, issue.unit)} –{" "}
            {formatEvidenceValue(issue.reference_high, issue.unit)}
          </dd>
        </div>
        <div>
          <dt>Percentile vs reference</dt>
          <dd className="font-mono text-[var(--fg)]">
            {formatPercentile(issue.reference_percentile) || "—"}
          </dd>
        </div>
        <div>
          <dt>Confidence</dt>
          <dd className="text-[var(--fg)]">
            {confidenceLabel(issue.confidence_label)} (
            {formatConfidencePct(combined)})
          </dd>
        </div>
        <div className="sm:col-span-2">
          <dt>Reference profile</dt>
          <dd className="break-all font-mono text-[var(--fg)]">
            {issue.reference_profile_id || "—"}
          </dd>
        </div>
      </dl>
    </div>
  );
}
