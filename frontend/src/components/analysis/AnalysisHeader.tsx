import { confidenceLabel, formatDate } from "@/lib/format";
import type { AnalysisResult } from "@/lib/types";

type Props = {
  analysis: AnalysisResult;
};

export function AnalysisHeader({ analysis }: Props) {
  const parts = [
    analysis.stroke_type,
    analysis.handedness,
    analysis.confidence
      ? `Confidence ${confidenceLabel(analysis.confidence.overall)}`
      : null,
  ].filter(Boolean);

  return (
    <header className="space-y-2 border-b border-[var(--border)] pb-5">
      <p className="text-xs font-semibold uppercase tracking-[0.14em] text-[var(--muted)]">
        Analysis result
      </p>
      <h1 className="font-display text-3xl font-semibold tracking-tight text-[var(--fg)] sm:text-4xl">
        {analysis.stroke_type}
      </h1>
      <p className="text-sm text-[var(--muted)]">{parts.join(" · ")}</p>
      <p className="text-xs text-[var(--muted)]">
        {formatDate(analysis.created_at)}
        {analysis.reference_profile_id
          ? ` · Reference ${analysis.reference_profile_id}`
          : ""}
      </p>
      {analysis.analysis_status !== "COMPLETE" && (
        <p className="text-sm text-[var(--warning)]" role="status">
          This analysis is partial — some artifacts may be missing.
        </p>
      )}
    </header>
  );
}
