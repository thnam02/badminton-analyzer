"use client";

import { useState } from "react";
import { formatMetric, formatPercentile } from "@/lib/format";
import type { CoachingReportView } from "@/lib/types";

type Props = {
  coaching: CoachingReportView;
  onSeek?: (seconds: number) => void;
};

export function CoachingFocus({ coaching, onSeek }: Props) {
  const [whyOpen, setWhyOpen] = useState(false);

  if (!coaching.available) {
    return (
      <section className="rounded-lg border border-[var(--border)] bg-[var(--bg-elevated)] p-5">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-[var(--muted)]">
          Coaching
        </h2>
        <p className="mt-2 text-sm text-[var(--muted)]" role="status">
          Analysis complete. Coaching guidance is temporarily unavailable.
        </p>
        {coaching.caveats?.map((c) => (
          <p key={c} className="mt-1 text-sm text-[var(--muted)]">
            {c}
          </p>
        ))}
      </section>
    );
  }

  const main = coaching.main_focus;

  return (
    <section className="space-y-5">
      <div className="rounded-lg border border-[var(--border)] bg-[var(--bg-elevated)] p-5">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-[var(--muted)]">
          Main focus
        </h2>
        {main ? (
          <>
            <h3 className="font-display mt-2 text-2xl font-semibold">
              {main.title}
            </h3>
            <p className="mt-3 text-sm leading-relaxed text-[var(--fg)]">
              {main.explanation}
            </p>
            {main.evidence && (
              <div className="mt-4 rounded-md bg-[var(--bg)] p-3 text-sm">
                <p className="text-xs font-semibold uppercase tracking-wide text-[var(--muted)]">
                  Measurement evidence
                </p>
                <p className="mt-1">
                  {main.title}:{" "}
                  {formatMetric(main.evidence.measured, main.evidence.unit)}
                </p>
                <p>
                  Reference median:{" "}
                  {formatMetric(
                    main.evidence.reference_median,
                    main.evidence.unit
                  )}
                </p>
                {formatPercentile(main.evidence.reference_percentile) && (
                  <p>{formatPercentile(main.evidence.reference_percentile)}</p>
                )}
              </div>
            )}
            {main.seek_timestamp != null && onSeek && (
              <button
                type="button"
                className="mt-3 text-sm font-medium text-[var(--accent)]"
                onClick={() => onSeek(main.seek_timestamp!)}
              >
                Show this moment in the video
              </button>
            )}
            <button
              type="button"
              className="mt-3 block text-sm text-[var(--muted)] underline-offset-2 hover:underline"
              aria-expanded={whyOpen}
              onClick={() => setWhyOpen((v) => !v)}
            >
              Why am I seeing this?
            </button>
            {whyOpen && (
              <p className="mt-2 text-xs text-[var(--muted)]">
                This coaching text explains deterministic technique findings. The
                numbers above come from measurement and the reference group — not
                from the AI narrative alone.
              </p>
            )}
          </>
        ) : (
          <p className="mt-2 text-sm text-[var(--muted)]">
            {coaching.summary || "No primary coaching focus for this stroke."}
          </p>
        )}
      </div>

      {coaching.secondary.length > 0 && (
        <div>
          <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-[var(--muted)]">
            Secondary findings
          </h2>
          <ul className="space-y-3">
            {coaching.secondary.map((item) => (
              <li
                key={item.issue_code}
                className="rounded-lg border border-[var(--border)] bg-[var(--bg-elevated)] p-4"
              >
                <h3 className="font-semibold">{item.title}</h3>
                <p className="mt-1 text-sm text-[var(--muted)]">
                  {item.explanation}
                </p>
              </li>
            ))}
          </ul>
        </div>
      )}

      {coaching.strengths.length > 0 && (
        <div>
          <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-[var(--muted)]">
            What you did well
          </h2>
          <ul className="space-y-2">
            {coaching.strengths.map((s) => (
              <li
                key={s.description}
                className="rounded-lg border border-[var(--border)] bg-[var(--bg-elevated)] px-4 py-3 text-sm"
              >
                {s.description}
              </li>
            ))}
          </ul>
        </div>
      )}
    </section>
  );
}
