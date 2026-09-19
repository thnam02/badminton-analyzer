"use client";

import { useState } from "react";
import { confidenceLabel } from "@/lib/format";
import type { ConfidenceSummary } from "@/lib/types";

type Props = {
  confidence: ConfidenceSummary;
  limitations?: string[];
};

export function ConfidenceSummaryPanel({ confidence, limitations }: Props) {
  const [open, setOpen] = useState(false);

  return (
    <section className="rounded-lg border border-[var(--border)] bg-[var(--bg-elevated)] p-4">
      <button
        type="button"
        className="flex w-full items-center justify-between text-left"
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
      >
        <div>
          <h2 className="text-sm font-semibold uppercase tracking-wide text-[var(--muted)]">
            Analysis confidence
          </h2>
          <p className="mt-1 text-lg font-semibold">
            {confidenceLabel(confidence.overall)}
          </p>
        </div>
        <span className="text-sm text-[var(--accent)]">
          {open ? "Hide details" : "Details"}
        </span>
      </button>

      {confidence.message && (
        <p className="mt-3 text-sm text-[var(--muted)]">{confidence.message}</p>
      )}

      {open && (
        <ul className="mt-4 space-y-2 border-t border-[var(--border)] pt-3">
          {confidence.components.map((c) => (
            <li
              key={c.id}
              className="flex items-center justify-between text-sm"
            >
              <span>{c.label}</span>
              <span
                className="font-medium"
                aria-label={`${c.label}: ${confidenceLabel(c.level)}`}
              >
                {confidenceLabel(c.level)}
              </span>
            </li>
          ))}
        </ul>
      )}

      {limitations && limitations.length > 0 && (
        <ul className="mt-4 space-y-1 border-t border-[var(--border)] pt-3 text-sm text-[var(--muted)]">
          {limitations.map((note) => (
            <li key={note}>{note}</li>
          ))}
        </ul>
      )}
    </section>
  );
}
