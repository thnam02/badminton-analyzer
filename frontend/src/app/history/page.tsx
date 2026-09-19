"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { fetchAnalyses } from "@/lib/api";
import { formatDate, confidenceLabel, issueTitleFromCode } from "@/lib/format";
import type { AnalysisSummary } from "@/lib/types";

function band(score?: number | null): string {
  if (score == null) return "—";
  if (score >= 0.75) return confidenceLabel("HIGH");
  if (score >= 0.5) return confidenceLabel("MODERATE");
  return confidenceLabel("LOW");
}

export default function HistoryPage() {
  const [items, setItems] = useState<AnalysisSummary[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchAnalyses(40)
      .then(setItems)
      .catch((err) =>
        setError(err instanceof Error ? err.message : "Could not load history")
      )
      .finally(() => setLoading(false));
  }, []);

  return (
    <main className="mx-auto w-full max-w-3xl px-4 py-10 sm:px-6">
      <header className="mb-8 space-y-2">
        <h1 className="font-display text-3xl font-semibold">History</h1>
        <p className="text-sm text-[var(--muted)]">
          Recent analyses on this machine. No account required.
        </p>
      </header>

      {loading && (
        <p className="text-sm text-[var(--muted)]" role="status">
          Loading history…
        </p>
      )}
      {error && (
        <p className="text-sm text-[var(--danger)]" role="alert">
          {error}
        </p>
      )}
      {!loading && !error && items.length === 0 && (
        <div className="rounded-lg border border-[var(--border)] bg-[var(--bg-elevated)] p-6">
          <p className="text-sm text-[var(--muted)]">
            No analyses yet.{" "}
            <Link href="/analyze" className="text-[var(--accent)]">
              Analyze a stroke
            </Link>
            .
          </p>
        </div>
      )}

      <ul className="space-y-3">
        {items.map((item) => (
          <li key={item.analysis_id}>
            <Link
              href={`/analysis/${item.analysis_id}`}
              className="block rounded-lg border border-[var(--border)] bg-[var(--bg-elevated)] p-4 transition hover:border-[var(--accent)]"
            >
              <div className="flex flex-wrap items-start justify-between gap-2">
                <div>
                  <p className="font-semibold">
                    {item.stroke_type || "Forehand Smash"}
                  </p>
                  <p className="mt-1 text-xs text-[var(--muted)]">
                    {formatDate(item.created_at)}
                  </p>
                </div>
                <p className="text-xs font-medium text-[var(--muted)]">
                  Confidence {band(item.analysis_confidence)}
                </p>
              </div>
              <p className="mt-3 text-sm text-[var(--muted)]">
                Main finding:{" "}
                {item.main_issue
                  ? issueTitleFromCode(item.main_issue)
                  : "None flagged"}
              </p>
            </Link>
          </li>
        ))}
      </ul>
    </main>
  );
}
