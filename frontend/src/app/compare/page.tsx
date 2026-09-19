"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Suspense, useEffect, useMemo, useState } from "react";
import { fetchAnalyses, fetchCompare, mediaUrl } from "@/lib/api";
import { formatDate, formatMetric } from "@/lib/format";
import type { AnalysisSummary, CompareResponse } from "@/lib/types";

function CompareInner() {
  const params = useSearchParams();
  const leftParam = params.get("left") || "";
  const rightParam = params.get("right") || "";

  const [history, setHistory] = useState<AnalysisSummary[]>([]);
  const [left, setLeft] = useState(leftParam);
  const [right, setRight] = useState(rightParam);
  const [result, setResult] = useState<CompareResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchAnalyses(40).then(setHistory).catch(() => setHistory([]));
  }, []);

  useEffect(() => {
    if (leftParam) setLeft(leftParam);
    if (rightParam) setRight(rightParam);
  }, [leftParam, rightParam]);

  const canCompare = Boolean(left && right && left !== right);

  async function runCompare() {
    if (!canCompare) return;
    setLoading(true);
    setError(null);
    try {
      const data = await fetchCompare(left, right);
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Compare failed");
      setResult(null);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    if (leftParam && rightParam && leftParam !== rightParam) {
      void runCompare();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const options = useMemo(
    () =>
      history.map((h) => ({
        id: h.analysis_id,
        label: `${formatDate(h.created_at)} · ${h.stroke_type || "Stroke"}`,
      })),
    [history]
  );

  return (
    <main className="mx-auto w-full max-w-4xl px-4 py-10 sm:px-6">
      <header className="mb-8 space-y-2">
        <h1 className="font-display text-3xl font-semibold">Compare sessions</h1>
        <p className="text-sm text-[var(--muted)]">
          Compare two compatible local analyses. No overall improvement score.
        </p>
      </header>

      <section className="grid gap-4 rounded-lg border border-[var(--border)] bg-[var(--bg-elevated)] p-5 sm:grid-cols-2">
        <label className="block text-sm">
          <span className="text-[var(--muted)]">Previous</span>
          <select
            className="mt-1 w-full rounded-md border border-[var(--border)] bg-white px-3 py-2"
            value={left}
            onChange={(e) => setLeft(e.target.value)}
          >
            <option value="">Select analysis</option>
            {options.map((o) => (
              <option key={o.id} value={o.id}>
                {o.label}
              </option>
            ))}
          </select>
        </label>
        <label className="block text-sm">
          <span className="text-[var(--muted)]">Current</span>
          <select
            className="mt-1 w-full rounded-md border border-[var(--border)] bg-white px-3 py-2"
            value={right}
            onChange={(e) => setRight(e.target.value)}
          >
            <option value="">Select analysis</option>
            {options.map((o) => (
              <option key={o.id} value={o.id}>
                {o.label}
              </option>
            ))}
          </select>
        </label>
        <button
          type="button"
          disabled={!canCompare || loading}
          onClick={runCompare}
          className="sm:col-span-2 rounded-md bg-[var(--accent)] px-4 py-2.5 text-sm font-semibold text-white disabled:opacity-40"
        >
          {loading ? "Comparing…" : "Compare"}
        </button>
      </section>

      {error && (
        <p className="mt-4 text-sm text-[var(--danger)]" role="alert">
          {error}
        </p>
      )}

      {result && !result.compatible && (
        <p className="mt-4 text-sm text-[var(--warning)]" role="status">
          {result.compatibility_reason}
        </p>
      )}

      {result?.compatible && (
        <section className="mt-8 space-y-6">
          <div className="overflow-x-auto rounded-lg border border-[var(--border)] bg-[var(--bg-elevated)]">
            <table className="min-w-full text-left text-sm">
              <thead className="border-b border-[var(--border)] text-xs uppercase tracking-wide text-[var(--muted)]">
                <tr>
                  <th className="px-4 py-3">Metric</th>
                  <th className="px-4 py-3">Previous</th>
                  <th className="px-4 py-3">Current</th>
                  <th className="px-4 py-3">Change</th>
                </tr>
              </thead>
              <tbody>
                {result.rows.map((row) => (
                  <tr
                    key={row.label}
                    className="border-b border-[var(--border)] last:border-0"
                  >
                    <td className="px-4 py-3">{row.label}</td>
                    <td className="px-4 py-3">
                      {formatMetric(row.left, row.unit, 2)}
                    </td>
                    <td className="px-4 py-3">
                      {formatMetric(row.right, row.unit, 2)}
                    </td>
                    <td className="px-4 py-3 font-medium">{row.change}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="grid gap-4 sm:grid-cols-2">
            {[result.left, result.right].map((side, idx) => (
              <div
                key={side.analysis_id}
                className="rounded-lg border border-[var(--border)] bg-[var(--bg-elevated)] p-4"
              >
                <p className="text-xs uppercase tracking-wide text-[var(--muted)]">
                  {idx === 0 ? "Previous" : "Current"}
                </p>
                <Link
                  href={`/analysis/${side.analysis_id}`}
                  className="mt-1 block font-medium text-[var(--accent)]"
                >
                  Open analysis
                </Link>
                {side.pose_video_url && (
                  <video
                    src={mediaUrl(side.pose_video_url) || undefined}
                    controls
                    className="mt-3 aspect-video w-full rounded-md bg-black"
                  />
                )}
              </div>
            ))}
          </div>
        </section>
      )}
    </main>
  );
}

export default function ComparePage() {
  return (
    <Suspense
      fallback={
        <p className="p-8 text-sm text-[var(--muted)]">Loading compare…</p>
      }
    >
      <CompareInner />
    </Suspense>
  );
}
