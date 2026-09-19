import { formatMetric, formatPercentile } from "@/lib/format";
import type { ReferenceComparisonData } from "@/lib/types";

type Props = {
  data: ReferenceComparisonData;
  profileHint?: string | null;
};

export function ReferenceComparison({ data, profileHint }: Props) {
  const low = data.reference_low;
  const high = data.reference_high;
  const median = data.reference_median;
  const you = data.player_value;

  let markerPct = 50;
  if (low != null && high != null && high !== low) {
    markerPct = ((you - low) / (high - low)) * 100;
  } else if (median != null) {
    markerPct = you <= median ? 35 : 65;
  }
  markerPct = Math.max(0, Math.min(100, markerPct));

  return (
    <section className="rounded-lg border border-[var(--border)] bg-[var(--bg-elevated)] p-4">
      <h3 className="text-base font-semibold">{data.title || "Metric"}</h3>
      <p className="mt-1 text-sm text-[var(--muted)]">
        Compared with the {data.group_label || "reference group"}
      </p>

      <p className="mt-4 font-display text-3xl font-semibold">
        {formatMetric(you, data.unit)}
      </p>
      {formatPercentile(data.percentile) && (
        <p className="text-sm text-[var(--muted)]">
          {formatPercentile(data.percentile)} in the comparison group
        </p>
      )}

      <div className="mt-5">
        <div className="mb-1 flex justify-between text-[10px] uppercase tracking-wide text-[var(--muted)]">
          <span>P10 {formatMetric(low, data.unit)}</span>
          <span>Median {formatMetric(median, data.unit)}</span>
          <span>P90 {formatMetric(high, data.unit)}</span>
        </div>
        <div
          className="relative h-2 rounded-full bg-[var(--timeline-track)]"
          role="img"
          aria-label={`Your value ${formatMetric(you, data.unit)} versus reference median ${formatMetric(median, data.unit)}`}
        >
          {median != null && low != null && high != null && high !== low && (
            <div
              className="absolute top-0 h-full w-0.5 bg-[var(--muted)]"
              style={{
                left: `${((median - low) / (high - low)) * 100}%`,
              }}
              aria-hidden
            />
          )}
          <div
            className="absolute top-1/2 h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-[var(--bg-elevated)] bg-[var(--accent)]"
            style={{ left: `${markerPct}%` }}
            aria-hidden
          />
        </div>
        <p className="mt-2 text-xs font-medium text-[var(--accent)]">You</p>
      </div>

      <p className="mt-4 text-xs text-[var(--muted)]">
        {profileHint || data.reference_profile_id || "Reference profile"}
      </p>
    </section>
  );
}
