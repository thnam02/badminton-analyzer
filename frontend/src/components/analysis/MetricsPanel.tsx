import { formatMetric } from "@/lib/format";

type Metric = { id: string; label: string; value: number; unit: string };

type Props = {
  metrics: Metric[];
};

export function MetricsPanel({ metrics }: Props) {
  if (!metrics.length) {
    return (
      <p className="text-sm text-[var(--muted)]" role="status">
        Stroke metrics are not available for this analysis.
      </p>
    );
  }

  return (
    <div className="overflow-x-auto rounded-lg border border-[var(--border)] bg-[var(--bg-elevated)]">
      <table className="min-w-full text-left text-sm">
        <caption className="sr-only">Stroke metrics</caption>
        <thead className="border-b border-[var(--border)] text-xs uppercase tracking-wide text-[var(--muted)]">
          <tr>
            <th className="px-4 py-3 font-semibold">Metric</th>
            <th className="px-4 py-3 font-semibold">Value</th>
          </tr>
        </thead>
        <tbody>
          {metrics.map((m) => (
            <tr key={m.id} className="border-b border-[var(--border)] last:border-0">
              <td className="px-4 py-3">{m.label}</td>
              <td className="px-4 py-3 font-medium">
                {formatMetric(m.value, m.unit, m.unit === "" ? 2 : 1)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
