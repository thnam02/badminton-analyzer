import type { Drill } from "@/lib/types";
import { issueTitleFromCode } from "@/lib/format";

type Props = {
  drill: Drill;
};

export function DrillCard({ drill }: Props) {
  return (
    <article className="rounded-lg border border-[var(--border)] bg-[var(--bg-elevated)] p-5">
      <p className="text-xs font-semibold uppercase tracking-wide text-[var(--muted)]">
        Next session focus
      </p>
      <h3 className="font-display mt-2 text-2xl font-semibold">{drill.title}</h3>
      {drill.repetitions && (
        <p className="mt-1 text-sm font-medium text-[var(--accent)]">
          {drill.repetitions}
        </p>
      )}
      <p className="mt-3 text-sm leading-relaxed text-[var(--fg)]">
        {drill.goal || drill.instructions}
      </p>
      {drill.related_issue && (
        <p className="mt-4 text-xs text-[var(--muted)]">
          Related to: {issueTitleFromCode(drill.related_issue)}
        </p>
      )}
    </article>
  );
}
