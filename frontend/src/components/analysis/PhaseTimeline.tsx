"use client";

import type { PhaseSegment } from "@/lib/types";
import { formatTimestamp } from "@/lib/format";

type Props = {
  phases: PhaseSegment[];
  currentTime: number;
  onSeek: (seconds: number) => void;
  contactTimestamp?: number | null;
};

export function PhaseTimeline({
  phases,
  currentTime,
  onSeek,
  contactTimestamp,
}: Props) {
  if (!phases.length) {
    return (
      <p className="text-sm text-[var(--muted)]" role="status">
        Stroke phase timeline is not available for this analysis.
      </p>
    );
  }

  const start = Math.min(...phases.map((p) => p.start_timestamp));
  const end = Math.max(...phases.map((p) => p.end_timestamp));
  const span = Math.max(end - start, 0.001);

  const active = phases.find(
    (p) => currentTime >= p.start_timestamp && currentTime <= p.end_timestamp + 0.02
  );

  return (
    <div className="space-y-3">
      <div className="flex items-end justify-between gap-3">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-[var(--muted)]">
          Stroke timeline
        </h2>
        <p className="text-xs text-[var(--muted)]" aria-live="polite">
          {active ? `Active: ${active.label}` : "—"} · {formatTimestamp(currentTime)}
        </p>
      </div>

      <div
        className="relative h-14 w-full rounded-md border border-[var(--border)] bg-[var(--timeline-track)]"
        role="list"
        aria-label="Stroke phases"
      >
        {phases.map((phase) => {
          const left = ((phase.start_timestamp - start) / span) * 100;
          const width = Math.max(
            ((phase.end_timestamp - phase.start_timestamp) / span) * 100,
            phase.is_contact_event ? 1.2 : 2
          );
          const isActive = active?.id === phase.id;
          return (
            <button
              key={`${phase.id}-${phase.start_timestamp}`}
              type="button"
              role="listitem"
              aria-label={`${phase.label}, ${formatTimestamp(phase.start_timestamp)} to ${formatTimestamp(phase.end_timestamp)}, confidence ${Math.round(phase.confidence * 100)} percent`}
              aria-current={isActive ? "true" : undefined}
              className={`absolute top-0 flex h-full flex-col items-center justify-center border-r border-white/40 px-1 text-[10px] font-semibold uppercase tracking-wide transition sm:text-xs ${
                isActive
                  ? "bg-[var(--timeline-active)] text-white"
                  : "bg-transparent text-[var(--fg)] hover:bg-black/5"
              } ${phase.is_contact_event ? "z-10" : ""}`}
              style={{ left: `${left}%`, width: `${width}%` }}
              onClick={() => onSeek(phase.seek_timestamp)}
            >
              <span className="truncate px-0.5">
                {phase.is_contact_event ? "●" : phase.label.split(" ")[0]}
              </span>
            </button>
          );
        })}

        {/* Playhead */}
        <div
          className="pointer-events-none absolute top-0 z-20 h-full w-0.5 bg-[var(--fg)]"
          style={{
            left: `${((currentTime - start) / span) * 100}%`,
          }}
          aria-hidden
        />

        {contactTimestamp != null && (
          <button
            type="button"
            aria-label={`Seek to contact at ${formatTimestamp(contactTimestamp)}`}
            className="absolute top-[-6px] z-30 h-3 w-3 -translate-x-1/2 rounded-full border-2 border-[var(--bg-elevated)] bg-[var(--accent)]"
            style={{
              left: `${((contactTimestamp - start) / span) * 100}%`,
            }}
            onClick={() => onSeek(contactTimestamp)}
          />
        )}
      </div>

      <div className="flex flex-wrap gap-2">
        {phases.map((phase) => (
          <button
            key={`chip-${phase.id}-${phase.start_timestamp}`}
            type="button"
            onClick={() => onSeek(phase.seek_timestamp)}
            className={`rounded-md border px-2.5 py-1.5 text-xs font-medium ${
              active?.id === phase.id
                ? "border-[var(--accent)] bg-[var(--accent-soft)] text-[var(--accent)]"
                : "border-[var(--border)] bg-[var(--bg-elevated)] text-[var(--muted)]"
            }`}
          >
            {phase.label}
            <span className="ml-1 opacity-70">
              {formatTimestamp(phase.start_timestamp)}
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}
