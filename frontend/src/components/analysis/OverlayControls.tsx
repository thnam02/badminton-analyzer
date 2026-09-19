"use client";

type OverlayMode = {
  id: string;
  label: string;
  available: boolean;
  toggleable?: boolean;
  note?: string;
};

type Props = {
  modes: OverlayMode[];
};

export function OverlayControls({ modes }: Props) {
  if (!modes.length) return null;

  return (
    <fieldset className="rounded-lg border border-[var(--border)] bg-[var(--bg-elevated)] p-4">
      <legend className="px-1 text-sm font-semibold">Overlays</legend>
      <ul className="mt-2 space-y-2">
        {modes.map((mode) => (
          <li key={mode.id} className="flex items-start gap-3 text-sm">
            <input
              type="checkbox"
              checked={mode.available}
              disabled={!mode.toggleable}
              readOnly={!mode.toggleable}
              className="mt-1"
              aria-describedby={`overlay-note-${mode.id}`}
            />
            <div>
              <span className="font-medium">{mode.label}</span>
              {mode.note && (
                <p
                  id={`overlay-note-${mode.id}`}
                  className="text-xs text-[var(--muted)]"
                >
                  {mode.note}
                </p>
              )}
              {!mode.toggleable && (
                <p className="text-xs text-[var(--muted)]">
                  Baked into the analysis video (not independently toggleable).
                </p>
              )}
            </div>
          </li>
        ))}
      </ul>
      <details className="mt-3 text-xs text-[var(--muted)]">
        <summary className="cursor-pointer">Developer / Experimental</summary>
        <p className="mt-2">
          3D mesh (WHAM) is hidden from the normal product experience and is not
          shown here.
        </p>
      </details>
    </fieldset>
  );
}
