import Link from "next/link";

export default function HomePage() {
  return (
    <main className="mx-auto flex min-h-[70vh] w-full max-w-5xl flex-col justify-center gap-10 px-4 py-16 sm:px-6">
      <div className="max-w-2xl space-y-5">
        <p className="text-xs font-semibold uppercase tracking-[0.16em] text-[var(--muted)]">
          Badminton technique analysis
        </p>
        <h1 className="font-display text-4xl font-semibold tracking-tight text-[var(--fg)] sm:text-5xl">
          Smash Lab
        </h1>
        <p className="text-lg leading-relaxed text-[var(--muted)]">
          Upload a forehand smash or clear, see stroke phases and contact, review
          evidence-backed findings against a reference group, and get one clear
          coaching focus for your next session.
        </p>
        <div className="flex flex-wrap gap-3 pt-2">
          <Link
            href="/analyze"
            className="inline-flex items-center justify-center rounded-md bg-[var(--accent)] px-5 py-2.5 text-sm font-semibold text-white"
          >
            Analyze a stroke
          </Link>
          <Link
            href="/history"
            className="inline-flex items-center justify-center rounded-md border border-[var(--border)] bg-[var(--bg-elevated)] px-5 py-2.5 text-sm font-semibold"
          >
            View history
          </Link>
        </div>
      </div>

      <ul className="grid gap-4 text-sm text-[var(--muted)] sm:grid-cols-3">
        <li className="rounded-lg border border-[var(--border)] bg-[var(--bg-elevated)] p-4">
          Interactive stroke timeline tied to your video
        </li>
        <li className="rounded-lg border border-[var(--border)] bg-[var(--bg-elevated)] p-4">
          Reference-group comparison — not a fake technique score
        </li>
        <li className="rounded-lg border border-[var(--border)] bg-[var(--bg-elevated)] p-4">
          Coaching that cites measurements you can verify
        </li>
      </ul>
    </main>
  );
}
