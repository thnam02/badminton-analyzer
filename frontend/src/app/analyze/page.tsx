"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useMemo, useRef, useState } from "react";
import { startAnalyze } from "@/lib/api";

const ACCEPTED = ".mp4,.mov,video/mp4,video/quicktime";

const STAGES = [
  { id: "prepare", label: "Preparing video" },
  { id: "pose", label: "Detecting body pose" },
  { id: "motion", label: "Analysing movement" },
  { id: "phases", label: "Finding stroke phases" },
  { id: "technique", label: "Evaluating technique" },
  { id: "coaching", label: "Generating coaching" },
] as const;

type StrokeOption = {
  id: string;
  label: string;
  available: boolean;
  apiValue?: string;
};

const STROKES: StrokeOption[] = [
  {
    id: "forehand_smash",
    label: "Forehand Smash",
    available: true,
    apiValue: "FOREHAND_SMASH",
  },
  {
    id: "forehand_clear",
    label: "Forehand Clear",
    available: true,
    apiValue: "FOREHAND_CLEAR",
  },
  { id: "drop", label: "Drop", available: false },
  { id: "serve", label: "Serve", available: false },
  { id: "net", label: "Net shot", available: false },
  { id: "lift", label: "Lift", available: false },
];

export default function AnalyzePage() {
  const router = useRouter();
  const inputRef = useRef<HTMLInputElement>(null);
  const [step, setStep] = useState(1);
  const [stroke, setStroke] = useState("forehand_smash");
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [meta, setMeta] = useState<{
    duration?: number;
    width?: number;
    height?: number;
  }>({});
  const [loading, setLoading] = useState(false);
  const [stageIndex, setStageIndex] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [analysisReadyId, setAnalysisReadyId] = useState<string | null>(null);
  const [coachingPending, setCoachingPending] = useState(false);

  const canUpload = Boolean(file);
  const fileSizeLabel = useMemo(() => {
    if (!file) return "";
    const mb = file.size / (1024 * 1024);
    return `${mb.toFixed(1)} MB`;
  }, [file]);

  function onFileChange(selected: File | null) {
    setError(null);
    setAnalysisReadyId(null);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    if (!selected) {
      setFile(null);
      setPreviewUrl(null);
      setMeta({});
      return;
    }
    const lower = selected.name.toLowerCase();
    if (!lower.endsWith(".mp4") && !lower.endsWith(".mov")) {
      setError("Please choose an .mp4 or .mov video.");
      return;
    }
    setFile(selected);
    const url = URL.createObjectURL(selected);
    setPreviewUrl(url);
    const video = document.createElement("video");
    video.preload = "metadata";
    video.src = url;
    video.onloadedmetadata = () => {
      setMeta({
        duration: video.duration,
        width: video.videoWidth,
        height: video.videoHeight,
      });
    };
  }

  async function runAnalyze() {
    if (!file) return;
    setLoading(true);
    setError(null);
    setStageIndex(0);
    setAnalysisReadyId(null);
    setCoachingPending(false);

    const timer = window.setInterval(() => {
      setStageIndex((i) => Math.min(i + 1, STAGES.length - 2));
    }, 1800);

    try {
      const selected = STROKES.find((s) => s.id === stroke);
      const data = await startAnalyze(file, {
        meshOverlay: false,
        strokeType: selected?.apiValue || "FOREHAND_SMASH",
      });
      window.clearInterval(timer);
      setStageIndex(STAGES.length - 1);
      const id = data.analysis_id;
      setAnalysisReadyId(id);
      const coachingDone =
        !data.coaching_status ||
        data.coaching_status === "COMPLETE" ||
        data.coaching_status === "ok";
      setCoachingPending(!coachingDone);
      // Deterministic analysis is ready — navigate even if coaching is soft-unavailable.
      router.push(`/analysis/${id}`);
    } catch (err) {
      window.clearInterval(timer);
      setError(err instanceof Error ? err.message : "Analysis failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="mx-auto flex w-full max-w-3xl flex-col gap-8 px-4 py-10 sm:px-6">
      <header className="space-y-2">
        <h1 className="font-display text-3xl font-semibold tracking-tight">
          Analyze a stroke
        </h1>
        <p className="text-[var(--muted)]">
          Choose forehand smash or forehand clear, then upload a clip for
          technique analysis.
        </p>
      </header>

      {/* Step 1 */}
      <section className="rounded-lg border border-[var(--border)] bg-[var(--bg-elevated)] p-5">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-[var(--muted)]">
          Step 1 · Stroke
        </h2>
        <div className="mt-4 grid gap-2 sm:grid-cols-2">
          {STROKES.map((s) => (
            <button
              key={s.id}
              type="button"
              disabled={!s.available}
              onClick={() => {
                setStroke(s.id);
                setStep(Math.max(step, 2));
              }}
              className={`rounded-md border px-4 py-3 text-left text-sm font-medium ${
                stroke === s.id
                  ? "border-[var(--accent)] bg-[var(--accent-soft)] text-[var(--accent)]"
                  : "border-[var(--border)]"
              } disabled:cursor-not-allowed disabled:opacity-50`}
            >
              {s.label}
              {!s.available && (
                <span className="mt-1 block text-xs text-[var(--muted)]">
                  Coming soon
                </span>
              )}
            </button>
          ))}
        </div>
      </section>

      {/* Step 2 */}
      <section className="rounded-lg border border-[var(--border)] bg-[var(--bg-elevated)] p-5">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-[var(--muted)]">
          Step 2 · Camera guidance
        </h2>
        <ul className="mt-4 space-y-2 text-sm text-[var(--fg)]">
          <li>✓ Keep your full body visible</li>
          <li>✓ Use a static camera</li>
          <li>✓ Rear-side or side 45° view</li>
          <li>✓ 60 FPS recommended</li>
          <li>✓ Good lighting</li>
        </ul>
        <button
          type="button"
          className="mt-4 text-sm font-medium text-[var(--accent)]"
          onClick={() => setStep(Math.max(step, 3))}
        >
          Continue to upload
        </button>
      </section>

      {/* Step 3 */}
      <section className="rounded-lg border border-[var(--border)] bg-[var(--bg-elevated)] p-5">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-[var(--muted)]">
          Step 3 · Upload
        </h2>
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED}
          className="mt-4 block w-full text-sm file:mr-4 file:rounded-md file:border-0 file:bg-[var(--accent)] file:px-4 file:py-2 file:text-sm file:font-semibold file:text-white"
          onChange={(e) => onFileChange(e.target.files?.[0] ?? null)}
        />
        {file && (
          <dl className="mt-4 grid gap-2 text-sm sm:grid-cols-2">
            <div>
              <dt className="text-[var(--muted)]">Filename</dt>
              <dd className="font-medium">{file.name}</dd>
            </div>
            <div>
              <dt className="text-[var(--muted)]">File size</dt>
              <dd className="font-medium">{fileSizeLabel}</dd>
            </div>
            <div>
              <dt className="text-[var(--muted)]">Duration</dt>
              <dd className="font-medium">
                {meta.duration ? `${meta.duration.toFixed(1)} s` : "—"}
              </dd>
            </div>
            <div>
              <dt className="text-[var(--muted)]">Resolution</dt>
              <dd className="font-medium">
                {meta.width && meta.height
                  ? `${meta.width}×${meta.height}`
                  : "—"}
              </dd>
            </div>
          </dl>
        )}
        {previewUrl && (
          <video
            src={previewUrl}
            controls
            className="mt-4 aspect-video w-full rounded-lg border border-[var(--border)] bg-black"
          />
        )}
      </section>

      {/* Step 4 */}
      <section className="rounded-lg border border-[var(--border)] bg-[var(--bg-elevated)] p-5">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-[var(--muted)]">
          Step 4 · Analyze
        </h2>
        <button
          type="button"
          disabled={!canUpload || loading}
          onClick={runAnalyze}
          className="mt-4 inline-flex items-center justify-center rounded-md bg-[var(--accent)] px-5 py-2.5 text-sm font-semibold text-white disabled:cursor-not-allowed disabled:opacity-40"
        >
          {loading ? "Analyzing…" : "Start analysis"}
        </button>

        {loading && (
          <ol className="mt-5 space-y-2" aria-live="polite">
            {STAGES.map((stage, idx) => {
              const done = idx < stageIndex;
              const current = idx === stageIndex;
              return (
                <li key={stage.id} className="flex items-center gap-3 text-sm">
                  <span aria-hidden className="w-4 text-center">
                    {done ? "✓" : current ? "●" : "○"}
                  </span>
                  <span
                    className={
                      current
                        ? "font-semibold text-[var(--fg)]"
                        : "text-[var(--muted)]"
                    }
                  >
                    {stage.label}
                  </span>
                </li>
              );
            })}
          </ol>
        )}

        {analysisReadyId && coachingPending && (
          <p className="mt-4 text-sm text-[var(--muted)]" role="status">
            Your movement analysis is ready.{" "}
            <Link
              href={`/analysis/${analysisReadyId}`}
              className="font-medium text-[var(--accent)]"
            >
              View results
            </Link>{" "}
            while coaching finishes.
          </p>
        )}

        {error && (
          <p className="mt-4 text-sm text-[var(--danger)]" role="alert">
            {error}
          </p>
        )}
      </section>
    </main>
  );
}
