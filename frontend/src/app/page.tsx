"use client";

import { useMemo, useRef, useState } from "react";

const API_BASE =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") || "http://localhost:8000";

const ACCEPTED = ".mp4,.mov,video/mp4,video/quicktime";

type AnalyzeResponse = {
  video_url: string;
  output_path: string;
};

function filenameFromUrl(url: string): string {
  try {
    const path = new URL(url, "http://localhost").pathname;
    const name = path.split("/").pop();
    return name && name.endsWith(".mp4") ? name : "processed_pose.mp4";
  } catch {
    return "processed_pose.mp4";
  }
}

export default function Home() {
  const inputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [resultUrl, setResultUrl] = useState<string | null>(null);
  const [resultFilename, setResultFilename] = useState("processed_pose.mp4");
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const canAnalyze = useMemo(() => Boolean(file) && !loading, [file, loading]);

  function onFileChange(selected: File | null) {
    setError(null);
    setResultUrl(null);
    setResultFilename("processed_pose.mp4");
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    if (!selected) {
      setFile(null);
      setPreviewUrl(null);
      return;
    }
    const lower = selected.name.toLowerCase();
    if (!lower.endsWith(".mp4") && !lower.endsWith(".mov")) {
      setError("Please choose an .mp4 or .mov video.");
      setFile(null);
      setPreviewUrl(null);
      return;
    }
    setFile(selected);
    setPreviewUrl(URL.createObjectURL(selected));
  }

  async function analyze() {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResultUrl(null);

    const form = new FormData();
    form.append("video", file);

    try {
      const res = await fetch(`${API_BASE}/analyze`, {
        method: "POST",
        body: form,
      });
      if (!res.ok) {
        let detail = `Request failed (${res.status})`;
        try {
          const body = await res.json();
          if (typeof body?.detail === "string") detail = body.detail;
        } catch {
          /* ignore */
        }
        throw new Error(detail);
      }
      const data = (await res.json()) as AnalyzeResponse;
      const url = `${API_BASE}${data.video_url}`;
      setResultUrl(url);
      setResultFilename(filenameFromUrl(url));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Analysis failed");
    } finally {
      setLoading(false);
    }
  }

  async function saveProcessedVideo() {
    if (!resultUrl) return;
    setSaving(true);
    setError(null);
    try {
      const res = await fetch(resultUrl);
      if (!res.ok) {
        throw new Error(`Could not download video (${res.status})`);
      }
      const blob = await res.blob();
      const objectUrl = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = objectUrl;
      anchor.download = resultFilename;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      URL.revokeObjectURL(objectUrl);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Save failed");
    } finally {
      setSaving(false);
    }
  }

  return (
    <main className="mx-auto flex min-h-screen w-full max-w-3xl flex-col gap-8 px-6 py-12">
      <header className="space-y-2">
        <h1 className="text-3xl font-semibold tracking-tight text-[var(--fg)]">
          Badminton Pose Analyzer
        </h1>
        <p className="text-[var(--muted)]">
          Upload a clip, run RTMPose, and preview the skeleton overlay.
        </p>
      </header>

      <section className="space-y-4 rounded-xl border border-[var(--border)] bg-[var(--bg-elevated)] p-6">
        <label className="block text-sm font-medium text-[var(--muted)]">
          Video (.mp4 / .mov)
        </label>
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED}
          className="block w-full text-sm text-[var(--fg)] file:mr-4 file:rounded-md file:border-0 file:bg-[var(--accent)] file:px-4 file:py-2 file:text-sm file:font-semibold file:text-[var(--bg)] hover:file:bg-[var(--accent-dim)]"
          onChange={(e) => onFileChange(e.target.files?.[0] ?? null)}
        />

        {previewUrl && (
          <div className="space-y-2">
            <p className="text-sm text-[var(--muted)]">Preview</p>
            <video
              src={previewUrl}
              controls
              className="w-full rounded-lg border border-[var(--border)] bg-black"
            />
          </div>
        )}

        <button
          type="button"
          disabled={!canAnalyze}
          onClick={analyze}
          className="inline-flex items-center justify-center rounded-md bg-[var(--accent)] px-5 py-2.5 text-sm font-semibold text-[var(--bg)] transition enabled:hover:bg-[var(--accent-dim)] disabled:cursor-not-allowed disabled:opacity-40"
        >
          {loading ? "Analyzing…" : "Analyze Video"}
        </button>

        {loading && (
          <p className="text-sm text-[var(--muted)]">
            Running pose estimation on every frame. This can take a while on CPU.
          </p>
        )}

        {error && (
          <p className="text-sm text-[var(--danger)]" role="alert">
            {error}
          </p>
        )}
      </section>

      {resultUrl && (
        <section className="space-y-3 rounded-xl border border-[var(--border)] bg-[var(--bg-elevated)] p-6">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <h2 className="text-lg font-medium">Processed video</h2>
            <button
              type="button"
              onClick={saveProcessedVideo}
              disabled={saving}
              className="inline-flex items-center justify-center rounded-md border border-[var(--border)] bg-transparent px-4 py-2 text-sm font-semibold text-[var(--fg)] transition hover:border-[var(--accent)] hover:text-[var(--accent)] disabled:cursor-not-allowed disabled:opacity-40"
            >
              {saving ? "Saving…" : "Save video"}
            </button>
          </div>
          <video
            src={resultUrl}
            controls
            autoPlay
            className="w-full rounded-lg border border-[var(--border)] bg-black"
          />
        </section>
      )}
    </main>
  );
}
