"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { AnalysisHeader } from "@/components/analysis/AnalysisHeader";
import { CoachingFocus } from "@/components/analysis/CoachingFocus";
import { ConfidenceSummaryPanel } from "@/components/analysis/ConfidenceSummary";
import { DrillCard } from "@/components/analysis/DrillCard";
import { MetricsPanel } from "@/components/analysis/MetricsPanel";
import { OverlayControls } from "@/components/analysis/OverlayControls";
import { PhaseTimeline } from "@/components/analysis/PhaseTimeline";
import { ReferenceComparison } from "@/components/analysis/ReferenceComparison";
import { TechniqueIssueCard } from "@/components/analysis/TechniqueIssueCard";
import {
  VideoPlayer,
  type VideoPlayerHandle,
} from "@/components/analysis/VideoPlayer";
import { fetchAnalysis, mediaUrl } from "@/lib/api";
import {
  estimateFpsFromPhases,
  splitMainAndOther,
} from "@/lib/techniqueCopy";
import type { AnalysisResult, TechniqueIssueView } from "@/lib/types";

type Props = {
  analysisId: string;
};

export function AnalysisResultView({ analysisId }: Props) {
  const [data, setData] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [currentTime, setCurrentTime] = useState(0);
  const [activeIssue, setActiveIssue] = useState<string | null>(null);
  const [techOpen, setTechOpen] = useState(false);
  const videoRef = useRef<VideoPlayerHandle>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    fetchAnalysis(analysisId)
      .then((result) => {
        if (!cancelled) setData(result);
      })
      .catch((err) => {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Could not load analysis");
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [analysisId]);

  const seek = useCallback((seconds: number, pause = true) => {
    videoRef.current?.seekTo(seconds, { pause });
    setCurrentTime(seconds);
  }, []);

  const onShowIssue = useCallback(
    (issue: TechniqueIssueView) => {
      setActiveIssue(issue.code);
      if (issue.seek_timestamp != null) {
        seek(issue.seek_timestamp, true);
      }
    },
    [seek]
  );

  if (loading) {
    return (
      <p className="text-sm text-[var(--muted)]" role="status">
        Loading analysis…
      </p>
    );
  }

  if (error || !data) {
    return (
      <div className="rounded-lg border border-[var(--border)] bg-[var(--bg-elevated)] p-6" role="alert">
        <h1 className="text-lg font-semibold">Analysis unavailable</h1>
        <p className="mt-2 text-sm text-[var(--muted)]">
          {error || "This analysis could not be found."}
        </p>
      </div>
    );
  }

  const videoSrc = mediaUrl(data.video.pose_video_url);
  const findings = data.findings || [];
  const insufficient = data.insufficient_evidence || [];

  return (
    <div className="mx-auto flex w-full max-w-6xl flex-col gap-8 px-4 py-8 sm:px-6">
      <AnalysisHeader analysis={data} />

      <section className="space-y-4">
        <VideoPlayer
          ref={videoRef}
          src={videoSrc}
          unavailableMessage="We could not load the annotated analysis video."
          onTimeUpdate={setCurrentTime}
        />
        <OverlayControls modes={data.video.overlay_modes || []} />
        <PhaseTimeline
          phases={data.phases}
          currentTime={currentTime}
          onSeek={(t) => seek(t, true)}
          contactTimestamp={data.contact.timestamp}
        />
        {data.contact.notes && (
          <p className="text-sm text-[var(--muted)]">{data.contact.notes}</p>
        )}
      </section>

      <CoachingFocus
        coaching={data.coaching}
        onSeek={(t) => seek(t, true)}
      />

      <TechniqueFindingsSection
        findings={findings}
        insufficient={insufficient}
        activeIssue={activeIssue}
        fps={estimateFpsFromPhases(data.phases)}
        onShowMoment={onShowIssue}
      />

      {data.reference_comparisons.length > 0 && (
        <section className="space-y-3">
          <h2 className="text-sm font-semibold uppercase tracking-wide text-[var(--muted)]">
            Reference comparison
          </h2>
          <div className="grid gap-3 lg:grid-cols-2">
            {data.reference_comparisons.map((ref) => (
              <ReferenceComparison
                key={`${ref.metric_id}-${ref.title}`}
                data={ref}
                profileHint={data.reference_profile_id}
              />
            ))}
          </div>
        </section>
      )}

      <section className="space-y-3">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-[var(--muted)]">
          Metrics
        </h2>
        <MetricsPanel metrics={data.metrics} />
      </section>

      {data.coaching.drills[0] && <DrillCard drill={data.coaching.drills[0]} />}

      <ConfidenceSummaryPanel
        confidence={data.confidence}
        limitations={data.limitations}
      />

      <details
        className="rounded-lg border border-[var(--border)] bg-[var(--bg-elevated)] p-4"
        open={techOpen}
        onToggle={(e) => setTechOpen((e.target as HTMLDetailsElement).open)}
      >
        <summary className="cursor-pointer text-sm font-semibold">
          Technical details
        </summary>
        <dl className="mt-3 grid gap-2 text-sm text-[var(--muted)] sm:grid-cols-2">
          <div>
            <dt>Analysis ID</dt>
            <dd className="font-mono text-xs text-[var(--fg)]">{data.analysis_id}</dd>
          </div>
          <div>
            <dt>Reference profile</dt>
            <dd className="font-mono text-xs text-[var(--fg)]">
              {data.reference_profile_id || "—"}
            </dd>
          </div>
          <div>
            <dt>Analysis status</dt>
            <dd>{data.analysis_status}</dd>
          </div>
          <div>
            <dt>Coaching status</dt>
            <dd>{data.coaching_status}</dd>
          </div>
          <div>
            <dt>Contact</dt>
            <dd>{data.contact.label}</dd>
          </div>
        </dl>
      </details>
    </div>
  );
}

function TechniqueFindingsSection({
  findings,
  insufficient,
  activeIssue,
  fps,
  onShowMoment,
}: {
  findings: TechniqueIssueView[];
  insufficient: TechniqueIssueView[];
  activeIssue: string | null;
  fps: number | null;
  onShowMoment: (issue: TechniqueIssueView) => void;
}) {
  const { main, other } = splitMainAndOther(findings);

  if (!main && other.length === 0 && insufficient.length === 0) {
    return (
      <section className="space-y-3">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-[var(--muted)]">
          Technique findings
        </h2>
        <p className="text-sm text-[var(--muted)]">
          No technique findings for this stroke.
        </p>
      </section>
    );
  }

  return (
    <section className="space-y-6">
      {main && (
        <div className="space-y-3">
          <h2 className="text-sm font-semibold uppercase tracking-wide text-[var(--muted)]">
            Main focus
          </h2>
          <TechniqueIssueCard
            issue={main}
            variant="main"
            active={activeIssue === main.code}
            fps={fps}
            onShowMoment={onShowMoment}
          />
        </div>
      )}

      {other.length > 0 && (
        <div className="space-y-3">
          <h2 className="text-sm font-semibold uppercase tracking-wide text-[var(--muted)]">
            Other findings
          </h2>
          <div className="grid gap-3 lg:grid-cols-2">
            {other.map((issue) => (
              <TechniqueIssueCard
                key={issue.code}
                issue={issue}
                variant="compact"
                active={activeIssue === issue.code}
                fps={fps}
                onShowMoment={onShowMoment}
              />
            ))}
          </div>
        </div>
      )}

      {insufficient.length > 0 && (
        <div className="space-y-3">
          <h2 className="text-sm font-semibold uppercase tracking-wide text-[var(--muted)]">
            Unable to assess reliably
          </h2>
          <p className="text-sm text-[var(--muted)]">
            These are confidence limits, not technique faults.
          </p>
          <div className="grid gap-3 lg:grid-cols-2">
            {insufficient.map((issue) => (
              <TechniqueIssueCard
                key={`ie-${issue.code}`}
                issue={issue}
                variant="insufficient"
                active={activeIssue === issue.code}
              />
            ))}
          </div>
        </div>
      )}
    </section>
  );
}
