/**
 * Player-facing copy for technique issues.
 * Presentation only — does not change detection, severity, or measurements.
 */

import type { TechniqueIssueView } from "./types";
import { formatMetric, formatNumber, severityLabel } from "./format";

export type IssueCoachingCopy = {
  title: string;
  whatHappened: string;
  whyItMatters: string;
  coachingFocus: string;
};

const ISSUE_COPY: Record<string, IssueCoachingCopy> = {
  INSUFFICIENT_ELBOW_EXTENSION: {
    title: "Extend more at contact",
    whatHappened:
      "Your hitting arm looked less extended than the reference group at the estimated contact moment.",
    whyItMatters:
      "A more extended arm at contact helps you hit through the shuttle with a longer lever and cleaner power transfer.",
    coachingFocus:
      "Reach tall through contact — finish the swing with a longer arm before you wrap.",
  },
  LOW_KNEE_CONTRIBUTION: {
    title: "Use more leg drive",
    whatHappened:
      "Your knees contributed less upward drive into the swing than the reference group.",
    whyItMatters:
      "Leg drive sets the rhythm for the rest of the stroke and helps you hit up through the shuttle, not just with the arm.",
    coachingFocus:
      "Load the legs in preparation, then drive up as the swing starts.",
  },
  PREPARATION_KNEE_OUT_OF_RANGE: {
    title: "Adjust your ready bend",
    whatHappened:
      "Your knee bend in preparation sat outside the usual range for this stroke.",
    whyItMatters:
      "A stable ready bend makes it easier to time the jump or push into the swing.",
    coachingFocus:
      "Settle into a comfortable athletic bend before the racket starts back.",
  },
  POOR_ARM_ACCELERATION_TIMING: {
    title: "Your arm sequence is out of sync",
    whatHappened:
      "The peak of your elbow speed arrived at a different moment relative to contact than the reference group.",
    whyItMatters:
      "When elbow and wrist peak in a smoother order, contact feels cleaner and less rushed.",
    coachingFocus:
      "Let the upper arm start the whip, then let the hand catch up through contact.",
  },
  LOW_CONTACT_POSTURE: {
    title: "Contact point is too low",
    whatHappened:
      "Your racket-side hand met the shuttle lower than the reference group’s typical contact height.",
    whyItMatters:
      "Contacting higher usually means better clearance and a stronger, more attacking geometry.",
    coachingFocus:
      "Get under the shuttle earlier so you can meet it higher in front of you.",
  },
  WEAK_FOLLOW_THROUGH: {
    title: "Finish the swing more fully",
    whatHappened:
      "Your racket hand slowed down sooner after contact than the reference group.",
    whyItMatters:
      "A fuller finish keeps you committed through the hit instead of cutting the stroke short.",
    coachingFocus:
      "Stay through the shuttle and let the swing unwind across the body.",
  },
  LIMITED_CLEAR_PREPARATION: {
    title: "Preparation is too compact",
    whatHappened:
      "Your clear setup looked tighter than the reference group — less room in the arm or legs before the swing.",
    whyItMatters:
      "A clearer preparation gives you space to accelerate upward and hit a higher, deeper clear.",
    coachingFocus:
      "Open the ready position earlier — create space with the racket arm before you accelerate.",
  },
  INSUFFICIENT_ARM_EXTENSION: {
    title: "Extend more at contact",
    whatHappened:
      "Your arm was less extended at estimated contact than players in the clear reference group.",
    whyItMatters:
      "Fuller extension at contact helps send the shuttle higher and deeper without muscling the wrist alone.",
    coachingFocus:
      "Reach long through contact as if you are throwing the shuttle to the back court.",
  },
  POOR_PROXIMAL_DISTAL_TIMING: {
    title: "Your arm sequence is out of sync",
    whatHappened:
      "The order of your elbow and wrist speed peaks did not match the usual clear timing pattern.",
    whyItMatters:
      "A smoother arm sequence helps you hit up through the shuttle instead of dumping it flat.",
    coachingFocus:
      "Start the whip from the elbow, then let the hand accelerate through a high contact.",
  },
  RESTRICTED_FOLLOW_THROUGH: {
    title: "Finish the clear more freely",
    whatHappened:
      "After contact, your racket hand did not carry through as freely as the reference group.",
    whyItMatters:
      "A freer finish supports an upward clear path and keeps you balanced for the next shot.",
    coachingFocus:
      "Stay tall through the hit and let the racket continue past your opposite hip.",
  },
  SLOW_RECOVERY: {
    title: "Get ready for the next shot sooner",
    whatHappened:
      "It took longer than usual for your motion to settle back toward a ready position after the clear.",
    whyItMatters:
      "Faster recovery keeps you available for the reply instead of stuck in the finish.",
    coachingFocus:
      "Land soft, bring the racket back to ready, and find your base quickly.",
  },
};

const SEVERITY_RANK: Record<string, number> = {
  MAJOR: 3,
  HIGH: 3,
  MODERATE: 2,
  MEDIUM: 2,
  MINOR: 1,
  LOW: 1,
};

export function playerIssueTitle(code?: string | null): string {
  if (!code) return "Finding";
  const mapped = ISSUE_COPY[code];
  if (mapped) return mapped.title;
  return code
    .toLowerCase()
    .split("_")
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");
}

export function coachingCopyForIssue(issue: TechniqueIssueView): IssueCoachingCopy {
  const mapped = ISSUE_COPY[issue.code];
  if (mapped) return mapped;
  return {
    title: playerIssueTitle(issue.code) || issue.title,
    whatHappened:
      sanitizeTechnicalLanguage(issue.explanation) ||
      "This part of the stroke differed from the reference group.",
    whyItMatters:
      "Addressing it should make the stroke feel more consistent and efficient.",
    coachingFocus: "Replay this moment and rehearse one small adjustment next session.",
  };
}

/** Strip internal jargon from backend descriptions if we fall back to them. */
export function sanitizeTechnicalLanguage(text?: string | null): string {
  if (!text) return "";
  return text
    .replace(/proximal[- ]to[- ]distal/gi, "arm sequence")
    .replace(/proximal[- ]distal/gi, "arm sequence")
    .replace(/normalized[_\s-]?y/gi, "height")
    .replace(/wrist[_\s-]?y/gi, "hand height")
    .replace(/ESTIMATED_CONTACT/g, "estimated contact")
    .replace(/peak_elbow_omega_offset_frames/gi, "elbow timing")
    .replace(/follow_through_speed_ratio/gi, "follow-through speed")
    .replace(/\b-?\d+\s*frames?\b/gi, (m) => {
      // Leave numeric frame phrases for evidence; strip from casual copy.
      return m.includes("-") ? "earlier than usual" : "later than usual";
    })
    .replace(/\s{2,}/g, " ")
    .trim();
}

export function estimateFpsFromPhases(
  phases: { start_timestamp: number; end_timestamp: number; start_frame_index?: number | null; end_frame_index?: number | null }[]
): number | null {
  for (const p of phases) {
    if (
      p.start_frame_index == null ||
      p.end_frame_index == null ||
      p.end_frame_index <= p.start_frame_index
    ) {
      continue;
    }
    const dt = p.end_timestamp - p.start_timestamp;
    if (dt <= 0) continue;
    const fps = (p.end_frame_index - p.start_frame_index) / dt;
    if (fps > 5 && fps < 240) return fps;
  }
  return null;
}

export function formatPlayerMeasurement(
  issue: TechniqueIssueView,
  fps?: number | null
): string {
  const value = issue.measured_value;
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "—";
  }
  const unit = String(issue.unit || "").toLowerCase();
  const code = issue.code;

  if (
    unit.includes("frame") ||
    code === "POOR_ARM_ACCELERATION_TIMING" ||
    code === "POOR_PROXIMAL_DISTAL_TIMING"
  ) {
    const frames = Number(value);
    if (fps && fps > 0) {
      const ms = Math.round((frames / fps) * 1000);
      if (ms === 0) return "roughly on time with contact";
      if (ms < 0) return `${Math.abs(ms)} ms before contact`;
      return `${ms} ms after contact`;
    }
    if (frames < 0) return "earlier than the reference timing";
    if (frames > 0) return "later than the reference timing";
    return "near contact timing";
  }

  if (
    unit.includes("normalized") ||
    code === "LOW_CONTACT_POSTURE" ||
    (unit === "" && code === "LOW_CONTACT_POSTURE")
  ) {
    const median = issue.reference_median;
    if (median != null && Number.isFinite(Number(median))) {
      // Image y increases downward — higher value ≈ lower contact.
      if (Number(value) > Number(median)) return "lower than the reference group";
      if (Number(value) < Number(median)) return "higher than the reference group";
      return "similar height to the reference group";
    }
    return "different contact height than the reference group";
  }

  if (unit.includes("ratio") || unit === "speed_ratio") {
    const pct = Math.round(Number(value) * 100);
    return `kept about ${pct}% of contact speed into the finish`;
  }

  if (unit === "deg" || unit === "degrees") {
    return formatMetric(value, "deg");
  }

  return formatMetric(value, issue.unit);
}

export function formatEvidenceValue(
  value: number | null | undefined,
  unit?: string,
  digits = 1
): string {
  return formatMetric(value, unit, digits);
}

export function formatConfidencePct(
  value: number | null | undefined
): string {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "—";
  }
  const n = Number(value);
  const pct = n <= 1 ? Math.round(n * 100) : Math.round(n);
  return `${pct}%`;
}

export function findingSeverityLabel(issue: TechniqueIssueView): string {
  return severityLabel(issue.status || issue.severity);
}

export function sortFindingsByPriority(
  findings: TechniqueIssueView[]
): TechniqueIssueView[] {
  return [...findings].sort((a, b) => {
    const sa = SEVERITY_RANK[String(a.status || a.severity || "").toUpperCase()] || 0;
    const sb = SEVERITY_RANK[String(b.status || b.severity || "").toUpperCase()] || 0;
    if (sb !== sa) return sb - sa;
    return (b.confidence || 0) - (a.confidence || 0);
  });
}

export function splitMainAndOther(
  findings: TechniqueIssueView[]
): { main: TechniqueIssueView | null; other: TechniqueIssueView[] } {
  const sorted = sortFindingsByPriority(findings);
  if (!sorted.length) return { main: null, other: [] };
  return { main: sorted[0], other: sorted.slice(1) };
}

export function phaseChipLabel(phaseLabel: string): string {
  return phaseLabel || "Stroke";
}

export { formatNumber };
