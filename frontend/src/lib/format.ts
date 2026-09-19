import type { ConfidenceLevel } from "./types";

export function formatNumber(
  value: number | null | undefined,
  digits = 1
): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return Number(value).toFixed(digits);
}

export function formatMetric(
  value: number | null | undefined,
  unit?: string,
  digits = 1
): string {
  const n = formatNumber(value, digits);
  if (n === "—") return n;
  return unit ? `${n}${unit === "deg" ? "°" : unit ? ` ${unit}` : ""}` : n;
}

export function formatPercentile(
  value: number | null | undefined
): string | null {
  if (value === null || value === undefined || Number.isNaN(value)) return null;
  const n = Math.round(Number(value));
  const suf =
    n % 10 === 1 && n % 100 !== 11
      ? "st"
      : n % 10 === 2 && n % 100 !== 12
        ? "nd"
        : n % 10 === 3 && n % 100 !== 13
          ? "rd"
          : "th";
  return `${n}${suf} percentile`;
}

export function formatTimestamp(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || Number.isNaN(seconds)) {
    return "—";
  }
  const s = Math.max(0, Number(seconds));
  const m = Math.floor(s / 60);
  const rem = s - m * 60;
  return `${m}:${rem.toFixed(2).padStart(5, "0")}`;
}

export function formatDate(iso?: string | null): string {
  if (!iso) return "Unknown date";
  try {
    return new Intl.DateTimeFormat(undefined, {
      dateStyle: "medium",
      timeStyle: "short",
    }).format(new Date(iso));
  } catch {
    return iso;
  }
}

export function confidenceLabel(level: ConfidenceLevel | string): string {
  switch (String(level).toUpperCase()) {
    case "HIGH":
      return "High";
    case "MODERATE":
      return "Moderate";
    case "LOW":
      return "Low";
    default:
      return String(level);
  }
}

export function severityLabel(statusOrSeverity?: string | null): string {
  const s = String(statusOrSeverity || "").toUpperCase();
  if (s === "MAJOR" || s === "HIGH") return "Major";
  if (s === "MODERATE" || s === "MEDIUM") return "Moderate";
  if (s === "MINOR" || s === "LOW") return "Minor";
  if (s === "INSUFFICIENT_EVIDENCE") return "Unable to assess";
  return s || "Finding";
}

export function issueTitleFromCode(code?: string | null): string {
  // Lazy import avoided — keep titles in sync via techniqueCopy.
  const titles: Record<string, string> = {
    INSUFFICIENT_ELBOW_EXTENSION: "Extend more at contact",
    LOW_KNEE_CONTRIBUTION: "Use more leg drive",
    PREPARATION_KNEE_OUT_OF_RANGE: "Adjust your ready bend",
    POOR_ARM_ACCELERATION_TIMING: "Your arm sequence is out of sync",
    LOW_CONTACT_POSTURE: "Contact point is too low",
    WEAK_FOLLOW_THROUGH: "Finish the swing more fully",
    LIMITED_CLEAR_PREPARATION: "Preparation is too compact",
    INSUFFICIENT_ARM_EXTENSION: "Extend more at contact",
    POOR_PROXIMAL_DISTAL_TIMING: "Your arm sequence is out of sync",
    RESTRICTED_FOLLOW_THROUGH: "Finish the clear more freely",
    SLOW_RECOVERY: "Get ready for the next shot sooner",
    INSUFFICIENT_EVIDENCE: "Unable to assess reliably",
  };
  if (!code) return "Finding";
  if (titles[code]) return titles[code];
  return code
    .toLowerCase()
    .split("_")
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");
}
