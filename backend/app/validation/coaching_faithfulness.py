"""Evidence faithfulness checks for CoachingReport text vs EvidencePackage."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from app.schemas.contact import CONTACT_TYPE_KINEMATIC, CONTACT_TYPE_TRACKED
from app.schemas.evidence import EvidencePackage
from app.schemas.phases import SmashPhase

# Degrees / unit / bare measurement mentions commonly quoted in coaching text.
_NUMBER_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"("
    r"\d+\.\d+|\d+"
    r")"
    r"(?:\s*(°|deg(?:rees)?|ms|s\b|%))?",
    re.IGNORECASE,
)
_CONFIDENCE_PATTERN = re.compile(
    r"\bconfidence(?:\s+(?:of|is|=|:))?\s*(0?\.\d+|1(?:\.0+)?|\d{1,3}\s*%)",
    re.IGNORECASE,
)
_FRAME_PATTERN = re.compile(
    r"\b(?:frame(?:\s*index)?|at\s+frame)\s*(?:=|:)?\s*(\d+)\b",
    re.IGNORECASE,
)
_ISSUE_CODE_PATTERN = re.compile(r"\b([A-Z][A-Z0-9]+(?:_[A-Z0-9]+)+)\b")
_PHASE_NAMES = {p.value for p in SmashPhase}
_CONTACT_TYPES = {CONTACT_TYPE_KINEMATIC, CONTACT_TYPE_TRACKED, "ESTIMATED"}


@dataclass(slots=True)
class EvidenceIndex:
    """Flattened, comparable facts from an EvidencePackage."""

    issue_codes: set[str] = field(default_factory=set)
    phases: set[str] = field(default_factory=set)
    metric_keys: set[str] = field(default_factory=set)
    contact_types: set[str] = field(default_factory=set)
    numeric_values: list[float] = field(default_factory=list)
    frame_indices: set[int] = field(default_factory=set)
    confidences: list[float] = field(default_factory=list)

    def supports_number(
        self,
        value: float,
        *,
        abs_tol: float = 0.06,
        rel_tol: float = 0.02,
    ) -> bool:
        for candidate in self.numeric_values:
            if math.isclose(value, candidate, rel_tol=rel_tol, abs_tol=abs_tol):
                return True
            # Percent form of a 0–1 confidence (e.g. 0.8 ↔ 80).
            if 0.0 <= candidate <= 1.0 and math.isclose(
                value, candidate * 100.0, rel_tol=rel_tol, abs_tol=abs_tol
            ):
                return True
            if 0.0 <= value <= 1.0 and math.isclose(
                value * 100.0, candidate, rel_tol=rel_tol, abs_tol=abs_tol
            ):
                return True
        return False

    def supports_frame(self, frame: int) -> bool:
        return int(frame) in self.frame_indices

    def supports_confidence(
        self,
        value: float,
        *,
        abs_tol: float = 0.06,
    ) -> bool:
        normalized = value / 100.0 if value > 1.0 else value
        for candidate in self.confidences:
            if math.isclose(normalized, candidate, abs_tol=abs_tol, rel_tol=0.0):
                return True
        return False


@dataclass(slots=True)
class FaithfulnessFinding:
    kind: str
    detail: str
    source_field: str
    quoted_value: str
    supported: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "detail": self.detail,
            "source_field": self.source_field,
            "quoted_value": self.quoted_value,
            "supported": self.supported,
        }


def build_evidence_index(evidence: EvidencePackage) -> EvidenceIndex:
    index = EvidenceIndex()
    index.metric_keys = {str(k) for k in (evidence.metrics or {}).keys()}
    _collect_numbers(index, evidence.metrics)

    for issue in evidence.technique_issues or []:
        code = issue.get("code")
        if code:
            index.issue_codes.add(str(code))
        phase = issue.get("phase")
        if phase:
            index.phases.add(str(phase))
        if issue.get("measured_value") is not None:
            _add_number(index, issue["measured_value"])
        if issue.get("confidence") is not None:
            conf = float(issue["confidence"])
            index.confidences.append(conf)
            _add_number(index, conf)
        ref = issue.get("reference_range") or {}
        if isinstance(ref, dict):
            for key in ("min", "max"):
                if ref.get(key) is not None:
                    _add_number(index, ref[key])

    for seg in evidence.phase_boundaries or []:
        phase = seg.get("phase")
        if phase:
            index.phases.add(str(phase))
        for key in (
            "start_frame_index",
            "end_frame_index",
            "frame_index",
        ):
            if seg.get(key) is not None:
                index.frame_indices.add(int(seg[key]))
                _add_number(index, seg[key])
        for key in ("start_timestamp", "end_timestamp", "timestamp"):
            if seg.get(key) is not None:
                _add_number(index, seg[key])
        if seg.get("confidence") is not None:
            conf = float(seg["confidence"])
            index.confidences.append(conf)
            _add_number(index, conf)

    contact = evidence.contact
    if contact.contact_type:
        index.contact_types.add(str(contact.contact_type))
    if contact.frame_index is not None:
        index.frame_indices.add(int(contact.frame_index))
        _add_number(index, contact.frame_index)
    if contact.kinematic_frame_index is not None:
        index.frame_indices.add(int(contact.kinematic_frame_index))
        _add_number(index, contact.kinematic_frame_index)
    if contact.timestamp is not None:
        _add_number(index, contact.timestamp)
    index.confidences.append(float(contact.confidence))
    _add_number(index, contact.confidence)

    for conf in (
        evidence.analysis_confidence,
        evidence.phase_confidence,
        evidence.technique_confidence,
    ):
        index.confidences.append(float(conf))
        _add_number(index, conf)

    for kf in evidence.keyframes or []:
        if kf.get("phase"):
            index.phases.add(str(kf["phase"]))
        if kf.get("frame_index") is not None:
            index.frame_indices.add(int(kf["frame_index"]))
            _add_number(index, kf["frame_index"])
        if kf.get("timestamp") is not None:
            _add_number(index, kf["timestamp"])
        if kf.get("confidence") is not None:
            conf = float(kf["confidence"])
            index.confidences.append(conf)
            _add_number(index, conf)

    # Always allow canonical phase vocabulary when phases exist in evidence.
    if index.phases:
        index.phases |= set(_PHASE_NAMES) & index.phases
    return index


def check_text_faithfulness(
    texts: list[tuple[str, str]],
    index: EvidenceIndex,
    *,
    known_issue_codes: set[str] | None = None,
) -> list[FaithfulnessFinding]:
    """Extract quoted facts from report text and verify against evidence."""
    findings: list[FaithfulnessFinding] = []
    known_codes = known_issue_codes or set()

    for source_field, text in texts:
        if not text:
            continue

        for match in _ISSUE_CODE_PATTERN.finditer(text):
            code = match.group(1)
            # Skip contact-type tokens and phase names mistaken as codes.
            if code in _PHASE_NAMES or code in _CONTACT_TYPES:
                continue
            if code in index.contact_types:
                continue
            # Only score tokens that look like technique issues we know, or
            # that appear in evidence; ignore random ALL_CAPS identifiers.
            if code not in index.issue_codes and code not in known_codes:
                # Still flag if it looks like a technique issue (has _ and
                # matches production vocabulary prefixes).
                if not _looks_like_issue_code(code):
                    continue
                findings.append(
                    FaithfulnessFinding(
                        kind="issue_code",
                        detail="Issue code not present in EvidencePackage",
                        source_field=source_field,
                        quoted_value=code,
                        supported=False,
                    )
                )
            elif code not in index.issue_codes:
                findings.append(
                    FaithfulnessFinding(
                        kind="issue_code",
                        detail="Issue code not present in EvidencePackage",
                        source_field=source_field,
                        quoted_value=code,
                        supported=False,
                    )
                )
            else:
                findings.append(
                    FaithfulnessFinding(
                        kind="issue_code",
                        detail="Issue code supported by EvidencePackage",
                        source_field=source_field,
                        quoted_value=code,
                        supported=True,
                    )
                )

        for phase in _PHASE_NAMES:
            if re.search(rf"\b{re.escape(phase)}\b", text):
                # If evidence lists phases, require membership; otherwise allow.
                supported = (not index.phases) or (phase in index.phases)
                findings.append(
                    FaithfulnessFinding(
                        kind="phase",
                        detail=(
                            "Phase present in evidence"
                            if supported
                            else "Phase not present in EvidencePackage phases"
                        ),
                        source_field=source_field,
                        quoted_value=phase,
                        supported=supported,
                    )
                )

        for contact_type in (CONTACT_TYPE_TRACKED, CONTACT_TYPE_KINEMATIC):
            if contact_type in text:
                supported = contact_type in index.contact_types
                findings.append(
                    FaithfulnessFinding(
                        kind="contact_type",
                        detail=(
                            "Contact type matches evidence"
                            if supported
                            else "Contact type contradicts EvidencePackage"
                        ),
                        source_field=source_field,
                        quoted_value=contact_type,
                        supported=supported,
                    )
                )

        for match in _CONFIDENCE_PATTERN.finditer(text):
            raw = match.group(1).strip()
            value = _parse_percent_or_float(raw)
            supported = index.supports_confidence(value)
            findings.append(
                FaithfulnessFinding(
                    kind="confidence",
                    detail=(
                        "Confidence value supported by evidence"
                        if supported
                        else "Confidence value not found in EvidencePackage"
                    ),
                    source_field=source_field,
                    quoted_value=raw,
                    supported=supported,
                )
            )

        for match in _FRAME_PATTERN.finditer(text):
            frame = int(match.group(1))
            supported = index.supports_frame(frame)
            findings.append(
                FaithfulnessFinding(
                    kind="contact_frame",
                    detail=(
                        "Frame index supported by evidence"
                        if supported
                        else "Frame index not found in EvidencePackage"
                    ),
                    source_field=source_field,
                    quoted_value=str(frame),
                    supported=supported,
                )
            )

        for match in _NUMBER_PATTERN.finditer(text):
            # Skip numbers already captured as confidence/frame patterns.
            start = match.start()
            window = text[max(0, start - 24) : match.end() + 8].lower()
            if "confidence" in window or "frame" in window:
                continue
            raw_num = match.group(1)
            unit = (match.group(2) or "").lower()
            value = float(raw_num)
            # Ignore tiny integers that are priorities / counts (1–3) without units.
            if not unit and value in {1.0, 2.0, 3.0} and "." not in raw_num:
                continue
            # Ignore years-like or very large bare ints unlikely to be metrics.
            if not unit and value >= 1000:
                continue
            supported = index.supports_number(value)
            # Percents map to confidence scale.
            if unit == "%" and not supported:
                supported = index.supports_confidence(value)
            findings.append(
                FaithfulnessFinding(
                    kind="measurement",
                    detail=(
                        "Measurement supported by EvidencePackage"
                        if supported
                        else "Invented or contradictory measurement vs EvidencePackage"
                    ),
                    source_field=source_field,
                    quoted_value=f"{raw_num}{(' ' + unit) if unit else ''}".strip(),
                    supported=supported,
                )
            )

    return findings


def check_structured_faithfulness(
    *,
    prioritized_issue_codes: list[str],
    metric_hints: list[str],
    drill_issue_codes: list[str],
    index: EvidenceIndex,
) -> list[FaithfulnessFinding]:
    findings: list[FaithfulnessFinding] = []
    for code in prioritized_issue_codes:
        supported = code in index.issue_codes
        findings.append(
            FaithfulnessFinding(
                kind="issue_code",
                detail=(
                    "Prioritized issue present in evidence"
                    if supported
                    else "Prioritized issue code not in EvidencePackage"
                ),
                source_field="prioritized_issues",
                quoted_value=code,
                supported=supported,
            )
        )
    for hint in metric_hints:
        # Allow dotted refs like metrics.peak_wrist_speed
        key = hint.split(".")[-1] if "." in hint else hint
        supported = key in index.metric_keys or hint in index.metric_keys
        findings.append(
            FaithfulnessFinding(
                kind="metric_hint",
                detail=(
                    "Metric hint present in evidence.metrics"
                    if supported
                    else "Metric hint not present in EvidencePackage.metrics"
                ),
                source_field="related_metric_hints",
                quoted_value=hint,
                supported=supported,
            )
        )
    for code in drill_issue_codes:
        supported = code in index.issue_codes
        findings.append(
            FaithfulnessFinding(
                kind="issue_code",
                detail=(
                    "Drill target issue present in evidence"
                    if supported
                    else "Drill targets unknown issue code"
                ),
                source_field="drills.targets_issue_codes",
                quoted_value=code,
                supported=supported,
            )
        )
    return findings


def _looks_like_issue_code(code: str) -> bool:
    prefixes = (
        "INSUFFICIENT_",
        "LOW_",
        "POOR_",
        "WEAK_",
        "HIGH_",
        "LATE_",
        "EARLY_",
        "PREPARATION_",
    )
    return any(code.startswith(p) for p in prefixes)


def _parse_percent_or_float(raw: str) -> float:
    cleaned = raw.strip().replace("%", "")
    value = float(cleaned)
    if "%" in raw:
        return value
    return value


def _add_number(index: EvidenceIndex, value: Any) -> None:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return
    if math.isnan(num) or math.isinf(num):
        return
    index.numeric_values.append(num)


def _collect_numbers(index: EvidenceIndex, payload: Any) -> None:
    if isinstance(payload, dict):
        for value in payload.values():
            _collect_numbers(index, value)
    elif isinstance(payload, (list, tuple)):
        for value in payload:
            _collect_numbers(index, value)
    else:
        _add_number(index, payload)
