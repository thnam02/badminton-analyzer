"""Versioned reference-data protocol and dataset metadata schemas.

Defines the controlled recording specification and typed metadata for collecting
forehand-smash reference footage. Does **not** compute reference ranges or
change technique evaluation rules.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Versions
# ---------------------------------------------------------------------------

REFERENCE_DATA_PROTOCOL_VERSION = "1.0.0"
REFERENCE_DATASET_SCHEMA_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Controlled vocabulary
# ---------------------------------------------------------------------------

STROKE_TYPE_FOREHAND_SMASH = "FOREHAND_SMASH"

CAMERA_VIEW_REAR_45 = "rear_45"
CAMERA_VIEW_SIDE_45 = "side_45"
PROTOCOL_CAMERA_VIEWS: tuple[str, ...] = (
    CAMERA_VIEW_REAR_45,
    CAMERA_VIEW_SIDE_45,
)

HANDEDNESS_LEFT = "LEFT"
HANDEDNESS_RIGHT = "RIGHT"
PROTOCOL_HANDEDNESS: tuple[str, ...] = (HANDEDNESS_LEFT, HANDEDNESS_RIGHT)

SKILL_BEGINNER = "beginner"
SKILL_INTERMEDIATE = "intermediate"
SKILL_ADVANCED = "advanced"
SKILL_EXPERT = "expert"
SKILL_ADVANCED_EXPERT = "advanced_expert"
PROTOCOL_SKILL_GROUPS: tuple[str, ...] = (
    SKILL_BEGINNER,
    SKILL_INTERMEDIATE,
    SKILL_ADVANCED_EXPERT,
)
# Accept advanced / expert as aliases of advanced_expert on ingest.
_SKILL_ALIASES: dict[str, str] = {
    SKILL_BEGINNER: SKILL_BEGINNER,
    SKILL_INTERMEDIATE: SKILL_INTERMEDIATE,
    SKILL_ADVANCED: SKILL_ADVANCED_EXPERT,
    SKILL_EXPERT: SKILL_ADVANCED_EXPERT,
    SKILL_ADVANCED_EXPERT: SKILL_ADVANCED_EXPERT,
    "advanced/expert": SKILL_ADVANCED_EXPERT,
}

# ---------------------------------------------------------------------------
# Recording specification (forehand smash, one controlled camera protocol)
# ---------------------------------------------------------------------------

PROTOCOL_STROKE_TYPE = STROKE_TYPE_FOREHAND_SMASH
PROTOCOL_MIN_FPS = 30.0
PROTOCOL_PREFERRED_FPS = 60.0
PROTOCOL_MIN_WIDTH = 1280
PROTOCOL_MIN_HEIGHT = 720
PROTOCOL_PREFERRED_WIDTH = 1920
PROTOCOL_PREFERRED_HEIGHT = 1080
PROTOCOL_PLAYERS_PER_SKILL_GROUP_MIN = 5
PROTOCOL_PLAYERS_PER_SKILL_GROUP_MAX = 10
PROTOCOL_VALID_SMASHES_PER_PLAYER_MIN = 15
PROTOCOL_VALID_SMASHES_PER_PLAYER_MAX = 30
PROTOCOL_REPEATED_SMASHES_PER_SESSION_MIN = 15
PROTOCOL_REPEATED_SMASHES_PER_SESSION_PREFERRED = 25

# Acceptance thresholds for a *valid reference stroke* (collection gate).
# Stricter than soft analysis warnings; used only for reference-data intake.
ACCEPTANCE_MIN_POSE_COVERAGE = 0.75
ACCEPTANCE_MIN_CAMERA_STABILITY = 0.60
ACCEPTANCE_REQUIRE_FULL_BODY = True
ACCEPTANCE_DISALLOW_MAJOR_OCCLUSION = True
ACCEPTANCE_REQUIRE_CONTACT_EVENT = True

_DEFAULT_PROTOCOL_NOTES = (
    "Initial reference-data protocol: forehand smash only, one controlled "
    "camera view per session (rear_45 or side_45). Does not define numeric "
    "technique reference ranges."
)


@dataclass(frozen=True, slots=True)
class RecordingProtocolSpec:
    """Controlled recording specification for forehand smash analysis."""

    protocol_version: str = REFERENCE_DATA_PROTOCOL_VERSION
    stroke_type: str = PROTOCOL_STROKE_TYPE
    allowed_camera_views: tuple[str, ...] = PROTOCOL_CAMERA_VIEWS
    min_fps: float = PROTOCOL_MIN_FPS
    preferred_fps: float = PROTOCOL_PREFERRED_FPS
    min_width: int = PROTOCOL_MIN_WIDTH
    min_height: int = PROTOCOL_MIN_HEIGHT
    preferred_width: int = PROTOCOL_PREFERRED_WIDTH
    preferred_height: int = PROTOCOL_PREFERRED_HEIGHT
    require_full_body_visibility: bool = True
    players_per_skill_group_min: int = PROTOCOL_PLAYERS_PER_SKILL_GROUP_MIN
    players_per_skill_group_max: int = PROTOCOL_PLAYERS_PER_SKILL_GROUP_MAX
    valid_smashes_per_player_min: int = PROTOCOL_VALID_SMASHES_PER_PLAYER_MIN
    valid_smashes_per_player_max: int = PROTOCOL_VALID_SMASHES_PER_PLAYER_MAX
    repeated_smashes_per_session_min: int = PROTOCOL_REPEATED_SMASHES_PER_SESSION_MIN
    repeated_smashes_per_session_preferred: int = (
        PROTOCOL_REPEATED_SMASHES_PER_SESSION_PREFERRED
    )
    notes: str = _DEFAULT_PROTOCOL_NOTES

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "stroke_type": self.stroke_type,
            "allowed_camera_views": list(self.allowed_camera_views),
            "min_fps": self.min_fps,
            "preferred_fps": self.preferred_fps,
            "min_resolution": {
                "width": self.min_width,
                "height": self.min_height,
            },
            "preferred_resolution": {
                "width": self.preferred_width,
                "height": self.preferred_height,
            },
            "require_full_body_visibility": self.require_full_body_visibility,
            "collection_targets": {
                "players_per_skill_group": {
                    "min": self.players_per_skill_group_min,
                    "max": self.players_per_skill_group_max,
                },
                "valid_smashes_per_player": {
                    "min": self.valid_smashes_per_player_min,
                    "max": self.valid_smashes_per_player_max,
                },
                "repeated_smashes_per_session": {
                    "min": self.repeated_smashes_per_session_min,
                    "preferred": self.repeated_smashes_per_session_preferred,
                },
            },
            "handedness_vocabulary": list(PROTOCOL_HANDEDNESS),
            "skill_group_vocabulary": list(PROTOCOL_SKILL_GROUPS),
            "notes": self.notes,
        }


def default_recording_protocol() -> RecordingProtocolSpec:
    return RecordingProtocolSpec()


# ---------------------------------------------------------------------------
# Typed metadata schemas
# ---------------------------------------------------------------------------


def normalize_skill_group(value: str) -> str:
    key = str(value).strip().lower().replace(" ", "_")
    if key not in _SKILL_ALIASES:
        raise ValueError(
            f"Invalid skill_group {value!r}; expected one of "
            f"{tuple(_SKILL_ALIASES.keys())}"
        )
    return _SKILL_ALIASES[key]


def normalize_handedness(value: str) -> str:
    key = str(value).strip().upper()
    if key not in PROTOCOL_HANDEDNESS:
        raise ValueError(
            f"Invalid handedness {value!r}; expected one of {PROTOCOL_HANDEDNESS}"
        )
    return key


def normalize_camera_view(value: str) -> str:
    key = str(value).strip().lower()
    if key not in PROTOCOL_CAMERA_VIEWS:
        raise ValueError(
            f"Invalid camera_view {value!r}; expected one of {PROTOCOL_CAMERA_VIEWS}"
        )
    return key


@dataclass(slots=True)
class Player:
    """Anonymized player metadata for reference collection."""

    anonymized_player_id: str
    skill_group: str
    handedness: str
    notes: str = ""

    def __post_init__(self) -> None:
        if not str(self.anonymized_player_id).strip():
            raise ValueError("anonymized_player_id is required")
        self.skill_group = normalize_skill_group(self.skill_group)
        self.handedness = normalize_handedness(self.handedness)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "anonymized_player_id": self.anonymized_player_id,
            "skill_group": self.skill_group,
            "handedness": self.handedness,
        }
        if self.notes:
            payload["notes"] = self.notes
        return payload


@dataclass(slots=True)
class CameraSetup:
    """Camera placement and capture settings for a session."""

    camera_view: str
    fps: float
    width: int
    height: int
    protocol_version: str = REFERENCE_DATA_PROTOCOL_VERSION
    camera_height_m: float | None = None
    distance_to_player_m: float | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        self.camera_view = normalize_camera_view(self.camera_view)
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")
        if self.width <= 0 or self.height <= 0:
            raise ValueError(
                f"resolution must be positive, got {self.width}x{self.height}"
            )

    @property
    def resolution(self) -> dict[str, int]:
        return {"width": int(self.width), "height": int(self.height)}

    def meets_minimum_protocol(
        self, protocol: RecordingProtocolSpec | None = None
    ) -> bool:
        spec = protocol or default_recording_protocol()
        return (
            self.camera_view in spec.allowed_camera_views
            and self.fps >= spec.min_fps
            and self.width >= spec.min_width
            and self.height >= spec.min_height
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "camera_view": self.camera_view,
            "fps": float(self.fps),
            "width": int(self.width),
            "height": int(self.height),
            "resolution": self.resolution,
            "protocol_version": self.protocol_version,
        }
        if self.camera_height_m is not None:
            payload["camera_height_m"] = float(self.camera_height_m)
        if self.distance_to_player_m is not None:
            payload["distance_to_player_m"] = float(self.distance_to_player_m)
        if self.notes:
            payload["notes"] = self.notes
        return payload


@dataclass(slots=True)
class RecordingSession:
    """One recording session for a player under a single camera setup."""

    session_id: str
    anonymized_player_id: str
    camera_setup: CameraSetup
    recorded_at: str = ""
    planned_smash_count: int | None = None
    venue_code: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        if not str(self.session_id).strip():
            raise ValueError("session_id is required")
        if not str(self.anonymized_player_id).strip():
            raise ValueError("anonymized_player_id is required")
        if (
            self.planned_smash_count is not None
            and self.planned_smash_count < 1
        ):
            raise ValueError("planned_smash_count must be >= 1 when set")

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "session_id": self.session_id,
            "anonymized_player_id": self.anonymized_player_id,
            "camera_setup": self.camera_setup.to_dict(),
        }
        if self.recorded_at:
            payload["recorded_at"] = self.recorded_at
        if self.planned_smash_count is not None:
            payload["planned_smash_count"] = int(self.planned_smash_count)
        if self.venue_code:
            payload["venue_code"] = self.venue_code
        if self.notes:
            payload["notes"] = self.notes
        return payload


@dataclass(slots=True)
class AnalysisArtifactRefs:
    """Paths/IDs linking a stroke sample to pipeline analysis outputs."""

    analysis_id: str = ""
    dataset_export_path: str | None = None
    pose_json_path: str | None = None
    phases_json_path: str | None = None
    contact_json_path: str | None = None
    technique_json_path: str | None = None
    video_quality_json_path: str | None = None
    evidence_json_path: str | None = None
    overlay_video_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.analysis_id:
            payload["analysis_id"] = self.analysis_id
        for key in (
            "dataset_export_path",
            "pose_json_path",
            "phases_json_path",
            "contact_json_path",
            "technique_json_path",
            "video_quality_json_path",
            "evidence_json_path",
            "overlay_video_path",
        ):
            value = getattr(self, key)
            if value is not None:
                payload[key] = value
        return payload


@dataclass(slots=True)
class StrokeSample:
    """One smash take within a session, with optional analysis artifact refs."""

    stroke_id: str
    session_id: str
    anonymized_player_id: str
    stroke_type: str
    take_number: int
    camera_view: str
    fps: float
    width: int
    height: int
    handedness: str
    skill_group: str
    acceptance_status: str = "pending"
    acceptance_failures: list[str] = field(default_factory=list)
    artifact_refs: AnalysisArtifactRefs = field(default_factory=AnalysisArtifactRefs)
    source_video_path: str | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if not str(self.stroke_id).strip():
            raise ValueError("stroke_id is required")
        if self.take_number < 1:
            raise ValueError(f"take_number must be >= 1, got {self.take_number}")
        self.camera_view = normalize_camera_view(self.camera_view)
        self.handedness = normalize_handedness(self.handedness)
        self.skill_group = normalize_skill_group(self.skill_group)
        self.stroke_type = str(self.stroke_type).strip().upper()
        status = str(self.acceptance_status).strip().lower()
        if status not in {"pending", "accepted", "rejected"}:
            raise ValueError(
                f"acceptance_status must be pending|accepted|rejected, got {status!r}"
            )
        self.acceptance_status = status

    @property
    def resolution(self) -> dict[str, int]:
        return {"width": int(self.width), "height": int(self.height)}

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "stroke_id": self.stroke_id,
            "session_id": self.session_id,
            "anonymized_player_id": self.anonymized_player_id,
            "stroke_type": self.stroke_type,
            "take_number": int(self.take_number),
            "camera_view": self.camera_view,
            "fps": float(self.fps),
            "width": int(self.width),
            "height": int(self.height),
            "resolution": self.resolution,
            "handedness": self.handedness,
            "skill_group": self.skill_group,
            "acceptance_status": self.acceptance_status,
            "acceptance_failures": list(self.acceptance_failures),
            "artifact_refs": self.artifact_refs.to_dict(),
        }
        if self.source_video_path is not None:
            payload["source_video_path"] = self.source_video_path
        if self.notes:
            payload["notes"] = self.notes
        return payload


@dataclass(slots=True)
class ReferenceLabel:
    """Human/system label attached to a stroke for future reference building.

    Stores categorical labels only — no computed percentile ranges.
    """

    label_id: str
    stroke_id: str
    labeler_id: str
    labeled_at: str = ""
    phase_boundary_notes: str = ""
    issue_labels: list[str] = field(default_factory=list)
    quality_tags: list[str] = field(default_factory=list)
    is_valid_reference_stroke: bool | None = None
    comments: str = ""

    def __post_init__(self) -> None:
        if not str(self.label_id).strip():
            raise ValueError("label_id is required")
        if not str(self.stroke_id).strip():
            raise ValueError("stroke_id is required")
        if not str(self.labeler_id).strip():
            raise ValueError("labeler_id is required")

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "label_id": self.label_id,
            "stroke_id": self.stroke_id,
            "labeler_id": self.labeler_id,
            "issue_labels": list(self.issue_labels),
            "quality_tags": list(self.quality_tags),
        }
        if self.labeled_at:
            payload["labeled_at"] = self.labeled_at
        if self.phase_boundary_notes:
            payload["phase_boundary_notes"] = self.phase_boundary_notes
        if self.is_valid_reference_stroke is not None:
            payload["is_valid_reference_stroke"] = bool(self.is_valid_reference_stroke)
        if self.comments:
            payload["comments"] = self.comments
        return payload


@dataclass(slots=True)
class ReferenceDatasetManifest:
    """Versioned collection of players, sessions, strokes, and reference labels."""

    schema_version: str = REFERENCE_DATASET_SCHEMA_VERSION
    protocol_version: str = REFERENCE_DATA_PROTOCOL_VERSION
    protocol: RecordingProtocolSpec = field(default_factory=default_recording_protocol)
    players: list[Player] = field(default_factory=list)
    sessions: list[RecordingSession] = field(default_factory=list)
    strokes: list[StrokeSample] = field(default_factory=list)
    labels: list[ReferenceLabel] = field(default_factory=list)
    notes: str = (
        "Reference-data manifest for controlled forehand-smash collection. "
        "Schema only — no reference-range computation."
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference_dataset_schema_version": self.schema_version,
            "reference_data_protocol_version": self.protocol_version,
            "protocol": self.protocol.to_dict(),
            "player_count": len(self.players),
            "session_count": len(self.sessions),
            "stroke_count": len(self.strokes),
            "label_count": len(self.labels),
            "players": [p.to_dict() for p in self.players],
            "sessions": [s.to_dict() for s in self.sessions],
            "strokes": [s.to_dict() for s in self.strokes],
            "labels": [label.to_dict() for label in self.labels],
            "notes": self.notes,
        }

    def save_json(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path


# ---------------------------------------------------------------------------
# Stroke acceptance criteria (collection gate only)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class StrokeAcceptanceSignals:
    """Observable signals used to decide if a stroke is valid for the dataset."""

    full_body_visible: bool
    major_occlusion: bool
    camera_stable: bool
    pose_coverage: float
    contact_available_or_estimable: bool
    fps: float
    width: int
    height: int
    camera_view: str


@dataclass(slots=True)
class StrokeAcceptanceResult:
    accepted: bool
    failures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "failures": list(self.failures),
        }


def evaluate_stroke_acceptance(
    signals: StrokeAcceptanceSignals,
    *,
    protocol: RecordingProtocolSpec | None = None,
    min_pose_coverage: float = ACCEPTANCE_MIN_POSE_COVERAGE,
    min_camera_stability_flag: bool = True,
) -> StrokeAcceptanceResult:
    """Apply reference-collection acceptance criteria (no range computation)."""
    spec = protocol or default_recording_protocol()
    failures: list[str] = []

    try:
        view = normalize_camera_view(signals.camera_view)
    except ValueError:
        failures.append("invalid_camera_view")
        view = signals.camera_view

    if view not in spec.allowed_camera_views:
        failures.append("camera_view_not_in_protocol")
    if signals.fps < spec.min_fps:
        failures.append("fps_below_minimum")
    if signals.width < spec.min_width or signals.height < spec.min_height:
        failures.append("resolution_below_minimum")
    if ACCEPTANCE_REQUIRE_FULL_BODY and not signals.full_body_visible:
        failures.append("full_body_not_visible")
    if ACCEPTANCE_DISALLOW_MAJOR_OCCLUSION and signals.major_occlusion:
        failures.append("major_occlusion")
    if min_camera_stability_flag and not signals.camera_stable:
        failures.append("unstable_camera")
    if signals.pose_coverage < min_pose_coverage:
        failures.append("pose_coverage_below_threshold")
    if ACCEPTANCE_REQUIRE_CONTACT_EVENT and not signals.contact_available_or_estimable:
        failures.append("contact_event_unavailable")

    return StrokeAcceptanceResult(accepted=not failures, failures=failures)


# ---------------------------------------------------------------------------
# Load / parse
# ---------------------------------------------------------------------------


def load_reference_dataset_manifest(path: Path) -> ReferenceDatasetManifest:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return reference_dataset_manifest_from_dict(data)


def reference_dataset_manifest_from_dict(
    data: dict[str, Any],
) -> ReferenceDatasetManifest:
    protocol_raw = data.get("protocol") or {}
    protocol = RecordingProtocolSpec(
        protocol_version=str(
            protocol_raw.get("protocol_version")
            or data.get("reference_data_protocol_version")
            or REFERENCE_DATA_PROTOCOL_VERSION
        ),
        stroke_type=str(protocol_raw.get("stroke_type") or PROTOCOL_STROKE_TYPE),
        allowed_camera_views=tuple(
            protocol_raw.get("allowed_camera_views") or PROTOCOL_CAMERA_VIEWS
        ),
        min_fps=float(protocol_raw.get("min_fps") or PROTOCOL_MIN_FPS),
        preferred_fps=float(
            protocol_raw.get("preferred_fps") or PROTOCOL_PREFERRED_FPS
        ),
        min_width=int(
            (protocol_raw.get("min_resolution") or {}).get("width")
            or protocol_raw.get("min_width")
            or PROTOCOL_MIN_WIDTH
        ),
        min_height=int(
            (protocol_raw.get("min_resolution") or {}).get("height")
            or protocol_raw.get("min_height")
            or PROTOCOL_MIN_HEIGHT
        ),
        preferred_width=int(
            (protocol_raw.get("preferred_resolution") or {}).get("width")
            or PROTOCOL_PREFERRED_WIDTH
        ),
        preferred_height=int(
            (protocol_raw.get("preferred_resolution") or {}).get("height")
            or PROTOCOL_PREFERRED_HEIGHT
        ),
        notes=str(protocol_raw.get("notes") or _DEFAULT_PROTOCOL_NOTES),
    )

    players = [
        Player(
            anonymized_player_id=str(raw["anonymized_player_id"]),
            skill_group=str(raw["skill_group"]),
            handedness=str(raw["handedness"]),
            notes=str(raw.get("notes") or ""),
        )
        for raw in data.get("players") or []
    ]

    sessions: list[RecordingSession] = []
    for raw in data.get("sessions") or []:
        cam = raw.get("camera_setup") or {}
        sessions.append(
            RecordingSession(
                session_id=str(raw["session_id"]),
                anonymized_player_id=str(raw["anonymized_player_id"]),
                camera_setup=CameraSetup(
                    camera_view=str(cam["camera_view"]),
                    fps=float(cam["fps"]),
                    width=int(cam.get("width") or (cam.get("resolution") or {})["width"]),
                    height=int(
                        cam.get("height") or (cam.get("resolution") or {})["height"]
                    ),
                    protocol_version=str(
                        cam.get("protocol_version") or protocol.protocol_version
                    ),
                    camera_height_m=(
                        float(cam["camera_height_m"])
                        if cam.get("camera_height_m") is not None
                        else None
                    ),
                    distance_to_player_m=(
                        float(cam["distance_to_player_m"])
                        if cam.get("distance_to_player_m") is not None
                        else None
                    ),
                    notes=str(cam.get("notes") or ""),
                ),
                recorded_at=str(raw.get("recorded_at") or ""),
                planned_smash_count=(
                    int(raw["planned_smash_count"])
                    if raw.get("planned_smash_count") is not None
                    else None
                ),
                venue_code=str(raw.get("venue_code") or ""),
                notes=str(raw.get("notes") or ""),
            )
        )

    strokes: list[StrokeSample] = []
    for raw in data.get("strokes") or []:
        refs = raw.get("artifact_refs") or {}
        strokes.append(
            StrokeSample(
                stroke_id=str(raw["stroke_id"]),
                session_id=str(raw["session_id"]),
                anonymized_player_id=str(raw["anonymized_player_id"]),
                stroke_type=str(raw.get("stroke_type") or PROTOCOL_STROKE_TYPE),
                take_number=int(raw["take_number"]),
                camera_view=str(raw["camera_view"]),
                fps=float(raw["fps"]),
                width=int(raw.get("width") or (raw.get("resolution") or {})["width"]),
                height=int(
                    raw.get("height") or (raw.get("resolution") or {})["height"]
                ),
                handedness=str(raw["handedness"]),
                skill_group=str(raw["skill_group"]),
                acceptance_status=str(raw.get("acceptance_status") or "pending"),
                acceptance_failures=[
                    str(x) for x in (raw.get("acceptance_failures") or [])
                ],
                artifact_refs=AnalysisArtifactRefs(
                    analysis_id=str(refs.get("analysis_id") or ""),
                    dataset_export_path=refs.get("dataset_export_path"),
                    pose_json_path=refs.get("pose_json_path"),
                    phases_json_path=refs.get("phases_json_path"),
                    contact_json_path=refs.get("contact_json_path"),
                    technique_json_path=refs.get("technique_json_path"),
                    video_quality_json_path=refs.get("video_quality_json_path"),
                    evidence_json_path=refs.get("evidence_json_path"),
                    overlay_video_path=refs.get("overlay_video_path"),
                ),
                source_video_path=raw.get("source_video_path"),
                notes=str(raw.get("notes") or ""),
            )
        )

    labels = [
        ReferenceLabel(
            label_id=str(raw["label_id"]),
            stroke_id=str(raw["stroke_id"]),
            labeler_id=str(raw["labeler_id"]),
            labeled_at=str(raw.get("labeled_at") or ""),
            phase_boundary_notes=str(raw.get("phase_boundary_notes") or ""),
            issue_labels=[str(x) for x in (raw.get("issue_labels") or [])],
            quality_tags=[str(x) for x in (raw.get("quality_tags") or [])],
            is_valid_reference_stroke=(
                bool(raw["is_valid_reference_stroke"])
                if raw.get("is_valid_reference_stroke") is not None
                else None
            ),
            comments=str(raw.get("comments") or ""),
        )
        for raw in data.get("labels") or []
    ]

    return ReferenceDatasetManifest(
        schema_version=str(
            data.get("reference_dataset_schema_version")
            or REFERENCE_DATASET_SCHEMA_VERSION
        ),
        protocol_version=str(
            data.get("reference_data_protocol_version")
            or protocol.protocol_version
        ),
        protocol=protocol,
        players=players,
        sessions=sessions,
        strokes=strokes,
        labels=labels,
        notes=str(
            data.get("notes")
            or ReferenceDatasetManifest().notes
        ),
    )
