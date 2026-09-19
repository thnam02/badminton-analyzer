"""ReferenceProfileSelector — choose the best versioned profile with explicit fallbacks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from app.schemas.reference import ReferenceProfile

# Match levels (most specific → least). ``none`` means use provisional hard-coded fallbacks.
MATCH_EXACT = "exact"
MATCH_SKILL_FALLBACK = "skill_fallback"
MATCH_HAND_VIEW = "hand_view"
MATCH_HAND = "hand_fallback"
MATCH_VIEW = "view_fallback"
MATCH_STROKE = "stroke_fallback"
MATCH_NONE = "none"


@dataclass(frozen=True, slots=True)
class ProfileSelection:
    """Result of profile selection — never silently invents a mismatch as exact."""

    profile: ReferenceProfile | None
    match_level: str
    reason: str

    @property
    def has_valid_profile(self) -> bool:
        return self.profile is not None and self.match_level != MATCH_NONE


def _default_catalog() -> list[ReferenceProfile]:
    # Lazy import avoids circular dependency with reference_profiles.
    from app.processing.reference_profiles import default_reference_profiles

    return default_reference_profiles()


class ReferenceProfileSelector:
    """Select the best ReferenceProfile for technique evaluation.

    Preference order:
    1. Explicit ``profile_id``
    2. Exact stroke + handedness + camera_view + skill_level (when skill requested)
    3. Exact stroke + hand + view (skill wildcard / skill not requested)
    4. stroke + hand + any view
    5. stroke + any hand + view
    6. stroke wildcard (any hand/view/skill)
    7. ``none`` — caller must use provisional fallback rules (not mix thresholds)
    """

    def __init__(
        self,
        profiles: Sequence[ReferenceProfile] | None = None,
        *,
        allow_stroke_wildcard: bool = True,
    ) -> None:
        self.profiles = list(profiles) if profiles is not None else _default_catalog()
        self.allow_stroke_wildcard = bool(allow_stroke_wildcard)

    def select(
        self,
        *,
        stroke_type: str = "SMASH",
        handedness: str | None = None,
        camera_view: str | None = None,
        skill_level: str | None = None,
        profile_id: str | None = None,
    ) -> ProfileSelection:
        catalog = self.profiles
        if not catalog:
            return ProfileSelection(
                profile=None,
                match_level=MATCH_NONE,
                reason="empty_profile_catalog",
            )

        if profile_id:
            for profile in catalog:
                if profile.profile_id == profile_id:
                    return ProfileSelection(
                        profile=profile,
                        match_level=MATCH_EXACT,
                        reason=f"explicit_profile_id:{profile_id}",
                    )
            raise KeyError(f"Unknown reference profile_id '{profile_id}'")

        stroke = _normalize_stroke(stroke_type)
        hand = _normalize_hand(handedness)
        view = _normalize_view(camera_view)
        skill = _normalize_skill(skill_level)
        skill_requested = skill is not None

        scored: list[tuple[tuple[int, int, int, int], ReferenceProfile, str]] = []
        for profile in catalog:
            if _normalize_stroke(profile.stroke_type) != stroke:
                continue
            hand_score, hand_ok = _axis_score(hand, _normalize_hand(profile.handedness))
            view_score, view_ok = _axis_score(view, _normalize_view(profile.camera_view))
            skill_score, skill_ok = _axis_score(
                skill, _normalize_skill(profile.skill_level)
            )
            if not (hand_ok and view_ok and skill_ok):
                continue
            # Prefer wildcard skill profiles when skill was not requested.
            rank_skill = skill_score if skill_requested else (2 if skill_score >= 1 else 0)
            score = (hand_score, view_score, rank_skill, skill_score)
            level = _match_level(
                hand_score,
                view_score,
                skill_score,
                skill_requested=skill_requested,
            )
            scored.append((score, profile, level))

        if not scored:
            if self.allow_stroke_wildcard:
                wildcards = [
                    p
                    for p in catalog
                    if _normalize_stroke(p.stroke_type) == stroke
                    and _normalize_hand(p.handedness) is None
                    and _normalize_view(p.camera_view) is None
                ]
                if wildcards:
                    return ProfileSelection(
                        profile=wildcards[0],
                        match_level=MATCH_STROKE,
                        reason="stroke_wildcard_profile",
                    )
            return ProfileSelection(
                profile=None,
                match_level=MATCH_NONE,
                reason="no_matching_reference_profile",
            )

        scored.sort(key=lambda item: item[0], reverse=True)
        best_score, best_profile, best_level = scored[0]
        return ProfileSelection(
            profile=best_profile,
            match_level=best_level,
            reason=(
                f"selected score=hand:{best_score[0]},view:{best_score[1]},"
                f"skill:{best_score[2]} → {best_level}"
            ),
        )


def select_reference_profile(
    *,
    stroke_type: str = "SMASH",
    handedness: str | None = None,
    camera_view: str | None = None,
    skill_level: str | None = None,
    profile_id: str | None = None,
    profiles: list[ReferenceProfile] | None = None,
) -> ReferenceProfile:
    """Backward-compatible helper that always returns a profile when catalog non-empty.

    Prefer ``ReferenceProfileSelector.select`` when fallback-to-hard-coded-rules
    must be explicit (``match_level == none``).
    """
    selection = ReferenceProfileSelector(profiles).select(
        stroke_type=stroke_type,
        handedness=handedness,
        camera_view=camera_view,
        skill_level=skill_level,
        profile_id=profile_id,
    )
    if selection.profile is None:
        catalog = list(profiles) if profiles is not None else _default_catalog()
        if not catalog:
            raise ValueError("No reference profiles available")
        return catalog[0]
    return selection.profile


def _match_level(
    hand_score: int,
    view_score: int,
    skill_score: int,
    *,
    skill_requested: bool,
) -> str:
    if hand_score == 2 and view_score == 2:
        if skill_requested:
            if skill_score == 2:
                return MATCH_EXACT
            if skill_score == 1:
                return MATCH_SKILL_FALLBACK
            return MATCH_HAND_VIEW
        # Skill not requested: hand+view match is exact enough.
        return MATCH_EXACT if skill_score >= 1 else MATCH_HAND_VIEW
    if hand_score == 2:
        return MATCH_HAND
    if view_score == 2:
        return MATCH_VIEW
    return MATCH_STROKE


def _axis_score(
    requested: str | None, profile_value: str | None
) -> tuple[int, bool]:
    """Return (score, compatible).

    score 2 = exact match, 1 = profile wildcard, 0 = request wildcard vs concrete,
    incompatible when both concrete and disagree.
    """
    if requested is not None and profile_value is not None:
        if requested == profile_value:
            return 2, True
        return -1, False
    if profile_value is None:
        return 1, True
    return 0, True


def _normalize_stroke(value: str | None) -> str:
    raw = (value or "SMASH").strip().upper()
    if raw in {"FOREHAND_SMASH", "SMASH_FOREHAND"}:
        return "SMASH"
    return raw


def _normalize_hand(value: str | None) -> str | None:
    if value is None or str(value).strip() == "":
        return None
    return str(value).strip().upper()


def _normalize_view(value: str | None) -> str | None:
    if value is None or str(value).strip() == "":
        return None
    raw = str(value).strip().lower().replace("-", "_")
    aliases = {
        "side": "side_45",
        "side_45": "side_45",
        "rear": "rear_45",
        "rear_45": "rear_45",
        "front": "front",
    }
    return aliases.get(raw, raw)


def _normalize_skill(value: str | None) -> str | None:
    if value is None or str(value).strip() == "":
        return None
    raw = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    if raw in {"advanced", "expert", "advanced/expert", "advanced_expert"}:
        return "advanced_expert"
    return raw
