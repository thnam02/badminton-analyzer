"""Optional OpenAI coaching layer (Responses API + Structured Outputs)."""

from app.ai.coaching import (
    CoachingConfigError,
    CoachingParseError,
    build_fallback_report,
    build_skipped_report,
    coaching_report_from_model,
    generate_coaching_report,
    is_coaching_configured,
    validate_coaching_model,
)

__all__ = [
    "CoachingConfigError",
    "CoachingParseError",
    "build_fallback_report",
    "build_skipped_report",
    "coaching_report_from_model",
    "generate_coaching_report",
    "is_coaching_configured",
    "validate_coaching_model",
]
