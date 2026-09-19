"""Offline validation tooling (not part of the production analyze pipeline)."""

from app.validation.angle_annotations import (
    ANGLE_VALIDATION_ANNOTATION_VERSION,
    AngleValidationAnnotationSet,
    AnnotatedAngleFrame,
    load_angle_annotation_set,
)
from app.validation.angle_debug_viz import render_worst_angle_debug_images
from app.validation.angle_joints import VALIDATED_ANGLE_NAMES
from app.validation.angle_report import (
    ANGLE_VALIDATION_REPORT_VERSION,
    AngleValidationReport,
)
from app.validation.angle_validator import (
    AngleValidator,
    export_angle_validation_report,
    resolve_ground_truth_angles,
    validate_angles,
)
from app.validation.annotations import (
    POSE_VALIDATION_ANNOTATION_VERSION,
    AnnotatedPoseFrame,
    PoseValidationAnnotationSet,
    load_annotation_set,
)
from app.validation.debug_viz import render_worst_frame_debug_images
from app.validation.joints import BADMINTON_CRITICAL_JOINTS, COCO17_JOINTS
from app.validation.phase_contact_annotations import (
    PHASE_CONTACT_VALIDATION_ANNOTATION_VERSION,
    BoundaryAnnotation,
    PhaseContactAnnotationSet,
    VideoPhaseContactAnnotation,
    load_phase_contact_annotation_set,
)
from app.validation.phase_contact_boundaries import BOUNDARY_NAMES
from app.validation.phase_contact_debug_viz import render_worst_timing_timelines
from app.validation.phase_contact_report import (
    PHASE_CONTACT_VALIDATION_REPORT_VERSION,
    PhaseContactValidationReport,
)
from app.validation.phase_contact_validator import (
    PhaseContactValidator,
    export_phase_contact_validation_report,
    extract_predicted_boundaries,
    validate_phase_contact,
)
from app.validation.report import (
    POSE_VALIDATION_REPORT_VERSION,
    PoseValidationReport,
)
from app.validation.technique_annotations import (
    TECHNIQUE_VALIDATION_ANNOTATION_VERSION,
    CoachIssueLabel,
    StrokeTechniqueAnnotation,
    TechniqueValidationAnnotationSet,
    load_technique_annotation_set,
)
from app.validation.technique_issues import (
    SUPPORTED_TECHNIQUE_ISSUE_CODES,
)
from app.validation.technique_report import (
    TECHNIQUE_VALIDATION_REPORT_VERSION,
    TechniqueValidationReport,
)
from app.validation.technique_validator import (
    TechniqueIssueValidator,
    export_technique_validation_report,
    validate_technique_issues,
)
from app.validation.validator import (
    PoseValidator,
    export_pose_validation_report,
    validate_pose,
)

__all__ = [
    "ANGLE_VALIDATION_ANNOTATION_VERSION",
    "ANGLE_VALIDATION_REPORT_VERSION",
    "AnnotatedAngleFrame",
    "AnnotatedPoseFrame",
    "AngleValidationAnnotationSet",
    "AngleValidationReport",
    "AngleValidator",
    "BADMINTON_CRITICAL_JOINTS",
    "BOUNDARY_NAMES",
    "BoundaryAnnotation",
    "COCO17_JOINTS",
    "CoachIssueLabel",
    "PHASE_CONTACT_VALIDATION_ANNOTATION_VERSION",
    "PHASE_CONTACT_VALIDATION_REPORT_VERSION",
    "POSE_VALIDATION_ANNOTATION_VERSION",
    "POSE_VALIDATION_REPORT_VERSION",
    "PhaseContactAnnotationSet",
    "PhaseContactValidationReport",
    "PhaseContactValidator",
    "PoseValidationAnnotationSet",
    "PoseValidationReport",
    "PoseValidator",
    "SUPPORTED_TECHNIQUE_ISSUE_CODES",
    "StrokeTechniqueAnnotation",
    "TECHNIQUE_VALIDATION_ANNOTATION_VERSION",
    "TECHNIQUE_VALIDATION_REPORT_VERSION",
    "TechniqueIssueValidator",
    "TechniqueValidationAnnotationSet",
    "TechniqueValidationReport",
    "VALIDATED_ANGLE_NAMES",
    "VideoPhaseContactAnnotation",
    "export_angle_validation_report",
    "export_phase_contact_validation_report",
    "export_pose_validation_report",
    "export_technique_validation_report",
    "extract_predicted_boundaries",
    "load_angle_annotation_set",
    "load_annotation_set",
    "load_phase_contact_annotation_set",
    "load_technique_annotation_set",
    "render_worst_angle_debug_images",
    "render_worst_frame_debug_images",
    "render_worst_timing_timelines",
    "resolve_ground_truth_angles",
    "validate_angles",
    "validate_phase_contact",
    "validate_pose",
    "validate_technique_issues",
]
