"""Offline pose-validation tooling (not part of the production analyze pipeline)."""

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
from app.validation.report import (
    POSE_VALIDATION_REPORT_VERSION,
    PoseValidationReport,
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
    "COCO17_JOINTS",
    "POSE_VALIDATION_ANNOTATION_VERSION",
    "POSE_VALIDATION_REPORT_VERSION",
    "PoseValidationAnnotationSet",
    "PoseValidationReport",
    "PoseValidator",
    "VALIDATED_ANGLE_NAMES",
    "export_angle_validation_report",
    "export_pose_validation_report",
    "load_angle_annotation_set",
    "load_annotation_set",
    "render_worst_angle_debug_images",
    "render_worst_frame_debug_images",
    "resolve_ground_truth_angles",
    "validate_angles",
    "validate_pose",
]
