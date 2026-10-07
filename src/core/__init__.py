"""Core domain models and common types for the weld inspection system."""

from src.core.auth import UserProfile, authenticate, get_demo_user
from src.core.types import (
    DefectCode,
    DefectClassInfo,
    DEFECT_CATALOG,
    PreprocessingMetrics,
    PreprocessedArtifacts,
    BoundingBox,
    MorphometricMeasurements,
    DefectDetection,
    WeldInspectionReport,
)

__all__ = [
    "UserProfile",
    "authenticate",
    "get_demo_user",
    "DefectCode",
    "DefectClassInfo",
    "DEFECT_CATALOG",
    "PreprocessingMetrics",
    "PreprocessedArtifacts",
    "BoundingBox",
    "MorphometricMeasurements",
    "DefectDetection",
    "WeldInspectionReport",
]
