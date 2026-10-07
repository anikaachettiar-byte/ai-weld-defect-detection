"""Core domain models and common types for the weld inspection system."""

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
