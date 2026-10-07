"""Core data types and structures for the NDT Weld Inspection System."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


class DefectCode(str, Enum):
    """Standard radiographic weld defect classification codes."""
    CR = "CR"  # Crack
    PO = "PO"  # Porosity
    LP = "LP"  # Lack of Penetration
    ND = "ND"  # No Defect


@dataclass
class DefectClassInfo:
    """Metadata regarding a defect classification."""
    code: DefectCode
    name: str
    description: str
    severity: str


DEFECT_CATALOG: Dict[DefectCode, DefectClassInfo] = {
    DefectCode.CR: DefectClassInfo(
        code=DefectCode.CR,
        name="Crack",
        description="Linear discontinuity with sharp tip; severe stress concentrator.",
        severity="Critical"
    ),
    DefectCode.PO: DefectClassInfo(
        code=DefectCode.PO,
        name="Porosity",
        description="Gas pocket cavity formed during solidification; rounded or clustered.",
        severity="Moderate"
    ),
    DefectCode.LP: DefectClassInfo(
        code=DefectCode.LP,
        name="Lack of Penetration",
        description="Failure of weld metal to extend into root of joint; continuous or intermittent line.",
        severity="Critical"
    ),
    DefectCode.ND: DefectClassInfo(
        code=DefectCode.ND,
        name="No Defect",
        description="Weld satisfies acceptance standards; sound joint without rejectable anomalies.",
        severity="Acceptable"
    ),
}


@dataclass
class PreprocessingMetrics:
    """Quantitative metrics measuring image contrast and quality before and after enhancement."""
    original_mean: float
    original_std: float
    original_entropy: float
    enhanced_mean: float
    enhanced_std: float
    enhanced_entropy: float
    contrast_improvement_ratio: float
    dynamic_range_utilization: float


@dataclass
class PreprocessedArtifacts:
    """Container holding intermediate and final results of the preprocessing stage."""
    raw_image: np.ndarray
    denoised_image: Optional[np.ndarray] = None
    clahe_image: Optional[np.ndarray] = None
    final_image: Optional[np.ndarray] = None
    difference_map: Optional[np.ndarray] = None
    metrics: Optional[PreprocessingMetrics] = None
    applied_parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BoundingBox:
    """Bounding box coordinates for detected defect."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    defect_class: DefectCode


@dataclass
class MorphometricMeasurements:
    """Physical and geometric measurements of a segmented defect."""
    length_px: float
    width_px: float
    area_px: float
    aspect_ratio: float
    centroid: Tuple[float, float]
    orientation_deg: float
    length_mm: Optional[float] = None
    width_mm: Optional[float] = None
    area_mm2: Optional[float] = None


@dataclass
class DefectDetection:
    """Comprehensive record of an individual defect detected on a radiograph."""
    defect_id: int
    defect_class: DefectCode
    confidence: float
    bounding_box: BoundingBox
    segmentation_mask: Optional[np.ndarray] = None
    measurements: Optional[MorphometricMeasurements] = None
    explainability_heatmap: Optional[np.ndarray] = None


@dataclass
class WeldInspectionReport:
    """Full end-to-end inspection record for a radiographic weld image."""
    image_name: str
    original_shape: Tuple[int, int]
    preprocessing: PreprocessedArtifacts
    detections: List[DefectDetection] = field(default_factory=list)
    overall_status: str = "PENDING"
    notes: List[str] = field(default_factory=list)
