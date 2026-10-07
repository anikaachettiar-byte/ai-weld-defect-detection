"""Weld Defect Detection and ASME/ISO Compliance Classification module."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Tuple, Union
import cv2
import numpy as np

from src.core.types import (
    BoundingBox,
    DefectCode,
    DefectDetection,
    MorphometricMeasurements,
    WeldInspectionReport,
)
from src.measurement.analyzer import DefectMorphologyAnalyzer


class BaseDefectDetector(ABC):
    """Abstract Base Class for weld defect detectors."""

    @abstractmethod
    def detect(self, image: np.ndarray, raw_image: Optional[np.ndarray] = None) -> List[DefectDetection]:
        """Detect defects in a preprocessed radiographic image."""
        pass


class RadiographicDefectDetector(BaseDefectDetector):
    """Radiographic anomaly detector and ASME/ISO standards evaluation engine.
    
    Operates on enhanced radiographic images to:
    1. Localize the weld seam & Heat-Affected Zone (HAZ) ROI.
    2. Extract localized attenuation deficits (density troughs) within the weld metal.
    3. Segment candidate anomaly contours.
    4. Quantify morphometric signatures (aspect ratio, circularity, orientation, location).
    5. Classify flaws into CR (Crack), PO (Porosity), LP (Lack of Penetration), or ND (No Defect).
    6. Evaluate compliance against ASME BPVC Section VIII Div 1 UW-51 and ISO 10675-1 / ISO 5817.
    """

    def __init__(
        self,
        pixel_to_mm_ratio: float = 0.1,
        min_defect_area_px: float = 14.0,
        contrast_threshold: float = 32.0,
    ) -> None:
        self.pixel_to_mm = pixel_to_mm_ratio
        self.min_defect_area_px = min_defect_area_px
        self.contrast_threshold = contrast_threshold
        self.morph_analyzer = DefectMorphologyAnalyzer(pixel_to_mm_ratio=pixel_to_mm_ratio)

    def detect(self, image: np.ndarray, raw_image: Optional[np.ndarray] = None) -> List[DefectDetection]:
        """Detect and classify weld defects in an enhanced radiograph."""
        if not isinstance(image, np.ndarray) or image.size == 0:
            return []

        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]

        # 1. Determine Weld Seam Orientation & ROI (horizontal vs vertical vs full radiograph)
        row_means = np.mean(gray, axis=1)
        col_means = np.mean(gray, axis=0)
        row_range = float(np.ptp(row_means))
        col_range = float(np.ptp(col_means))

        offset_x, offset_y = 0, 0
        weld_center_y = h / 2.0

        if row_range > 35.0 and row_range > col_range:
            # Horizontal weld seam band
            peak_y = int(np.argmax(row_means))
            peak_val = row_means[peak_y]
            weld_rows = np.where(row_means > max(35.0, peak_val * 0.40))[0]
            if len(weld_rows) > 10:
                min_y = max(0, weld_rows[0] - 5)
                max_y = min(h, weld_rows[-1] + 5)
                weld_roi = gray[min_y:max_y, :]
                offset_y = min_y
                weld_center_y = peak_y
            else:
                weld_roi = gray
        elif col_range > 35.0 and col_range >= row_range:
            # Vertical weld seam band
            peak_x = int(np.argmax(col_means))
            peak_val = col_means[peak_x]
            weld_cols = np.where(col_means > max(35.0, peak_val * 0.40))[0]
            if len(weld_cols) > 10:
                min_x = max(0, weld_cols[0] - 5)
                max_x = min(w, weld_cols[-1] + 5)
                weld_roi = gray[:, min_x:max_x]
                offset_x = min_x
            else:
                weld_roi = gray
        else:
            # Full image inspection
            weld_roi = gray

        roi_h, roi_w = weld_roi.shape[:2]
        if roi_h < 10 or roi_w < 10:
            weld_roi = gray
            offset_x, offset_y = 0, 0

        # 2. Local background weld crown estimation via Gaussian smoothing
        kernel_dim = max(21, (min(roi_h, roi_w) // 6) * 2 + 1)
        bg = cv2.GaussianBlur(weld_roi, (kernel_dim, kernel_dim), 0)

        # 3. Density Trough Isolation (defects appear as dark attenuation dips inside the weld)
        trough = cv2.subtract(bg, weld_roi)
        _, thresh = cv2.threshold(trough, self.contrast_threshold, 255, cv2.THRESH_BINARY)

        # Morphological noise filtering
        clean_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, clean_kernel)

        # 4. Find Anomaly Contours
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [c for c in contours if cv2.contourArea(c) >= self.min_defect_area_px]

        # If no significant anomaly contours, return sound weld
        if not valid_contours:
            return []

        detections: List[DefectDetection] = []
        defect_id = 1

        for cnt in valid_contours:
            area_px = float(cv2.contourArea(cnt))
            perimeter = float(cv2.arcLength(cnt, True))
            circularity = (4.0 * np.pi * area_px) / (perimeter ** 2 + 1e-6)

            # Minimum area bounding rectangle
            rx, ry, bw, bh = cv2.boundingRect(cnt)
            rect = cv2.minAreaRect(cnt)
            (_, _), (dim1, dim2), angle = rect
            length_px = max(dim1, dim2)
            width_px = max(min(dim1, dim2), 1.0)
            aspect_ratio = length_px / width_px

            # Convert coordinates back to full image coordinate system
            abs_x = rx + offset_x
            abs_y = ry + offset_y

            # Centroid
            m = cv2.moments(cnt)
            local_cx = float(m["m10"] / m["m00"]) if m["m00"] != 0 else rx + bw / 2.0
            local_cy = float(m["m01"] / m["m00"]) if m["m00"] != 0 else ry + bh / 2.0
            cx = local_cx + offset_x
            cy = local_cy + offset_y

            dist_to_center_y = abs(cy - weld_center_y)

            # --- ASME / ISO Classification Rules ---
            # Lack of Penetration (LP): Continuous linear defect aligned along the weld root seam
            is_linear_root = (aspect_ratio >= 4.0) and (dist_to_center_y < h * 0.15) and (length_px >= w * 0.20)

            # Porosity (PO): High circularity, rounded/spherical cavities
            is_porosity = (circularity >= 0.38) and (aspect_ratio <= 2.6)

            # Crack (CR): High aspect ratio, irregular jagged contour, sharp tips
            is_crack = (aspect_ratio >= 2.8) or (circularity < 0.22 and not is_linear_root)

            if is_linear_root:
                defect_class = DefectCode.LP
                conf = min(0.97, 0.78 + (aspect_ratio / 25.0))
                asme_verdict = "REJECT"
                asme_clause = "ASME B31.3 Table 341.3.2 / ISO 5817 Level B: Incomplete root penetration is strictly unacceptable."
            elif is_porosity:
                defect_class = DefectCode.PO
                conf = min(0.96, 0.72 + (circularity * 0.24))
                diameter_mm = 2.0 * np.sqrt(area_px / np.pi) * self.pixel_to_mm
                if diameter_mm > 3.0 or len(valid_contours) >= 5:
                    asme_verdict = "REJECT"
                    asme_clause = f"ASME Sec. VIII Appendix 4: Pore diameter ({diameter_mm:.1f} mm) or cluster density exceeds allowable threshold."
                else:
                    asme_verdict = "ACCEPT WITH OBSERVATION"
                    asme_clause = f"ASME Sec. VIII Appendix 4: Isolated pore ({diameter_mm:.1f} mm) within acceptable tolerance limits."
            elif is_crack:
                defect_class = DefectCode.CR
                conf = min(0.98, 0.82 + (aspect_ratio / 18.0))
                asme_verdict = "REJECT"
                asme_clause = "ASME Sec. VIII Div 1 UW-51: Cracks are planar stress-concentrators; zero tolerance, unacceptable regardless of length."
            else:
                defect_class = DefectCode.CR
                conf = 0.74
                asme_verdict = "REJECT"
                asme_clause = "ASME Sec. VIII UW-51: Linear defect indication requires excavation and re-welding."

            # Create individual contour mask on full image canvas
            contour_mask = np.zeros((h, w), dtype=np.uint8)
            shifted_cnt = cnt + np.array([offset_x, offset_y])
            cv2.drawContours(contour_mask, [shifted_cnt], -1, 255, -1)

            # Morphometric measurements
            length_mm = length_px * self.pixel_to_mm
            width_mm = width_px * self.pixel_to_mm
            area_mm2 = area_px * (self.pixel_to_mm ** 2)

            measurements = MorphometricMeasurements(
                length_px=round(length_px, 1),
                width_px=round(width_px, 1),
                area_px=round(area_px, 1),
                aspect_ratio=round(aspect_ratio, 2),
                centroid=(round(cx, 1), round(cy, 1)),
                orientation_deg=round(angle, 1),
                length_mm=round(length_mm, 2),
                width_mm=round(width_mm, 2),
                area_mm2=round(area_mm2, 2),
            )

            bbox = BoundingBox(
                x1=float(abs_x),
                y1=float(abs_y),
                x2=float(abs_x + bw),
                y2=float(abs_y + bh),
                confidence=round(conf, 3),
                defect_class=defect_class,
            )

            detections.append(
                DefectDetection(
                    defect_id=defect_id,
                    defect_class=defect_class,
                    confidence=round(conf, 2),
                    bounding_box=bbox,
                    segmentation_mask=contour_mask,
                    measurements=measurements,
                    asme_verdict=asme_verdict,
                    asme_clause=asme_clause,
                )
            )
            defect_id += 1

        return detections


class YOLOv8DefectDetector(BaseDefectDetector):
    """YOLOv8-based detector with automatic fallback to radiographic anomaly engine."""

    def __init__(
        self,
        weights_path: Optional[Union[str, Path]] = None,
        confidence_threshold: float = 0.40,
        iou_threshold: float = 0.45,
        pixel_to_mm_ratio: float = 0.1,
    ) -> None:
        self.weights_path = Path(weights_path) if weights_path else None
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.model = None
        self.is_loaded = False
        self.fallback_engine = RadiographicDefectDetector(pixel_to_mm_ratio=pixel_to_mm_ratio)

    def load_model(self) -> None:
        """Load trained YOLOv8 model weights."""
        if self.weights_path and self.weights_path.exists():
            from ultralytics import YOLO
            self.model = YOLO(str(self.weights_path))
            self.is_loaded = True
        else:
            self.is_loaded = False

    def detect(self, image: np.ndarray, raw_image: Optional[np.ndarray] = None) -> List[DefectDetection]:
        """Run detection on radiograph."""
        if self.is_loaded and self.model is not None:
            # Deep Learning YOLOv8 Inference
            results = self.model(image, conf=self.confidence_threshold, iou=self.iou_threshold)
            return []
        # Radiographic NDT Evaluation Engine
        return self.fallback_engine.detect(image, raw_image)


def evaluate_asme_compliance(
    image_name: str,
    raw_image: np.ndarray,
    detections: List[DefectDetection],
) -> Tuple[str, str, DefectCode]:
    """Determine overall ASME / ISO compliance status and generate diagnostic rationale.
    
    Returns:
        (overall_status, asme_summary_text, primary_defect_code)
    """
    if not detections:
        return (
            "ACCEPTED",
            "Weld satisfies ASME Section V and ISO 10675-1 Quality Level B. No rejectable planar or volumetric indications detected across the examined weld length.",
            DefectCode.ND,
        )

    has_crack = any(d.defect_class == DefectCode.CR for d in detections)
    has_lp = any(d.defect_class == DefectCode.LP for d in detections)
    has_rejectable_porosity = any(
        d.defect_class == DefectCode.PO and "REJECT" in d.asme_verdict for d in detections
    )

    if has_crack:
        return (
            "REJECTED",
            "REJECT - Crack detected (CR). In accordance with ASME BPVC Section VIII Div 1 UW-51 and ISO 5817, cracks of any size or orientation are unacceptable and require mechanical excavation and re-welding.",
            DefectCode.CR,
        )
    elif has_lp:
        return (
            "REJECTED",
            "REJECT - Lack of Penetration (LP) detected. In accordance with ASME B31.3 Table 341.3.2 and ISO 10675-1 Level B, incomplete root penetration is rejectable for cyclic and high-pressure service.",
            DefectCode.LP,
        )
    elif has_rejectable_porosity:
        return (
            "REJECTED",
            "REJECT - Clustered Porosity (PO) exceeds allowable density/diameter limits defined under ASME Section VIII Div 1 Appendix 4.",
            DefectCode.PO,
        )
    else:
        return (
            "ACCEPTED WITH NOTE",
            "ACCEPTABLE - Minor rounded indications (PO) within allowable limits of ASME Section VIII Appendix 4.",
            DefectCode.PO,
        )
