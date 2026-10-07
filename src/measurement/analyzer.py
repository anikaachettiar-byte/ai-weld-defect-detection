"""Morphometric analysis module for calculating physical and geometric defect measurements."""

from typing import List, Optional, Tuple
import cv2
import numpy as np

from src.core.types import MorphometricMeasurements


class DefectMorphologyAnalyzer:
    """Calculates quantitative defect dimensions from binary segmentation masks.
    
    Metrics:
        - Length (maximum Feret diameter or major axis of fitted ellipse)
        - Width (minor axis or minimum caliper dimension)
        - Area (total pixel count / surface area)
        - Aspect Ratio (Length / Width)
        - Centroid (defect center (X, Y) relative to weld seam)
        - Orientation (angle in degrees relative to the horizontal weld axis)
    """

    def __init__(self, pixel_to_mm_ratio: float = 0.1) -> None:
        """Initialize analyzer.
        
        Args:
            pixel_to_mm_ratio: Calibration factor (e.g. 0.1 mm per pixel).
        """
        self.pixel_to_mm = pixel_to_mm_ratio

    def analyze_mask(self, mask: np.ndarray) -> List[MorphometricMeasurements]:
        """Analyze connected components in a binary mask.
        
        Args:
            mask: 2D binary uint8 mask (0 = background, 255 = defect).
            
        Returns:
            List of MorphometricMeasurements for each detected anomaly contour.
        """
        if mask.ndim != 2 or np.sum(mask) == 0:
            return []

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        measurements: List[MorphometricMeasurements] = []

        for cnt in contours:
            area_px = float(cv2.contourArea(cnt))
            if area_px < 3.0:
                continue  # Skip negligible noise specks

            # Compute image moments for centroid
            moments = cv2.moments(cnt)
            if moments["m00"] != 0:
                cx = float(moments["m10"] / moments["m00"])
                cy = float(moments["m01"] / moments["m00"])
            else:
                cx, cy = 0.0, 0.0

            # Minimum area bounding rectangle to extract length, width, and orientation angle
            rect = cv2.minAreaRect(cnt)
            (_, _), (dim1, dim2), angle = rect
            length_px = max(dim1, dim2)
            width_px = max(min(dim1, dim2), 1.0)
            aspect_ratio = length_px / width_px

            # Calibrate to physical mm units
            length_mm = length_px * self.pixel_to_mm
            width_mm = width_px * self.pixel_to_mm
            area_mm2 = area_px * (self.pixel_to_mm ** 2)

            measurements.append(
                MorphometricMeasurements(
                    length_px=round(length_px, 2),
                    width_px=round(width_px, 2),
                    area_px=round(area_px, 2),
                    aspect_ratio=round(aspect_ratio, 2),
                    centroid=(round(cx, 1), round(cy, 1)),
                    orientation_deg=round(angle, 1),
                    length_mm=round(length_mm, 3),
                    width_mm=round(width_mm, 3),
                    area_mm2=round(area_mm2, 3),
                )
            )

        return measurements
