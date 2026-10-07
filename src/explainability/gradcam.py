"""Grad-CAM Explainability module interface (Stub for Phase 5)."""

from typing import Optional, Union
import cv2
import numpy as np


class GradCAMExplainer:
    """Generates visual explanation heatmaps highlighting regions influencing classification decisions.
    
    Status:
        [STUB] - Interface prepared for deep network Grad-CAM integration in Phase 5.
    """

    def __init__(self, target_layer_name: str = "layer4") -> None:
        self.target_layer_name = target_layer_name

    def generate_heatmap(
        self,
        image: np.ndarray,
        target_class_id: Optional[int] = None,
    ) -> np.ndarray:
        """Generate Grad-CAM activation heatmap overlay.
        
        Args:
            image: Preprocessed radiograph.
            target_class_id: Class to explain (e.g. CR, PO, LP).
            
        Returns:
            Color heatmap overlay (H, W, 3).
        """
        # When models are loaded in Phase 5, calculates gradients of target score wrt feature activation map
        h, w = image.shape[:2]
        dummy_heatmap = np.zeros((h, w, 3), dtype=np.uint8)
        return dummy_heatmap
