"""U-Net with ResNet34 Encoder Defect Segmentation module interface (Stub for Phase 3)."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np


class BaseDefectSegmenter(ABC):
    """Abstract Base Class for defect pixel-level segmentation."""

    @abstractmethod
    def segment(self, image: np.ndarray, bbox: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """Segment defect pixels, generating binary or multiclass mask."""
        pass


class UNetResNet34Segmenter(BaseDefectSegmenter):
    """U-Net architecture with pretrained ResNet34 backbone for fine defect contour segmentation.
    
    Status:
        [STUB] - Interface prepared for model integration in Phase 3.
    """

    def __init__(
        self,
        weights_path: Optional[Union[str, Path]] = None,
        input_size: Tuple[int, int] = (512, 512),
        encoder_name: str = "resnet34",
    ) -> None:
        self.weights_path = Path(weights_path) if weights_path else None
        self.input_size = input_size
        self.encoder_name = encoder_name
        self.model = None
        self.is_loaded = False

    def load_model(self) -> None:
        """Load U-Net ResNet34 checkpoint weights."""
        raise NotImplementedError(
            "U-Net ResNet34 segmentation model will be integrated in Phase 3."
        )

    def segment(self, image: np.ndarray, bbox: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """Generate binary segmentation mask of defect indications.
        
        Args:
            image: Preprocessed radiograph.
            bbox: Optional bounding box (x1, y1, x2, y2) to crop Region of Interest (ROI).
            
        Returns:
            Binary mask (H, W) where pixel > 0 indicates defect territory.
        """
        h, w = image.shape[:2]
        return np.zeros((h, w), dtype=np.uint8)
