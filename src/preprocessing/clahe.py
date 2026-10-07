"""Contrast Limited Adaptive Histogram Equalization (CLAHE) module for radiographic X-ray images."""

from typing import Tuple, Union
import cv2
import numpy as np


class CLAHEPreprocessor:
    """Enhances radiographic weld images using Contrast Limited Adaptive Histogram Equalization.
    
    Radiographic X-ray images often suffer from low contrast between sound metal, weld beads,
    and fine defect indications (such as tight cracks or small gas pores). Standard global histogram
    equalization frequently over-amplifies background quantum noise and washes out critical regions.
    CLAHE divides the image into contextual contextual tiles (grid), calculates local histograms,
    clips the histogram at a predetermined threshold (clip_limit) to prevent noise over-amplification,
    and redistributes the clipped pixels before bilinear interpolation between adjacent tiles.
    """

    def __init__(
        self,
        clip_limit: float = 2.5,
        tile_grid_size: Tuple[int, int] = (8, 8),
    ) -> None:
        """Initialize the CLAHE processor.
        
        Args:
            clip_limit: Threshold for contrast limiting. Higher values increase contrast
                        at the expense of potential noise amplification. Typically 1.5 - 4.0.
            tile_grid_size: Number of tiles in [horizontal, vertical] directions (e.g. (8, 8)).
        """
        self.clip_limit = self._validate_clip_limit(clip_limit)
        self.tile_grid_size = self._validate_tile_grid_size(tile_grid_size)
        self._clahe_cv = cv2.createCLAHE(
            clipLimit=float(self.clip_limit),
            tileGridSize=self.tile_grid_size,
        )

    @staticmethod
    def _validate_clip_limit(clip_limit: float) -> float:
        """Validate the clip limit parameter."""
        val = float(clip_limit)
        if val <= 0.0:
            raise ValueError(f"clip_limit must be > 0, got {val}")
        return val

    @staticmethod
    def _validate_tile_grid_size(tile_grid_size: Tuple[int, int]) -> Tuple[int, int]:
        """Validate tile grid dimensions."""
        if not (isinstance(tile_grid_size, (tuple, list)) and len(tile_grid_size) == 2):
            raise ValueError(f"tile_grid_size must be a 2-element tuple/list, got {tile_grid_size}")
        gx, gy = int(tile_grid_size[0]), int(tile_grid_size[1])
        if gx < 1 or gy < 1:
            raise ValueError(f"tile_grid_size dimensions must be >= 1, got ({gx}, {gy})")
        return (gx, gy)

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE to an input image.
        
        Args:
            image: Input radiographic image as numpy array (grayscale uint8, uint16, or 3-channel RGB/BGR).
            
        Returns:
            Contrast-enhanced image as uint8 numpy array.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Expected numpy.ndarray, got {type(image)}")
        if image.size == 0:
            raise ValueError("Input image is empty.")

        # Handle 16-bit radiographic images (common in NDT digital radiography / DICOM)
        working_img = image.copy()
        if working_img.dtype == np.uint16:
            # Rescale 16-bit to 8-bit dynamic range [0, 255]
            working_img = cv2.normalize(working_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        elif working_img.dtype != np.uint8:
            # Normalize float/int formats to uint8
            working_img = cv2.normalize(working_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Grayscale single-channel image
        if working_img.ndim == 2:
            return self._clahe_cv.apply(working_img)

        # 3-channel image: operate on Luminance channel to preserve color balance
        if working_img.ndim == 3 and working_img.shape[2] == 3:
            lab = cv2.cvtColor(working_img, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            enhanced_l = self._clahe_cv.apply(l_channel)
            merged_lab = cv2.merge([enhanced_l, a_channel, b_channel])
            return cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)

        # 1-channel with trailing dimension (H, W, 1)
        if working_img.ndim == 3 and working_img.shape[2] == 1:
            enhanced = self._clahe_cv.apply(working_img[:, :, 0])
            return enhanced

        raise ValueError(f"Unsupported image shape for CLAHE: {image.shape}")

    def update_parameters(self, clip_limit: float, tile_grid_size: Tuple[int, int]) -> None:
        """Update CLAHE parameters dynamically."""
        self.clip_limit = self._validate_clip_limit(clip_limit)
        self.tile_grid_size = self._validate_tile_grid_size(tile_grid_size)
        self._clahe_cv = cv2.createCLAHE(
            clipLimit=float(self.clip_limit),
            tileGridSize=self.tile_grid_size,
        )
