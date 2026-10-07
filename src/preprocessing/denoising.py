"""Denoising filter implementations optimized for radiographic X-ray images."""

from enum import Enum
from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np


class DenoiseMethod(str, Enum):
    """Supported radiographic denoising algorithms."""
    BILATERAL = "bilateral"
    GAUSSIAN = "gaussian"
    MEDIAN = "median"
    NLM = "nlm"


class DenoiseProcessor:
    """Applies edge-preserving noise reduction to radiographic images.
    
    In radiographic non-destructive testing (NDT), images typically contain quantum mottle,
    film grain, and detector noise. Preserving high-frequency crack edges and small gas pores
    is paramount, which is why Bilateral Filtering and Fast Non-Local Means (NLM) are preferred
    over destructive blurring.
    """

    def __init__(
        self,
        method: DenoiseMethod = DenoiseMethod.BILATERAL,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the DenoiseProcessor.
        
        Args:
            method: Denoising algorithm to use.
            params: Dictionary of parameters specific to the selected algorithm.
        """
        self.method = DenoiseMethod(method)
        self.params = params or {}

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply noise reduction to the input image.
        
        Args:
            image: Grayscale or BGR image (uint8).
            
        Returns:
            Denoised image.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Expected numpy.ndarray, got {type(image)}")
        if image.size == 0:
            raise ValueError("Input image is empty.")

        # Ensure image is uint8
        working_img = image
        if working_img.dtype != np.uint8:
            working_img = cv2.normalize(working_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        is_grayscale = (working_img.ndim == 2) or (working_img.ndim == 3 and working_img.shape[2] == 1)
        gray = working_img if working_img.ndim == 2 else working_img[:, :, 0]

        if self.method == DenoiseMethod.BILATERAL:
            d = int(self.params.get("d", 7))
            sigma_color = float(self.params.get("sigma_color", 50.0))
            sigma_space = float(self.params.get("sigma_space", 50.0))
            if is_grayscale:
                return cv2.bilateralFilter(gray, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
            return cv2.bilateralFilter(working_img, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)

        elif self.method == DenoiseMethod.GAUSSIAN:
            k = self.params.get("kernel_size", (5, 5))
            if isinstance(k, int):
                k = (k, k)
            kx = k[0] if k[0] % 2 != 0 else k[0] + 1
            ky = k[1] if k[1] % 2 != 0 else k[1] + 1
            sigma_x = float(self.params.get("sigma_x", 1.0))
            return cv2.GaussianBlur(working_img, (kx, ky), sigmaX=sigma_x)

        elif self.method == DenoiseMethod.MEDIAN:
            k = int(self.params.get("kernel_size", 5))
            if k % 2 == 0:
                k += 1
            return cv2.medianBlur(working_img, k)

        elif self.method == DenoiseMethod.NLM:
            h = float(self.params.get("h", 10.0))
            template_window = int(self.params.get("template_window_size", 7))
            search_window = int(self.params.get("search_window_size", 21))
            if is_grayscale:
                return cv2.fastNlMeansDenoising(
                    gray,
                    None,
                    h=h,
                    templateWindowSize=template_window,
                    searchWindowSize=search_window,
                )
            return cv2.fastNlMeansDenoisingColored(
                working_img,
                None,
                h=h,
                hColor=h,
                templateWindowSize=template_window,
                searchWindowSize=search_window,
            )

        raise ValueError(f"Unsupported denoise method: {self.method}")
