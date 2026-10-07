"""Preprocessing pipeline orchestrating radiographic noise reduction, CLAHE, and metrics computation."""

from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np

from src.core.types import PreprocessedArtifacts, PreprocessingMetrics
from src.preprocessing.clahe import CLAHEPreprocessor
from src.preprocessing.denoising import DenoiseMethod, DenoiseProcessor


class PreprocessingPipeline:
    """Configurable pipeline for preprocessing radiographic weld X-ray images."""

    def __init__(
        self,
        clahe_clip_limit: float = 2.5,
        clahe_tile_grid_size: Tuple[int, int] = (8, 8),
        enable_denoising: bool = True,
        denoise_method: str = "bilateral",
        denoise_params: Optional[Dict[str, Any]] = None,
        enable_normalization: bool = True,
        norm_range: Tuple[int, int] = (0, 255),
    ) -> None:
        """Initialize the preprocessing pipeline.
        
        Args:
            clahe_clip_limit: CLAHE contrast clip limit.
            clahe_tile_grid_size: CLAHE contextual tile grid dimensions.
            enable_denoising: Whether to apply edge-preserving noise reduction prior to CLAHE.
            denoise_method: Filtering method ("bilateral", "gaussian", "median", "nlm").
            denoise_params: Parameter dictionary for denoising.
            enable_normalization: Whether to normalize dynamic range.
            norm_range: Target (min, max) dynamic range.
        """
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid_size = clahe_tile_grid_size
        self.enable_denoising = enable_denoising
        self.denoise_method = denoise_method
        self.denoise_params = denoise_params or {}
        self.enable_normalization = enable_normalization
        self.norm_range = norm_range

        self.clahe_processor = CLAHEPreprocessor(
            clip_limit=clahe_clip_limit,
            tile_grid_size=clahe_tile_grid_size,
        )

        if self.enable_denoising:
            self.denoise_processor = DenoiseProcessor(
                method=DenoiseMethod(denoise_method),
                params=self.denoise_params,
            )
        else:
            self.denoise_processor = None

    def process(self, image: np.ndarray) -> PreprocessedArtifacts:
        """Execute the full preprocessing pipeline on a radiographic image.
        
        Args:
            image: Raw input image (grayscale or RGB/BGR).
            
        Returns:
            PreprocessedArtifacts containing all intermediate and final outputs plus metrics.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Expected numpy.ndarray, got {type(image)}")

        # Convert to grayscale if color, since RT radiographs are single-channel transmission images
        if image.ndim == 3 and image.shape[2] == 3:
            gray_raw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.ndim == 3 and image.shape[2] == 1:
            gray_raw = image[:, :, 0]
        else:
            gray_raw = image.copy()

        # Step 1: Denoising (optional edge-preserving)
        if self.enable_denoising and self.denoise_processor is not None:
            denoised = self.denoise_processor.apply(gray_raw)
        else:
            denoised = gray_raw.copy()

        # Step 2: CLAHE Contrast Enhancement
        clahe_enhanced = self.clahe_processor.apply(denoised)

        # Step 3: Intensity Normalization (optional)
        if self.enable_normalization:
            final_img = cv2.normalize(
                clahe_enhanced,
                None,
                alpha=self.norm_range[0],
                beta=self.norm_range[1],
                norm_type=cv2.NORM_MINMAX,
                dtype=cv2.CV_8U,
            )
        else:
            final_img = clahe_enhanced

        # Step 4: Compute difference map (shows highlighted structural defect details)
        diff_map = cv2.absdiff(final_img, gray_raw)

        # Step 5: Calculate quantitative quality metrics
        metrics = self._calculate_metrics(gray_raw, final_img)

        return PreprocessedArtifacts(
            raw_image=gray_raw,
            denoised_image=denoised,
            clahe_image=clahe_enhanced,
            final_image=final_img,
            difference_map=diff_map,
            metrics=metrics,
            applied_parameters={
                "clahe_clip_limit": self.clahe_clip_limit,
                "clahe_tile_grid_size": self.clahe_tile_grid_size,
                "enable_denoising": self.enable_denoising,
                "denoise_method": self.denoise_method,
                "enable_normalization": self.enable_normalization,
            },
        )

    @staticmethod
    def _calculate_metrics(orig: np.ndarray, enhanced: np.ndarray) -> PreprocessingMetrics:
        """Compute statistical and contrast improvement metrics."""
        orig_mean = float(np.mean(orig))
        orig_std = float(np.std(orig))
        enh_mean = float(np.mean(enhanced))
        enh_std = float(np.std(enhanced))

        # Shannon entropy: measures information richness / detail distribution
        orig_entropy = PreprocessingPipeline._compute_entropy(orig)
        enh_entropy = PreprocessingPipeline._compute_entropy(enhanced)

        # Contrast improvement ratio (ratio of standard deviation)
        cir = (enh_std / orig_std) if orig_std > 1e-6 else 1.0

        # Dynamic range utilization: (max - min) / 255
        dyn_range = float((np.max(enhanced) - np.min(enhanced)) / 255.0)

        return PreprocessingMetrics(
            original_mean=round(orig_mean, 2),
            original_std=round(orig_std, 2),
            original_entropy=round(orig_entropy, 3),
            enhanced_mean=round(enh_mean, 2),
            enhanced_std=round(enh_std, 2),
            enhanced_entropy=round(enh_entropy, 3),
            contrast_improvement_ratio=round(cir, 3),
            dynamic_range_utilization=round(dyn_range, 3),
        )

    @staticmethod
    def _compute_entropy(img: np.ndarray) -> float:
        """Calculate Shannon entropy for an 8-bit image."""
        hist, _ = np.histogram(img.ravel(), bins=256, range=(0, 256))
        prob = hist / (img.size + 1e-10)
        prob = prob[prob > 0]
        return float(-np.sum(prob * np.log2(prob)))
