"""Test suite for radiographic image preprocessing and CLAHE pipeline."""

import numpy as np
import pytest

from src.core.types import DefectCode
from src.measurement.analyzer import DefectMorphologyAnalyzer
from src.preprocessing.clahe import CLAHEPreprocessor
from src.preprocessing.denoising import DenoiseMethod, DenoiseProcessor
from src.preprocessing.pipeline import PreprocessingPipeline
from src.utils.image_io import generate_synthetic_weld_sample, load_radiograph, save_radiograph


def test_clahe_initialization():
    """Verify CLAHEPreprocessor initializes with valid parameters and rejects invalid ones."""
    proc = CLAHEPreprocessor(clip_limit=3.0, tile_grid_size=(8, 8))
    assert proc.clip_limit == 3.0
    assert proc.tile_grid_size == (8, 8)

    with pytest.raises(ValueError):
        CLAHEPreprocessor(clip_limit=-1.0)

    with pytest.raises(ValueError):
        CLAHEPreprocessor(tile_grid_size=(0, 8))


def test_clahe_application_grayscale():
    """Verify CLAHE on synthetic grayscale radiographic image."""
    img = generate_synthetic_weld_sample(defect_type="crack", width=320, height=180)
    assert img.ndim == 2
    assert img.dtype == np.uint8

    proc = CLAHEPreprocessor(clip_limit=2.5, tile_grid_size=(8, 8))
    enhanced = proc.apply(img)

    assert enhanced.shape == img.shape
    assert enhanced.dtype == np.uint8
    # Standard deviation (contrast) should typically increase or redistribute
    assert np.std(enhanced) > 0


def test_clahe_application_color():
    """Verify CLAHE handles 3-channel input by processing luminance."""
    img_gray = generate_synthetic_weld_sample(defect_type="porosity", width=300, height=150)
    img_color = np.stack([img_gray]*3, axis=-1)

    proc = CLAHEPreprocessor(clip_limit=2.0, tile_grid_size=(4, 4))
    enhanced = proc.apply(img_color)

    assert enhanced.shape == img_color.shape
    assert enhanced.dtype == np.uint8


def test_denoising_methods():
    """Verify all edge-preserving denoising filters execute cleanly."""
    img = generate_synthetic_weld_sample(defect_type="lack_of_penetration", width=200, height=100)

    methods = [
        (DenoiseMethod.BILATERAL, {"d": 5, "sigma_color": 30.0, "sigma_space": 30.0}),
        (DenoiseMethod.GAUSSIAN, {"kernel_size": (3, 3), "sigma_x": 1.0}),
        (DenoiseMethod.MEDIAN, {"kernel_size": 3}),
        (DenoiseMethod.NLM, {"h": 8.0, "template_window_size": 5, "search_window_size": 15}),
    ]

    for method, params in methods:
        denoiser = DenoiseProcessor(method=method, params=params)
        denoised = denoiser.apply(img)
        assert denoised.shape == img.shape
        assert denoised.dtype == np.uint8


def test_preprocessing_pipeline():
    """Verify full end-to-end preprocessing pipeline with metrics computation."""
    img = generate_synthetic_weld_sample(defect_type="crack", width=400, height=200)

    pipeline = PreprocessingPipeline(
        clahe_clip_limit=2.5,
        clahe_tile_grid_size=(8, 8),
        enable_denoising=True,
        denoise_method="bilateral",
        enable_normalization=True,
    )

    artifacts = pipeline.process(img)

    assert artifacts.raw_image is not None
    assert artifacts.denoised_image is not None
    assert artifacts.clahe_image is not None
    assert artifacts.final_image is not None
    assert artifacts.difference_map is not None
    assert artifacts.metrics is not None

    m = artifacts.metrics
    assert m.enhanced_std > 0
    assert m.contrast_improvement_ratio > 0
    assert 0 <= m.dynamic_range_utilization <= 1.0


def test_synthetic_sample_generation():
    """Verify all four defect types generate valid images."""
    for def_type in ["crack", "porosity", "lack_of_penetration", "clean"]:
        sample = generate_synthetic_weld_sample(defect_type=def_type, width=320, height=160)
        assert sample.shape == (160, 320)
        assert sample.dtype == np.uint8
        assert np.min(sample) >= 0
        assert np.max(sample) <= 255


def test_morphology_analyzer():
    """Verify morphometric analysis on synthetic binary mask."""
    # Create a synthetic binary mask with a simulated crack (elongated line)
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[95:105, 50:150] = 255  # 10x100 rectangular defect

    analyzer = DefectMorphologyAnalyzer(pixel_to_mm_ratio=0.1)
    measurements = analyzer.analyze_mask(mask)

    assert len(measurements) == 1
    m = measurements[0]
    assert m.length_px > 0
    assert m.width_px > 0
    assert m.area_px > 0
    assert m.length_mm is not None


def test_radiographic_defect_detector_and_asme_compliance():
    """Verify RadiographicDefectDetector and ASME evaluation on clean vs crack welds."""
    from src.detection.detector import RadiographicDefectDetector, evaluate_asme_compliance

    detector = RadiographicDefectDetector(pixel_to_mm_ratio=0.1)
    pipeline = PreprocessingPipeline()

    # 1. Test clean weld -> ND, ACCEPTED
    clean_sample = generate_synthetic_weld_sample(defect_type="clean", width=400, height=200)
    art_clean = pipeline.process(clean_sample)
    clean_dets = detector.detect(art_clean.final_image, art_clean.raw_image)
    verdict_clean, _, primary_clean = evaluate_asme_compliance("clean.png", art_clean.raw_image, clean_dets)
    assert verdict_clean == "ACCEPTED"
    assert primary_clean == DefectCode.ND

    # 2. Test crack weld -> CR, REJECTED
    crack_sample = generate_synthetic_weld_sample(defect_type="crack", width=400, height=200)
    art_crack = pipeline.process(crack_sample)
    crack_dets = detector.detect(art_crack.final_image, art_crack.raw_image)
    assert len(crack_dets) > 0
    verdict_crack, summary_crack, primary_crack = evaluate_asme_compliance("crack.png", art_crack.raw_image, crack_dets)
    assert verdict_crack == "REJECTED"
    assert primary_crack in [DefectCode.CR, DefectCode.LP]
    assert "ASME" in summary_crack

