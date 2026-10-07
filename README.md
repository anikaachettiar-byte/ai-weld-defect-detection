# AI-Assisted NDT Radiographic Weld Defect Inspection System

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Phase%201%20Prototype%20(Active)-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-orange.svg)]()

An automated Non-Destructive Testing (NDT) computer vision pipeline designed for radiographic weld inspection. The system processes industrial X-ray radiographs to detect, classify, segment, and quantify weld anomalies according to international standards (ASME Section V / ISO 17636).

---

## 📌 Project Overview & End-to-End Vision

| Stage | Module | Status | Description |
| :--- | :--- | :---: | :--- |
| **Stage 1** | **Image Ingestion & I/O** | ✅ **Operational** | Ingests 8-bit/16-bit radiographs (PNG, TIFF, JPEG) and includes a realistic synthetic radiographic test generator. |
| **Stage 2** | **CLAHE & Denoising** | ✅ **Operational** | Contrast Limited Adaptive Histogram Equalization with edge-preserving Bilateral and Non-Local Means (NLM) filtering. |
| **Stage 3** | **YOLOv8 Defect Detection** | ⏳ *Staged (Phase 2)* | Localizes defects into 4 classes: Cracks (**CR**), Porosity (**PO**), Lack of Penetration (**LP**), and No Defect (**ND**). |
| **Stage 4** | **U-Net ResNet34 Segmentation**| ⏳ *Staged (Phase 3)* | Generates pixel-accurate binary/multiclass masks for defect boundaries. |
| **Stage 5** | **Morphometric Quantification**| ⏳ *Staged (Phase 4)* | Calculates length, width, area, aspect ratio, centroid, and orientation angle in physical (mm) units. |
| **Stage 6** | **Explainability (Grad-CAM)** | ⏳ *Staged (Phase 5)* | Visual attribution heatmaps explaining model decisions to NDT Level II/III inspectors. |
| **Stage 7** | **Web Inspection Dashboard** | ✅ **Operational** | Interactive Streamlit interface with real-time parameter tuning, intensity profiling, and metric calculation. |

---

## 🔬 Radiographic Physics & The Role of CLAHE

In industrial radioscopic inspection:
- **Base Metal & Weld Seam:** Thicker steel absorbs more ionizing radiation, creating lighter regions, while thinner or less dense sections appear darker.
- **Defects as Attenuation Deficits:** Cracks, lack of penetration, and gas pores represent missing material, appearing as subtle, low-contrast dark indications that can easily be obscured by background noise.
- **Why Standard Equalization Fails:** Global histogram equalization over-amplifies background quantum mottle and film grain, washing out critical defect boundaries.
- **Why CLAHE Works:** CLAHE divides the radiograph into local contextual tiles (e.g. $8 \times 8$), calculates local histograms, clips them at a predefined threshold (`clip_limit`) to prevent noise over-amplification, and smoothly redistributes pixel intensities via bilinear interpolation.

---

## 📁 Modular Project Structure

```
mini project/
├── configs/
│   └── default_config.yaml         # Central YAML configuration (CLAHE, filters, models)
├── data/
│   ├── samples/                    # Synthetic radiographic test specimens (CR, PO, LP, ND)
│   └── uploads/                    # Temporary storage for user-uploaded radiographs
├── outputs/                        # Saved CLI artifacts (enhanced images, difference maps)
├── src/
│   ├── core/
│   │   ├── __init__.py
│   │   └── types.py                # Dataclasses (PreprocessedArtifacts, DefectCode, etc.)
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── clahe.py                # CLAHEPreprocessor engine
│   │   ├── denoising.py            # Bilateral, Gaussian, Median, and NLM filters
│   │   └── pipeline.py             # PreprocessingPipeline orchestrator & metrics
│   ├── detection/
│   │   ├── __init__.py
│   │   └── detector.py             # YOLOv8DefectDetector modular stub interface
│   ├── segmentation/
│   │   ├── __init__.py
│   │   └── segmenter.py            # UNetResNet34Segmenter modular stub interface
│   ├── measurement/
│   │   ├── __init__.py
│   │   └── analyzer.py             # DefectMorphologyAnalyzer (length, width, area, angle)
│   ├── explainability/
│   │   ├── __init__.py
│   │   └── gradcam.py              # GradCAMExplainer modular stub interface
│   └── utils/
│       ├── __init__.py
│       ├── image_io.py             # Radiograph loader, saver, and synthetic specimen generator
│       └── visualization.py        # Intensity profiles, histograms, and side-by-side comps
├── web_app/
│   └── app.py                      # Interactive Streamlit Web UI
├── tests/
│   ├── __init__.py
│   └── test_preprocessing.py       # Pytest unit test suite (7/7 tests passing)
├── run_app.py                      # Launcher for the Web UI
├── run_cli.py                      # Command-line interface for batch/quick CLAHE testing
├── requirements.txt                # Core & staged dependencies
└── README.md
```

---

## 🚀 Getting Started

### 1. Environment Setup
The project uses a dedicated Python 3.11 virtual environment. If not already active:

```powershell
# Activate virtual environment
.venv\Scripts\activate
```

Install core dependencies (if configuring a new machine):
```bash
pip install -r requirements.txt
```

---

## 💻 Running the Application

### Option A: Launch Interactive Web Dashboard
Run the launcher script:
```powershell
python run_app.py
```
Or run directly with Streamlit:
```powershell
.venv\Scripts\streamlit run web_app\app.py
```
Open your browser at `http://localhost:8501`.

#### Features in the Web Dashboard:
1. **Specimen Selector:** Switch between built-in standard radiographic specimens:
   - **Crack Defect (CR):** Jagged transverse/longitudinal fissure.
   - **Porosity (PO):** Cluster of spherical gas void cavities.
   - **Lack of Penetration (LP):** Continuous linear root joint groove.
   - **Sound Weld (ND):** Clean, defect-free joint.
   - **Custom Upload:** Upload any real industrial weld radiograph (PNG, JPG, TIFF, BMP).
2. **Real-time CLAHE Sliders:** Interactively tune `clip_limit` (1.0 - 8.0) and `tile_grid_size` (4x4 to 32x32).
3. **Edge-Preserving Denoising:** Toggle and tune Bilateral Filtering or Non-Local Means (NLM).
4. **Visual Difference Map:** High-contrast color-mapped visualization (`INFERNO`) isolating enhanced structural details.
5. **Horizontal Cross-Section Intensity Profile:** Select any scan line across the weld seam to observe how CLAHE deepens defect valleys.
6. **Quantitative Contrast Metrics:** Live feedback for Contrast Improvement Ratio, Shannon Entropy, RMS Contrast (Std Dev), and Dynamic Range Utilization.

---

### Option B: Run via Command Line (CLI)
Process any radiograph and output enhanced composites directly to disk:

```powershell
# Process sample crack image with default parameters
python run_cli.py

# Or process a custom radiograph with custom CLAHE settings
python run_cli.py --input path/to/weld.png --clip-limit 3.0 --grid-size 16 --output-dir outputs/
```

Outputs generated:
- `*_clahe_enhanced.png`: High-contrast processed radiograph.
- `*_diff_map.png`: Feature difference heatmap.
- `*_comparison.png`: Side-by-side labeled composite.

---

### Option C: Run Automated Unit Tests
Run the test suite to verify all modules:
```powershell
.venv\Scripts\python.exe -m pytest tests/ -v
```

---

## 📋 Defect Classification Taxonomy

| Code | Defect Name | Radiographic Appearance | Severity |
| :---: | :--- | :--- | :---: |
| **CR** | **Crack** | Dark, irregular, jagged fine line with sharp tips. | **Critical** (Rejectable) |
| **PO** | **Porosity** | Dark circular or oval voids, isolated or in dense clusters. | **Moderate** (Threshold-based) |
| **LP** | **Lack of Penetration** | Continuous or intermittent dark straight line along the weld root. | **Critical** (Rejectable) |
| **ND** | **No Defect** | Uniform weld reinforcement and sound parent metal fusion. | **Acceptable** |

---

## 🗺️ Next Steps & Roadmap

1. **Phase 2 (Detection):** Train or integrate YOLOv8 on labelled NDT datasets (e.g. GDXray / custom RT datasets) for bounding box localization (`src/detection/detector.py`).
2. **Phase 3 (Segmentation):** Connect U-Net with ResNet34 backbone (`src/segmentation/segmenter.py`) for sub-millimeter contour extraction.
3. **Phase 4 (Quantification):** Wire contour outputs into `DefectMorphologyAnalyzer` to automatically calculate ASME compliance length/width thresholds.
4. **Phase 5 (Grad-CAM):** Generate attribution heatmaps overlaying convolutional feature maps to assist certified NDT inspectors.
