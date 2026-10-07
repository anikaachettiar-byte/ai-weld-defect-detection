"""Visualization utilities for radiographic weld inspection images and contrast analysis."""

from typing import Optional, Tuple
import cv2
import matplotlib.pyplot as plt
import numpy as np


def plot_histogram_comparison(
    raw_img: np.ndarray,
    enhanced_img: np.ndarray,
    title: str = "Radiographic Histogram & CDF Contrast Profile",
) -> plt.Figure:
    """Generate Matplotlib figure comparing pixel intensity histograms and CDFs before and after CLAHE.
    
    Args:
        raw_img: Original grayscale radiograph.
        enhanced_img: CLAHE preprocessed radiograph.
        title: Overall plot title.
        
    Returns:
        Matplotlib Figure object.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), dpi=100)
    fig.patch.set_facecolor('#1e1e24')

    for ax, img, label, color in [
        (axes[0], raw_img, "Original Radiograph", "#38bdf8"),
        (axes[1], enhanced_img, "CLAHE Enhanced", "#34d399"),
    ]:
        ax.set_facecolor('#111827')
        hist, bins = np.histogram(img.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_normalized = cdf * float(hist.max()) / (cdf.max() + 1e-8)

        # Plot Histogram
        ax.hist(img.flatten(), 256, [0, 256], color=color, alpha=0.65, label="Intensity Freq")
        # Plot CDF
        ax.plot(cdf_normalized, color="#f59e0b", linewidth=1.5, linestyle="--", label="Normalized CDF")

        ax.set_title(label, color="#f3f4f6", fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("Pixel Intensity [0-255]", color="#9ca3af", fontsize=9)
        ax.set_ylabel("Pixel Count", color="#9ca3af", fontsize=9)
        ax.tick_params(colors="#9ca3af", labelsize=8)
        ax.grid(True, linestyle=":", alpha=0.3, color="#4b5563")
        ax.set_xlim([0, 256])
        ax.legend(loc="upper left", facecolor="#1f2937", edgecolor="#374151", labelcolor="#e5e7eb", fontsize=8)

        for spine in ax.spines.values():
            spine.set_color("#374151")

    fig.suptitle(title, color="#f9fafb", fontsize=12, fontweight="bold", y=0.98)
    plt.tight_layout()
    return fig


def plot_cross_section_profile(
    raw_img: np.ndarray,
    enhanced_img: np.ndarray,
    line_y: Optional[int] = None,
) -> plt.Figure:
    """Plot horizontal pixel intensity profile across the weld seam at line_y."""
    h, w = raw_img.shape[:2]
    if line_y is None or line_y < 0 or line_y >= h:
        line_y = h // 2

    raw_profile = raw_img[line_y, :]
    enh_profile = enhanced_img[line_y, :]

    fig, ax = plt.subplots(figsize=(10, 3.2), dpi=100)
    fig.patch.set_facecolor('#1e1e24')
    ax.set_facecolor('#111827')

    ax.plot(raw_profile, label="Original Profile", color="#38bdf8", alpha=0.7, linewidth=1.2)
    ax.plot(enh_profile, label="CLAHE Profile", color="#34d399", linewidth=1.5)

    ax.set_title(f"Horizontal Intensity Slice at Y = {line_y} px (Across Weld/Defect)", color="#f3f4f6", fontsize=11, fontweight="bold")
    ax.set_xlabel("X Coordinate (pixels)", color="#9ca3af", fontsize=9)
    ax.set_ylabel("Gray Value (0-255)", color="#9ca3af", fontsize=9)
    ax.tick_params(colors="#9ca3af", labelsize=8)
    ax.grid(True, linestyle=":", alpha=0.3, color="#4b5563")
    ax.set_xlim([0, w])
    ax.set_ylim([0, 260])
    ax.legend(loc="upper right", facecolor="#1f2937", edgecolor="#374151", labelcolor="#e5e7eb", fontsize=8)

    for spine in ax.spines.values():
        spine.set_color("#374151")

    plt.tight_layout()
    return fig


def create_side_by_side_comparison(
    raw_img: np.ndarray,
    enhanced_img: np.ndarray,
    diff_map: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Create a side-by-side composite image with labels."""
    h, w = raw_img.shape[:2]

    # Convert to 3-channel for colored annotations if needed
    raw_bgr = cv2.cvtColor(raw_img, cv2.COLOR_GRAY2BGR) if raw_img.ndim == 2 else raw_img.copy()
    enh_bgr = cv2.cvtColor(enhanced_img, cv2.COLOR_GRAY2BGR) if enhanced_img.ndim == 2 else enhanced_img.copy()

    # Draw titles
    cv2.putText(raw_bgr, "ORIGINAL RADIOGRAPH", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 180, 255), 2)
    cv2.putText(enh_bgr, "CLAHE ENHANCED", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 220, 100), 2)

    if diff_map is not None:
        # Colormap difference map (e.g. Inferno or Jet)
        diff_color = cv2.applyColorMap(diff_map, cv2.COLORMAP_INFERNO)
        cv2.putText(diff_color, "CONTRAST DIFFERENCE", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        composite = np.hstack([raw_bgr, enh_bgr, diff_color])
    else:
        composite = np.hstack([raw_bgr, enh_bgr])

    return composite


def draw_defect_annotations(
    image: np.ndarray,
    detections: list,
    show_masks: bool = True,
    show_boxes: bool = True,
) -> np.ndarray:
    """Overlay detected defect bounding boxes, masks, and ASME tags onto radiograph.
    
    Returns:
        RGB image formatted for Streamlit or report display.
    """
    if image.ndim == 2:
        canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        canvas = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.shape[2] == 3 else image.copy()

    # Color palette (RGB)
    color_map = {
        "CR": (239, 68, 68),    # Red (Critical Crack)
        "LP": (249, 115, 22),   # Orange (Lack of Penetration)
        "PO": (234, 179, 8),    # Yellow (Porosity)
        "ND": (34, 197, 94),    # Green (No Defect)
    }

    overlay = canvas.copy()

    for det in detections:
        code_str = det.defect_class.value if hasattr(det.defect_class, "value") else str(det.defect_class)
        color = color_map.get(code_str, (56, 189, 248))

        # 1. Draw segmentation mask overlay if present
        if show_masks and det.segmentation_mask is not None:
            mask = det.segmentation_mask
            overlay[mask > 0] = color

        # 2. Draw bounding box and label
        if show_boxes:
            b = det.bounding_box
            x1, y1, x2, y2 = int(b.x1), int(b.y1), int(b.x2), int(b.y2)

            # Draw rectangle
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

            # Build label text
            m = det.measurements
            if m and m.length_mm is not None:
                dim_str = f" | {m.length_mm:.1f}x{m.width_mm:.1f}mm"
            else:
                dim_str = ""

            label = f"[{code_str}] {det.confidence*100:.0f}%{dim_str}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)

            # Label banner
            banner_y1 = max(0, y1 - th - 6)
            banner_y2 = y1
            cv2.rectangle(canvas, (x1, banner_y1), (x1 + tw + 8, banner_y2), color, -1)
            cv2.putText(
                canvas,
                label,
                (x1 + 4, banner_y2 - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    # Blend mask overlay
    if show_masks:
        cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0, canvas)

    return canvas

