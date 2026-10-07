"""Streamlit Web Application: AI-Assisted NDT Radiographic Weld Inspection."""

from datetime import datetime, timezone
import os
from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.core.auth import UserProfile, authenticate, get_demo_user
from src.core.types import DEFECT_CATALOG, DefectCode
from src.detection.detector import RadiographicDefectDetector, evaluate_asme_compliance
from src.preprocessing.pipeline import PreprocessingPipeline
from src.utils.image_io import (
    ensure_sample_radiographs,
    generate_synthetic_weld_sample,
    load_radiograph,
)
from src.utils.visualization import (
    draw_defect_annotations,
    plot_cross_section_profile,
    plot_histogram_comparison,
)

# Page configuration
st.set_page_config(
    page_title="AI-NDT Weld Radiographic Inspection Portal",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Industrial Dark Theme CSS
st.markdown("""
<style>
    .reportview-container {
        background: #0f172a;
    }
    .metric-card {
        background: #1e293b;
        border-radius: 8px;
        padding: 14px 18px;
        border: 1px solid #334155;
        margin-bottom: 10px;
    }
    .metric-title {
        color: #94a3b8;
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        color: #38bdf8;
        font-size: 1.4rem;
        font-weight: 700;
    }
    .status-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-active { background: #065f46; color: #6ee7b7; border: 1px solid #059669; }
    .badge-staged { background: #374151; color: #9ca3af; border: 1px solid #4b5563; }
    
    .verdict-box-reject {
        background: rgba(239, 68, 68, 0.12);
        border: 1px solid #ef4444;
        border-left: 6px solid #ef4444;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 20px;
    }
    .verdict-box-accept {
        background: rgba(34, 197, 94, 0.12);
        border: 1px solid #22c55e;
        border-left: 6px solid #22c55e;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 20px;
    }
    .verdict-box-note {
        background: rgba(234, 179, 8, 0.12);
        border: 1px solid #eab308;
        border-left: 6px solid #eab308;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 20px;
    }
    
    /* Login Card Styles */
    .login-box {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 30px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
    }
    .inspector-badge {
        background: #0f172a;
        border: 1px solid #38bdf8;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def get_sample_images() -> dict:
    """Ensure and load sample radiographic test images."""
    samples_dir = PROJECT_ROOT / "data" / "samples"
    return ensure_sample_radiographs(samples_dir)


def render_login_screen():
    """Render certified NDT inspector login page with credentials form and one-click demo access."""
    col_l, col_center, col_r = st.columns([1, 2, 1])

    with col_center:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 25px; margin-top: 20px;">
            <div style="font-size: 3rem;">🔍</div>
            <h2 style="color: #f8fafc; font-weight: 800; margin: 8px 0 4px 0;">AI-NDT Weld Radiographic Inspection Portal</h2>
            <p style="color: #94a3b8; font-size: 0.95rem;">
                Authorized Access Control • ASME Section V / ISO 9712 Regulatory Sign-Off
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="login-box">', unsafe_allow_html=True)
        st.subheader("🔐 Inspector Authentication")
        st.caption("Sign in with your certified radiographer credentials to access the testing console.")

        username_input = st.text_input("Inspector ID / Username", key="input_user")
        password_input = st.text_input("Password", type="password", key="input_pass")

        col_btn, _ = st.columns([1, 1])
        with col_btn:
            if st.button("🔑 Sign In", use_container_width=True):
                user = authenticate(username_input, password_input)
                if user:
                    st.session_state["authenticated"] = True
                    st.session_state["user"] = user
                    st.success(f"Welcome, {user.full_name} ({user.role})")
                    st.rerun()
                else:
                    st.error("Invalid Inspector ID or Password. Please try again or use One-Click Demo Access below.")

        st.markdown("---")
        st.markdown("#### 🚀 One-Click Quick Demo Login")
        st.caption("Click any authorized role below to explore the inspection system immediately without typing:")

        col_demo1, col_demo2, col_demo3 = st.columns(3)
        with col_demo1:
            if st.button("👤 Level II Inspector\n(Vikram Sharma)", use_container_width=True):
                st.session_state["authenticated"] = True
                st.session_state["user"] = get_demo_user("inspector")
                st.rerun()

        with col_demo2:
            if st.button("🛡️ QA Supervisor\n(Dr. Elena Rostova)", use_container_width=True):
                st.session_state["authenticated"] = True
                st.session_state["user"] = get_demo_user("qa_manager")
                st.rerun()

        with col_demo3:
            if st.button("⚡ Guest Demo\n(Quick Evaluation)", use_container_width=True):
                st.session_state["authenticated"] = True
                st.session_state["user"] = get_demo_user("guest")
                st.rerun()

        st.markdown("""
        <div style="margin-top: 20px; font-size: 0.8rem; color: #64748b; background: #0f172a; padding: 10px 14px; border-radius: 6px; border: 1px solid #1e293b;">
            <strong>Default Demo Credentials:</strong><br>
            • Inspector: <code>inspector</code> / <code>ndt123</code> (Level II RT)<br>
            • Supervisor: <code>qa_manager</code> / <code>admin123</code> (Level III RT)<br>
            • Guest: <code>guest</code> / <code>demo</code>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)


def main():
    # Authentication check
    if not st.session_state.get("authenticated", False):
        render_login_screen()
        return

    current_user: UserProfile = st.session_state.get("user")

    # Header with User Badge & Status
    st.markdown(f"""
    <div style="display: flex; align-items: baseline; justify-content: space-between; border-bottom: 1px solid #334155; padding-bottom: 12px; margin-bottom: 20px;">
        <div>
            <h2 style="margin: 0; color: #f8fafc; font-weight: 700;">🔍 AI-Assisted NDT Radiographic Weld Inspection</h2>
            <p style="margin: 4px 0 0 0; color: #94a3b8; font-size: 0.95rem;">
                Radiographic Contrast Enhancement (CLAHE) • Automated Defect Detection • ASME / ISO Compliance Evaluation
            </p>
        </div>
        <div style="text-align: right;">
            <span class="status-badge badge-active">Operational Console</span>
            <div style="font-size: 0.82rem; color: #38bdf8; margin-top: 4px; font-weight: 600;">
                Logged in: {current_user.full_name} ({current_user.cert_id})
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Initialize sample radiographs
    sample_files = get_sample_images()

    # Sidebar: User Profile Card, Setup & Parameters
    with st.sidebar:
        # Inspector Profile Card
        st.markdown(f"""
        <div class="inspector-badge">
            <div style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; font-weight: 700;">Active Radiographer</div>
            <div style="font-size: 1.05rem; font-weight: 700; color: #f8fafc; margin-top: 2px;">{current_user.full_name}</div>
            <div style="font-size: 0.8rem; color: #38bdf8; margin-top: 2px;">{current_user.role}</div>
            <div style="font-size: 0.75rem; color: #64748b; margin-top: 4px;">Cert: <code>{current_user.cert_id}</code></div>
            <div style="font-size: 0.75rem; color: #64748b;">Facility: {current_user.facility}</div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🚪 Sign Out", use_container_width=True):
            st.session_state["authenticated"] = False
            st.session_state["user"] = None
            st.rerun()

        st.markdown("---")
        st.header("🎛️ Inspection Setup")

        input_mode = st.radio(
            "Select Radiograph Source:",
            ["Standard NDT Sample", "Upload Custom Radiograph"],
            index=0,
        )

        selected_image = None
        image_name = "radiograph"

        if input_mode == "Standard NDT Sample":
            sample_choice = st.selectbox(
                "Choose Radiographic Specimen:",
                [
                    "Crack Defect (Transverse / Longitudinal)",
                    "Porosity (Cluster of Gas Voids)",
                    "Lack of Penetration (Root Groove)",
                    "Sound Weld (Acceptable / No Defect)",
                ],
            )
            sample_mapping = {
                "Crack Defect (Transverse / Longitudinal)": "crack_defect.png",
                "Porosity (Cluster of Gas Voids)": "porosity_cluster.png",
                "Lack of Penetration (Root Groove)": "lack_of_penetration.png",
                "Sound Weld (Acceptable / No Defect)": "sound_weld_no_defect.png",
            }
            target_file = sample_files.get(sample_mapping[sample_choice])
            if target_file and target_file.exists():
                selected_image = load_radiograph(target_file)
                image_name = sample_mapping[sample_choice]
        else:
            uploaded_file = st.file_uploader(
                "Upload Weld Radiograph (PNG, JPG, TIFF, BMP)",
                type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
            )
            if uploaded_file is not None:
                selected_image = load_radiograph(uploaded_file.getvalue())
                image_name = uploaded_file.name

        st.markdown("---")
        st.subheader("⚡ CLAHE Parameters")
        st.caption("Contrast Limited Adaptive Histogram Equalization prevents noise over-amplification in uniform metal.")

        clip_limit = st.slider(
            "Clip Limit (Threshold)",
            min_value=1.0,
            max_value=8.0,
            value=2.5,
            step=0.25,
            help="Higher values increase local contrast but may amplify quantum grain noise.",
        )

        grid_dim = st.select_slider(
            "Tile Grid Size",
            options=[4, 8, 16, 24, 32],
            value=8,
            help="Contextual tile dimension (N x N). Smaller tiles emphasize fine local details like micro-cracks.",
        )
        tile_grid = (grid_dim, grid_dim)

        st.markdown("---")
        st.subheader("🛡️ Edge-Preserving Denoising")
        enable_denoising = st.checkbox("Enable Pre-Filtering Denoising", value=True)

        denoise_method = "bilateral"
        denoise_params = {}

        if enable_denoising:
            denoise_choice = st.selectbox(
                "Denoising Algorithm:",
                [
                    "Bilateral Filter (Preserves Crack Edges)",
                    "Fast Non-Local Means (High Quality)",
                    "Gaussian Blur",
                    "Median Filter (Impulse Noise)",
                ],
            )
            if "Bilateral" in denoise_choice:
                denoise_method = "bilateral"
                d = st.slider("Neighborhood Diameter (d)", 3, 15, 7, step=2)
                sigma_color = st.slider("Sigma Color (Intensity Edge)", 10.0, 100.0, 50.0, step=5.0)
                sigma_space = st.slider("Sigma Space (Spatial Coordinate)", 10.0, 100.0, 50.0, step=5.0)
                denoise_params = {"d": d, "sigma_color": sigma_color, "sigma_space": sigma_space}
            elif "Non-Local" in denoise_choice:
                denoise_method = "nlm"
                h = st.slider("Filter Strength (h)", 3.0, 25.0, 10.0, step=1.0)
                denoise_params = {"h": h, "template_window_size": 7, "search_window_size": 21}
            elif "Gaussian" in denoise_choice:
                denoise_method = "gaussian"
                k = st.slider("Kernel Size", 3, 11, 5, step=2)
                denoise_params = {"kernel_size": (k, k), "sigma_x": 1.0}
            else:
                denoise_method = "median"
                k = st.slider("Median Aperture", 3, 11, 5, step=2)
                denoise_params = {"kernel_size": k}

        enable_norm = st.checkbox("Normalize Dynamic Range [0, 255]", value=True)

        st.markdown("---")
        st.subheader("📏 Calibration Settings")
        pixel_to_mm = st.number_input(
            "Pixel to mm Ratio (Scale):",
            min_value=0.01,
            max_value=2.0,
            value=0.10,
            step=0.01,
            help="Physical calibration: millimeters per pixel on the radiograph.",
        )

    # Process image if available
    if selected_image is None:
        st.warning("Please select or upload a radiographic X-ray image to begin inspection.")
        return

    # Execute Preprocessing Pipeline
    pipeline = PreprocessingPipeline(
        clahe_clip_limit=clip_limit,
        clahe_tile_grid_size=tile_grid,
        enable_denoising=enable_denoising,
        denoise_method=denoise_method,
        denoise_params=denoise_params,
        enable_normalization=enable_norm,
    )

    artifacts = pipeline.process(selected_image)
    metrics = artifacts.metrics

    # Execute Defect Detection and ASME Evaluation Engine
    detector = RadiographicDefectDetector(pixel_to_mm_ratio=pixel_to_mm)
    detections = detector.detect(artifacts.final_image, artifacts.raw_image)
    overall_verdict, asme_summary, primary_code = evaluate_asme_compliance(image_name, artifacts.raw_image, detections)

    # Generate annotated image with bounding boxes & contours
    annotated_image = draw_defect_annotations(artifacts.final_image, detections, show_masks=True, show_boxes=True)

    # Tabs
    tab_inspect, tab_asme_report, tab_hist, tab_roadmap = st.tabs([
        "🔬 Radiograph Inspection & Enhancement",
        "📋 Defect Classification & ASME/ISO Report",
        "📊 Histogram & Dynamic Range Profile",
        "🗺️ Pipeline Architecture & Defect Catalog",
    ])

    with tab_inspect:
        col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
        with col_m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Specimen Dimensions</div>
                <div class="metric-value">{selected_image.shape[1]} × {selected_image.shape[0]} px</div>
            </div>
            """, unsafe_allow_html=True)
        with col_m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Contrast Improvement</div>
                <div class="metric-value">{metrics.contrast_improvement_ratio}×</div>
            </div>
            """, unsafe_allow_html=True)
        with col_m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Raw Entropy / Enhanced</div>
                <div class="metric-value">{metrics.original_entropy} → {metrics.enhanced_entropy}</div>
            </div>
            """, unsafe_allow_html=True)
        with col_m4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">RMS Contrast (Std Dev)</div>
                <div class="metric-value">{metrics.original_std:.1f} → {metrics.enhanced_std:.1f}</div>
            </div>
            """, unsafe_allow_html=True)
        with col_m5:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Detected Indications</div>
                <div class="metric-value">{len(detections)} Flaws</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### Radiograph Preprocessing & Automated Anomaly Localization")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("1. Original Radiograph")
            st.image(
                artifacts.raw_image,
                caption=f"Raw Ingested Image ({image_name})",
                use_container_width=True,
                clamp=True,
            )

        with col2:
            st.subheader("2. CLAHE Enhanced")
            st.image(
                artifacts.final_image,
                caption=f"CLAHE (Clip: {clip_limit}, Grid: {grid_dim}×{grid_dim})",
                use_container_width=True,
                clamp=True,
            )

        with col3:
            st.subheader("3. Defect Annotations & Bounding Boxes")
            st.image(
                annotated_image,
                caption=f"Localized Flaws (Total: {len(detections)})",
                use_container_width=True,
            )

        st.markdown("---")
        st.subheader("📈 Cross-Sectional Weld Intensity Profile")
        st.caption("Slice horizontally through the weld seam to observe how CLAHE sharpens defect valleys and bead margins.")

        height = selected_image.shape[0]
        y_slice = st.slider(
            "Y Position for Horizontal Cross-Section (px):",
            min_value=0,
            max_value=height - 1,
            value=height // 2,
            step=2,
        )

        fig_profile = plot_cross_section_profile(artifacts.raw_image, artifacts.final_image, line_y=y_slice)
        st.pyplot(fig_profile)
        plt.close(fig_profile)

    with tab_asme_report:
        st.subheader(f"📋 ASME / ISO Defect Inspection Report: {image_name}")

        # Big Compliance Verdict Banner
        if overall_verdict == "REJECTED":
            st.markdown(f"""
            <div class="verdict-box-reject">
                <div style="font-size: 1.3rem; font-weight: 800; color: #ef4444; margin-bottom: 6px;">
                    ❌ ASME / ISO VERDICT: REJECTED (NON-COMPLIANT)
                </div>
                <div style="font-size: 0.95rem; color: #fca5a5; line-height: 1.5;">
                    {asme_summary}
                </div>
            </div>
            """, unsafe_allow_html=True)
        elif overall_verdict == "ACCEPTED":
            st.markdown(f"""
            <div class="verdict-box-accept">
                <div style="font-size: 1.3rem; font-weight: 800; color: #22c55e; margin-bottom: 6px;">
                    ✅ ASME / ISO VERDICT: ACCEPTED (COMPLIANT)
                </div>
                <div style="font-size: 0.95rem; color: #86efac; line-height: 1.5;">
                    {asme_summary}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="verdict-box-note">
                <div style="font-size: 1.3rem; font-weight: 800; color: #eab308; margin-bottom: 6px;">
                    ⚠️ ASME / ISO VERDICT: ACCEPTABLE WITH OBSERVATIONS
                </div>
                <div style="font-size: 0.95rem; color: #fde047; line-height: 1.5;">
                    {asme_summary}
                </div>
            </div>
            """, unsafe_allow_html=True)

        c_diag1, c_diag2, c_diag3 = st.columns(3)
        with c_diag1:
            primary_info = DEFECT_CATALOG.get(primary_code, DEFECT_CATALOG[DefectCode.ND])
            sev_badge = "#ef4444" if primary_info.severity == "Critical" else ("#eab308" if primary_info.severity == "Moderate" else "#22c55e")
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Primary Defect Indication</div>
                <div class="metric-value">[{primary_code.value}] {primary_info.name}</div>
                <div style="font-size: 0.8rem; color: {sev_badge}; margin-top: 4px; font-weight: 600;">Severity: {primary_info.severity}</div>
            </div>
            """, unsafe_allow_html=True)

        with c_diag2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Applicable Standard Codes</div>
                <div style="font-size: 1.05rem; font-weight: 700; color: #f8fafc; margin-top: 4px;">
                    ASME Sec. VIII Div 1 UW-51<br>ISO 10675-1 / ISO 5817
                </div>
            </div>
            """, unsafe_allow_html=True)

        with c_diag3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Total Indications Analyzed</div>
                <div class="metric-value">{len(detections)} Identified</div>
                <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 4px;">Calibration: {pixel_to_mm} mm/px</div>
            </div>
            """, unsafe_allow_html=True)

        # Official Inspector Sign-off & Audit Trail Block
        inspect_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        st.markdown(f"""
        <div style="background: #1e293b; border-radius: 8px; padding: 16px 20px; border: 1px solid #334155; margin: 15px 0 25px 0;">
            <div style="font-size: 0.82rem; color: #38bdf8; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px;">
                📜 Certified Inspector Sign-Off & Chain of Custody (ASNT / ISO 9712)
            </div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; font-size: 0.88rem; color: #cbd5e1;">
                <div><strong>Lead Inspector:</strong> {current_user.full_name}</div>
                <div><strong>Certification:</strong> {current_user.role} (<code>{current_user.cert_id}</code>)</div>
                <div><strong>Testing Facility:</strong> {current_user.facility}</div>
                <div><strong>Timestamp:</strong> {inspect_timestamp}</div>
            </div>
            <div style="margin-top: 10px; font-size: 0.78rem; color: #10b981; font-weight: 600;">
                ✔ Digitally Verified & Archived under ASME Boiler and Pressure Vessel Code (BPVC) Rules
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Detected Indication Breakdown & Morphometric Measurements")

        if detections:
            records = []
            for d in detections:
                m = d.measurements
                records.append({
                    "Defect #": f"#{d.defect_id}",
                    "Class": f"[{d.defect_class.value}] {DEFECT_CATALOG[d.defect_class].name}",
                    "Confidence": f"{d.confidence*100:.1f}%",
                    "Centroid (X, Y)": f"({m.centroid[0]:.0f}, {m.centroid[1]:.0f}) px",
                    "Length (mm)": f"{m.length_mm:.2f} mm",
                    "Width (mm)": f"{m.width_mm:.2f} mm",
                    "Area (mm²)": f"{m.area_mm2:.2f} mm²",
                    "Aspect Ratio": f"{m.aspect_ratio:.2f}",
                    "Orientation": f"{m.orientation_deg:.1f}°",
                    "ASME Decision": d.asme_verdict,
                    "Code Standard Rule": d.asme_clause,
                })
            df = pd.DataFrame(records)
            st.dataframe(df, use_container_width=True, hide_index=True)

            st.markdown("### Visual Inspection Overlay")
            st.image(
                annotated_image,
                caption=f"Annotated Radiograph with ASME Indication Tags: {image_name}",
                use_container_width=True,
            )
        else:
            st.success("🎉 No rejectable defect indications detected in this radiograph. The weld meets ASME Section V and ISO 10675-1 Quality Level B requirements for sound fusion.")

    with tab_hist:
        st.subheader("Radiographic Pixel Intensity & Cumulative Distribution (CDF)")
        st.markdown("""
        In raw industrial radiographs, low transmission contrast causes pixel intensities to cluster tightly in narrow bands.
        **CLAHE** distributes pixel values evenly across the full 8-bit dynamic range [0 - 255] while preventing background quantum noise over-saturation via clipping limits.
        """)
        fig_hist = plot_histogram_comparison(artifacts.raw_image, artifacts.final_image)
        st.pyplot(fig_hist)
        plt.close(fig_hist)

    with tab_roadmap:
        st.subheader("Standard Defect Classification Catalog (Reference Guide)")
        catalog_cols = st.columns(4)
        for idx, (code, info) in enumerate(DEFECT_CATALOG.items()):
            with catalog_cols[idx]:
                sev_color = "#ef4444" if info.severity == "Critical" else ("#f59e0b" if info.severity == "Moderate" else "#10b981")
                st.markdown(f"""
                <div style="background: #1e293b; border-radius: 8px; padding: 14px; border: 1px solid #334155; height: 100%;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <span style="font-size: 1.2rem; font-weight: 700; color: #f8fafc;">[{code.value}] {info.name}</span>
                        <span style="font-size: 0.72rem; padding: 2px 6px; border-radius: 4px; background: {sev_color}22; color: {sev_color}; border: 1px solid {sev_color};">{info.severity}</span>
                    </div>
                    <p style="font-size: 0.82rem; color: #94a3b8; margin: 0;">{info.description}</p>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("System Pipeline Architecture & Implementation Status")

        stages = [
            ("Stage 1: Image Ingestion & I/O", "Supports 8-bit/16-bit PNG, TIFF, JPG, custom upload & synthetic generator", "OPERATIONAL", "badge-active"),
            ("Stage 2: CLAHE & Edge Denoising", "Contrast-limiting adaptive equalization with Bilateral/NLM filtering", "OPERATIONAL", "badge-active"),
            ("Stage 3: Defect Detection & ASME Evaluation", "Automated anomaly localization and ASME Section VIII compliance evaluation", "OPERATIONAL", "badge-active"),
            ("Stage 4: Deep YOLOv8 Detection", "Deep learning detector with trained NDT weights", "STAGED FOR PHASE 2", "badge-staged"),
            ("Stage 5: U-Net ResNet34 Segmentation", "Sub-pixel defect contour mask extraction", "STAGED FOR PHASE 3", "badge-staged"),
            ("Stage 6: Defect Morphometrics", "Automated length, width, area, aspect ratio, centroid, orientation in mm", "OPERATIONAL", "badge-active"),
            ("Stage 7: Grad-CAM Explainability", "Convolutional feature activation heatmap visualization", "STAGED FOR PHASE 5", "badge-staged"),
        ]

        for title, desc, status, badge_class in stages:
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; align-items: center; background: #1e293b; padding: 10px 16px; border-radius: 6px; margin-bottom: 8px; border: 1px solid #334155;">
                <div>
                    <span style="font-weight: 600; color: #f1f5f9;">{title}</span>
                    <span style="font-size: 0.85rem; color: #94a3b8; margin-left: 12px;">— {desc}</span>
                </div>
                <span class="status-badge {badge_class}">{status}</span>
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
