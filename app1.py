import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# ==============================================================================
# Page Configuration & Styling
# ==============================================================================
st.set_page_config(
    page_title="RiverGuard CV | River Waste Distribution Analyzer",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #0F4C81;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4A5568;
        margin-bottom: 1.0rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# Built-in Sample Generator (Optional Fallback When No Photo Is Uploaded)
# ==============================================================================
def generate_demo_river_image(scenario: str = "Bank Accumulation") -> np.ndarray:
    """Generates a synthetic river surface image with floating debris for quick testing."""
    rng = np.random.default_rng(42 if scenario == "Bank Accumulation" else 99)
    h, w = 540, 860

    y_grid, x_grid = np.mgrid[0:h, 0:w]
    b = 75 + (15 * np.sin(y_grid / 18.0)).astype(np.int16)
    g = 98 + (18 * np.cos(x_grid / 25.0 + y_grid / 15.0)).astype(np.int16)
    r = 68 + (12 * np.sin(x_grid / 30.0)).astype(np.int16)
    water = np.stack([b, g, r], axis=-1)

    noise = rng.normal(0, 7, (h, w, 3)).astype(np.int16)
    water = np.clip(water + noise, 0, 255).astype(np.uint8)
    water = cv2.GaussianBlur(water, (5, 5), 1.2)

    cv2.fillPoly(water, [np.array([[0, 0], [55, 0], [35, h], [0, h]])], (48, 72, 45))
    cv2.fillPoly(water, [np.array([[w - 45, 0], [w, 0], [w, h], [w - 65, h]])], (45, 68, 42))

    if scenario == "Bank Accumulation":
        centers = [
            (95, 110), (130, 145), (85, 190), (155, 220), (110, 265),
            (175, 295), (95, 340), (140, 385), (195, 420), (115, 455),
            (420, 240), (680, 370)
        ]
    else:
        centers = [
            (310, 95), (460, 130), (540, 175), (370, 230), (490, 265),
            (330, 320), (580, 345), (430, 395), (510, 445), (290, 470)
        ]

    palette = [
        (238, 242, 245),  # White styrofoam / plastic bag
        (225, 232, 238),  # Translucent bottle
        (45, 65, 220),    # Red wrapper
        (215, 135, 35),   # Blue plastic container
        (35, 205, 235),   # Yellow packaging
    ]

    for idx, (cx, cy) in enumerate(centers):
        color = palette[idx % len(palette)]
        if idx % 3 == 0:
            pts = np.array([
                [cx - 24, cy - 12],
                [cx + 18, cy - 18],
                [cx + 28, cy + 10],
                [cx - 10, cy + 22],
                [cx - 28, cy + 8],
            ], dtype=np.int32)
            cv2.fillPoly(water, [pts], color)
        elif idx % 3 == 1:
            rect = ((cx, cy), (38, 16), float((idx * 37) % 180))
            box = np.int32(cv2.boxPoints(rect))
            cv2.fillPoly(water, [box], color)
        else:
            cv2.ellipse(water, (cx, cy), (18, 12), (idx * 25) % 180, 0, 360, color, -1)

    return water


# ==============================================================================
# Core Computer Vision Pipeline (Exactly 6 Distinct CV Techniques)
# ==============================================================================
def run_cv_pipeline(
    image_bgr: np.ndarray,
    bilateral_d: int = 7,
    clahe_clip: float = 2.5,
    sensitivity: float = 1.0,
    min_area: int = 300,
    grid_rows: int = 4,
    grid_cols: int = 6,
):
    """
    Executes a clean 6-technique computer vision pipeline for river waste distribution:
      1. Bilateral Filtering (Edge-preserving water ripple denoising)
      2. CLAHE Contrast Enhancement (Local contrast improvement in turbid/shadowed water)
      3. HSV Colour Space Conversion (Water background modelling & anomaly scoring)
      4. Otsu's Adaptive Thresholding (Foreground waste segmentation)
      5. Morphological Operations (Opening & Closing for glare removal and mask repair)
      6. Contour Detection & Feature Extraction (Object localization + spatial distribution map)
    """
    orig_h, orig_w = image_bgr.shape[:2]
    max_dim = 960
    if max(orig_h, orig_w) > max_dim:
        scale = max_dim / float(max(orig_h, orig_w))
        img = cv2.resize(image_bgr, (int(orig_w * scale), int(orig_h * scale)), interpolation=cv2.INTER_AREA)
    else:
        img = image_bgr.copy()

    h, w = img.shape[:2]

    # --------------------------------------------------------------------------
    # Technique 1: Bilateral Filtering (Noise Suppression)
    # --------------------------------------------------------------------------
    filtered_bgr = cv2.bilateralFilter(img, d=bilateral_d, sigmaColor=55, sigmaSpace=55)

    # --------------------------------------------------------------------------
    # Technique 2: CLAHE Contrast Enhancement
    # --------------------------------------------------------------------------
    lab_temp = cv2.cvtColor(filtered_bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab_temp)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_ch)
    enhanced_bgr = cv2.cvtColor(cv2.merge([l_enhanced, a_ch, b_ch]), cv2.COLOR_LAB2BGR)

    # --------------------------------------------------------------------------
    # Technique 3: HSV Colour Space Conversion & Water-Anomaly Modelling
    # --------------------------------------------------------------------------
    hsv = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)

    med_h = float(np.median(h_ch))
    med_s = float(np.median(s_ch))
    med_v = float(np.median(v_ch))

    # Circular hue difference (OpenCV Hue range is 0..179)
    raw_h_diff = np.abs(h_ch.astype(np.float32) - med_h)
    hue_diff = np.minimum(raw_h_diff, 180.0 - raw_h_diff)
    sat_diff = np.abs(s_ch.astype(np.float32) - med_s)
    val_anomaly = np.clip(v_ch.astype(np.float32) - med_v, 0, 255)

    anomaly_score = (1.4 * hue_diff) + (1.1 * sat_diff) + (1.2 * val_anomaly)
    anomaly_map = cv2.normalize(anomaly_score, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # --------------------------------------------------------------------------
    # Technique 4: Otsu's Adaptive Thresholding
    # --------------------------------------------------------------------------
    otsu_thresh, _ = cv2.threshold(anomaly_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adjusted_thresh = np.clip(otsu_thresh / max(sensitivity, 0.3), 25, 245)
    _, raw_mask = cv2.threshold(anomaly_map, adjusted_thresh, 255, cv2.THRESH_BINARY)

    # --------------------------------------------------------------------------
    # Technique 5: Morphological Operations (Opening & Closing)
    # --------------------------------------------------------------------------
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    opened_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
    morph_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    # --------------------------------------------------------------------------
    # Technique 6: Contour Detection & Feature Extraction
    # --------------------------------------------------------------------------
    contours, _ = cv2.findContours(morph_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    annotated_img = img.copy()
    final_mask = np.zeros_like(morph_mask)

    records = []
    max_valid_area = 0.40 * (h * w)
    obj_id = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_valid_area:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect_ratio = round(bw / float(bh + 1e-6), 2)
        if aspect_ratio > 12.0 or aspect_ratio < 0.08:
            continue

        obj_id += 1
        cv2.drawContours(final_mask, [cnt], -1, 255, -1)

        perimeter = cv2.arcLength(cnt, True)
        circularity = round((4.0 * np.pi * area) / (perimeter ** 2 + 1e-6), 2)
        cx = int(x + bw / 2)
        cy = int(y + bh / 2)

        rel_x = cx / float(w)
        if rel_x < 0.333:
            zone = "Left Bank"
        elif rel_x < 0.667:
            zone = "Mid-Channel"
        else:
            zone = "Right Bank"

        obj_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(obj_mask, [cnt], -1, 255, -1)
        _, mean_s, mean_v, _ = cv2.mean(hsv, mask=obj_mask)

        if mean_s < 55 and mean_v > 155:
            debris_type = "White Plastic / Foam"
            box_color = (0, 230, 255)
        elif mean_s >= 75:
            debris_type = "Colored Packaging / Bottle"
            box_color = (50, 80, 255)
        else:
            debris_type = "Dark / Mixed Debris"
            box_color = (255, 140, 0)

        cv2.drawContours(annotated_img, [cnt], -1, (0, 255, 120), 2)
        cv2.rectangle(annotated_img, (x, y), (x + bw, y + bh), box_color, 2)
        label = f"#{obj_id} ({int(area)}px)"
        cv2.putText(
            annotated_img,
            label,
            (x, max(y - 6, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            box_color,
            2,
            cv2.LINE_AA,
        )

        records.append({
            "Object ID": f"W-{obj_id:02d}",
            "River Zone": zone,
            "Visual Category": debris_type,
            "Centroid (X, Y)": f"({cx}, {cy})",
            "Area (px²)": int(area),
            "Coverage (%)": round((area / (h * w)) * 100, 3),
            "Aspect Ratio": aspect_ratio,
            "Circularity": circularity,
        })

    # --------------------------------------------------------------------------
    # Spatial Distribution Heatmap & Cross-Sectional Zoning Output
    # --------------------------------------------------------------------------
    density_float = cv2.GaussianBlur(final_mask.astype(np.float32), (0, 0), sigmaX=42)
    if np.max(density_float) > 0:
        density_norm = cv2.normalize(density_float, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        density_norm = np.zeros((h, w), dtype=np.uint8)

    heatmap_color = cv2.applyColorMap(density_norm, cv2.COLORMAP_JET)
    heatmap_overlay = img.copy()
    active_heat = density_norm > 25
    blended = cv2.addWeighted(img, 0.45, heatmap_color, 0.55, 0)
    heatmap_overlay[active_heat] = blended[active_heat]

    x1, x2 = int(w * 0.333), int(w * 0.667)
    cv2.line(heatmap_overlay, (x1, 0), (x1, h), (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(heatmap_overlay, (x2, 0), (x2, h), (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(heatmap_overlay, "LEFT BANK", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(heatmap_overlay, "MID-CHANNEL", (x1 + 15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(heatmap_overlay, "RIGHT BANK", (x2 + 15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    grid_matrix = np.zeros((grid_rows, grid_cols), dtype=np.float32)
    cell_h, cell_w = h / float(grid_rows), w / float(grid_cols)
    for r_idx in range(grid_rows):
        for c_idx in range(grid_cols):
            y_start, y_end = int(r_idx * cell_h), int((r_idx + 1) * cell_h)
            x_start, x_end = int(c_idx * cell_w), int((c_idx + 1) * cell_w)
            cell_slice = final_mask[y_start:y_end, x_start:x_end]
            if cell_slice.size > 0:
                grid_matrix[r_idx, c_idx] = (np.count_nonzero(cell_slice) / float(cell_slice.size)) * 100.0

    total_waste_px = float(np.count_nonzero(final_mask))
    left_px = float(np.count_nonzero(final_mask[:, :x1]))
    mid_px = float(np.count_nonzero(final_mask[:, x1:x2]))
    right_px = float(np.count_nonzero(final_mask[:, x2:]))

    zone_shares = {
        "Left Bank": round((left_px / total_waste_px) * 100, 1) if total_waste_px > 0 else 0.0,
        "Mid-Channel": round((mid_px / total_waste_px) * 100, 1) if total_waste_px > 0 else 0.0,
        "Right Bank": round((right_px / total_waste_px) * 100, 1) if total_waste_px > 0 else 0.0,
    }

    total_coverage_pct = round((total_waste_px / float(h * w)) * 100, 2)

    return {
        "original": img,
        "filtered": filtered_bgr,
        "enhanced": enhanced_bgr,
        "anomaly_map": anomaly_map,
        "raw_mask": raw_mask,
        "morph_mask": morph_mask,
        "annotated": annotated_img,
        "heatmap": heatmap_overlay,
        "records": pd.DataFrame(records),
        "grid_matrix": grid_matrix,
        "zone_shares": zone_shares,
        "total_coverage_pct": total_coverage_pct,
        "total_items": len(records),
    }


# ==============================================================================
# Sidebar Controls
# ==============================================================================
with st.sidebar:
    st.header("⚙️ CV Pipeline Controls")
    st.caption("Adjust computer vision parameters in real time for your uploaded photo:")

    bilateral_d = st.select_slider("Bilateral Filter Diameter (d)", options=[5, 7, 9, 11], value=7)
    clahe_clip = st.slider("CLAHE Contrast Clip Limit", min_value=1.0, max_value=5.0, value=2.5, step=0.5)
    sensitivity = st.slider("Otsu Threshold Sensitivity", min_value=0.6, max_value=1.6, value=1.0, step=0.05)
    min_area = st.slider("Minimum Debris Area (px²)", min_value=100, max_value=2500, value=300, step=50)

    st.divider()
    st.subheader("📐 Spatial Grid Resolution")
    grid_choice = st.selectbox("Distribution Grid (Rows × Cols)", options=["3 × 3", "4 × 6", "5 × 8"], index=1)
    grid_rows, grid_cols = map(int, grid_choice.split(" × "))


# ==============================================================================
# Main Header & Prominent Photo Upload Section
# ==============================================================================
st.markdown('<div class="main-header">🌊 RiverGuard CV: River Floating Waste Distribution Analyzer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">AIT204 Computer Vision & Robotics Prototype — Upload a River Photo to Detect Floating Debris & Map Spatial Hotspots</div>',
    unsafe_allow_html=True,
)

if "use_demo" not in st.session_state:
    st.session_state.use_demo = False

upload_col, demo_col = st.columns([3, 1])

with upload_col:
    uploaded_file = st.file_uploader(
        "📤 Upload River Surface Photo (JPG, JPEG, PNG, WEBP)",
        type=["jpg", "jpeg", "png", "webp"],
        help="Upload an overhead, bridge-view, or drone photograph of a river surface containing floating debris.",
    )

with demo_col:
    st.markdown("<div style='height: 1.75rem;'></div>", unsafe_allow_html=True)
    if st.button("🧪 Load Built-in Demo Photo", use_container_width=True):
        st.session_state.use_demo = True

input_bgr = None
source_label = ""

if uploaded_file is not None:
    st.session_state.use_demo = False
    pil_img = Image.open(uploaded_file).convert("RGB")
    input_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    orig_w, orig_h = pil_img.size
    source_label = f"Uploaded File: **{uploaded_file.name}** ({orig_w} × {orig_h} px)"
elif st.session_state.use_demo:
    demo_scenario = st.radio(
        "Select Built-in Demo Scenario:",
        options=["Bank Accumulation", "Scattered Mid-Stream Drift"],
        horizontal=True,
    )
    input_bgr = generate_demo_river_image(demo_scenario)
    source_label = f"Built-in Demo Sample: **{demo_scenario}** (860 × 540 px)"
else:
    st.info(
        "👆 **Drag and drop a river photo above** (or click **Browse files**) to run the 6-stage computer vision waste distribution analysis. "
        "If you don't have a photo ready yet, click **Load Built-in Demo Photo** on the right."
    )
    st.stop()

st.caption(f"Active Input Source — {source_label}")

# ==============================================================================
# Run Computer Vision Pipeline & Display Results
# ==============================================================================
results = run_cv_pipeline(
    image_bgr=input_bgr,
    bilateral_d=bilateral_d,
    clahe_clip=clahe_clip,
    sensitivity=sensitivity,
    min_area=min_area,
    grid_rows=grid_rows,
    grid_cols=grid_cols,
)

zone_shares = results["zone_shares"]
dominant_zone = max(zone_shares, key=zone_shares.get) if results["total_items"] > 0 else "None"
dominant_pct = zone_shares.get(dominant_zone, 0.0)

cov = results["total_coverage_pct"]
if cov == 0:
    severity = "Clean (0%)"
elif cov < 1.5:
    severity = "Low"
elif cov < 4.5:
    severity = "Moderate"
else:
    severity = "High / Critical"

m1, m2, m3, m4 = st.columns(4)
m1.metric("Detected Waste Objects", f"{results['total_items']} items")
m2.metric("Surface Coverage", f"{cov:.2f}%")
m3.metric("Primary Accumulation Zone", f"{dominant_zone} ({dominant_pct}%)")
m4.metric("Pollution Severity Rating", severity)

st.divider()

# ==============================================================================
# Tabbed Interface for Clear Demonstration & Report Screenshots
# ==============================================================================
tab1, tab2, tab3 = st.tabs([
    "🎯 Detection & Spatial Distribution",
    "🔬 6 CV Processing Techniques (Step-by-Step)",
    "📊 Quantitative Report & Cleanup Insights",
])

# --- TAB 1: Input, Detection, and Spatial Distribution Heatmap ---
with tab1:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("1. Uploaded Visual Input")
        st.image(cv2.cvtColor(results["original"], cv2.COLOR_BGR2RGB), use_container_width=True)
        st.caption("Original river surface photograph.")

    with col2:
        st.subheader("2. Detected Floating Debris")
        st.image(cv2.cvtColor(results["annotated"], cv2.COLOR_BGR2RGB), use_container_width=True)
        st.caption("Bounding boxes & contours labeled by object ID and pixel area.")

    with col3:
        st.subheader("3. Spatial Distribution Heatmap")
        st.image(cv2.cvtColor(results["heatmap"], cv2.COLOR_BGR2RGB), use_container_width=True)
        st.caption("Gaussian density overlay across Left Bank, Mid-Channel, and Right Bank.")

    st.subheader("River Cross-Section Distribution Breakdown")
    z_col1, z_col2, z_col3 = st.columns(3)
    z_col1.metric("Left Bank Share", f"{zone_shares['Left Bank']}%")
    z_col2.metric("Mid-Channel Share", f"{zone_shares['Mid-Channel']}%")
    z_col3.metric("Right Bank Share", f"{zone_shares['Right Bank']}%")

# --- TAB 2: Exact 6 Computer Vision Techniques ---
with tab2:
    st.markdown("Visual output of each of the **6 computer vision techniques** implemented in the pipeline:")
    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown("**Technique 1: Bilateral Filtering**")
        st.image(cv2.cvtColor(results["filtered"], cv2.COLOR_BGR2RGB), use_container_width=True)
        st.caption("Suppresses water ripple noise while keeping debris edges sharp.")

    with s2:
        st.markdown("**Technique 2: CLAHE Enhancement**")
        st.image(cv2.cvtColor(results["enhanced"], cv2.COLOR_BGR2RGB), use_container_width=True)
        st.caption("Improves local contrast between turbid/shadowed water and debris.")

    with s3:
        st.markdown("**Technique 3: HSV Colour Space Anomaly**")
        st.image(results["anomaly_map"], use_container_width=True)
        st.caption("Measures hue, saturation, and brightness deviation from river water.")

    s4, s5, s6 = st.columns(3)
    with s4:
        st.markdown("**Technique 4: Otsu's Thresholding**")
        st.image(results["raw_mask"], use_container_width=True)
        st.caption("Automatically binarises foreground waste candidates from background.")

    with s5:
        st.markdown("**Technique 5: Morphological Opening & Closing**")
        st.image(results["morph_mask"], use_container_width=True)
        st.caption("Removes sun-glint speckles (Opening) and fills holes in plastics (Closing).")

    with s6:
        st.markdown("**Technique 6: Contour & Feature Extraction**")
        st.image(cv2.cvtColor(results["annotated"], cv2.COLOR_BGR2RGB), use_container_width=True)
        st.caption("Filters contours by area/aspect ratio and extracts geometric & colour features.")

# --- TAB 3: Tabular Feature Data, Grid Density & Automated Decision Support ---
with tab3:
    df = results["records"]
    if df.empty:
        st.warning("No floating waste objects exceeded the minimum area threshold. Try lowering 'Minimum Debris Area' or increasing 'Otsu Threshold Sensitivity' in the left sidebar.")
    else:
        st.subheader("Extracted Debris Feature Table")
        st.dataframe(df, use_container_width=True)

        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Detection & Distribution Data (.csv)",
            data=csv_bytes,
            file_name="river_waste_distribution_report.csv",
            mime="text/csv",
        )

        st.subheader(f"Spatial Grid Coverage Matrix ({grid_rows}×{grid_cols}, %)")
        grid_df = pd.DataFrame(
            np.round(results["grid_matrix"], 2),
            index=[f"Row {i+1}" for i in range(grid_rows)],
            columns=[f"Col {j+1}" for j in range(grid_cols)],
        )
        st.dataframe(grid_df, use_container_width=True)

        st.subheader("Automated Spatial Interpretation & Environmental Action Plan")
        top_category = df["Visual Category"].mode()[0]
        max_cell_val = float(np.max(results["grid_matrix"]))
        max_r, max_c = np.unravel_index(np.argmax(results["grid_matrix"]), results["grid_matrix"].shape)

        if dominant_pct >= 50.0:
            pattern_desc = f"**Clustered Accumulation ({dominant_zone})**: {dominant_pct}% of detected waste is concentrated along the **{dominant_zone}**, likely driven by slow boundary currents or bankside vegetation."
            action_rec = f"Deploy a stationary **floating trash boom / interceptor barrier** along the **{dominant_zone}** (around Grid Sector **Row {max_r+1}, Col {max_c+1}**, peak density {max_cell_val:.2f}%) for efficient collection."
        else:
            pattern_desc = f"**Dispersed Drift Pattern**: Waste is spread across multiple zones, with the highest share in **{dominant_zone}** ({dominant_pct}%)."
            action_rec = f"Deploy a **cross-channel floating diversion boom** downstream to funnel dispersed {top_category.lower()} toward a collection point on the **{dominant_zone}**."

        st.success(
            f"- **Spatial Distribution Pattern:** {pattern_desc}\n"
            f"- **Dominant Waste Type:** **{top_category}** accounts for the largest share of detected items.\n"
            f"- **Operational Recommendation:** {action_rec}"
        )