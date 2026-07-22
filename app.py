import json
import time
from pathlib import Path

import cv2
import numpy as np
import requests
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# ----------------------------------------------------------------------------
# MODEL DOWNLOAD (used when best_model.pt is hosted as a GitHub Release asset
# instead of being committed directly to the repo, e.g. because it's >25MB)
# ----------------------------------------------------------------------------
# If you upload best_model.pt as a GitHub Release asset, paste its direct
# download URL here (Release page -> right-click the asset -> Copy link).
# Leave empty if best_model.pt is already committed next to this file.
MODEL_URL = ""

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Road Image Detection",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# PATHS
# ----------------------------------------------------------------------------
APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "best_model.pt"
CLASSES_PATH = APP_DIR / "classes.json"
INFO_PATH = APP_DIR / "model_info.json"

# ----------------------------------------------------------------------------
# STYLE
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Poppins', sans-serif;
    }

    .stApp {
        background: linear-gradient(180deg, #0f172a 0%, #111827 40%, #0f172a 100%);
        color: #e5e7eb;
    }

    /* Hero header */
    .hero {
        padding: 2.2rem 2rem;
        border-radius: 20px;
        background: linear-gradient(120deg, #f97316 0%, #f59e0b 35%, #14b8a6 100%);
        box-shadow: 0 10px 30px rgba(0,0,0,0.35);
        margin-bottom: 1.6rem;
    }
    .hero h1 {
        font-size: 2.4rem;
        font-weight: 700;
        color: #0b1220;
        margin: 0;
    }
    .hero p {
        color: #0b1220;
        opacity: 0.85;
        margin-top: 0.4rem;
        font-size: 1.05rem;
    }

    /* Cards */
    .card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1.1rem 1.3rem;
        backdrop-filter: blur(6px);
    }

    .metric-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.09);
        border-radius: 14px;
        padding: 0.9rem 1rem;
        text-align: center;
    }
    .metric-card h2 {
        margin: 0;
        font-size: 1.6rem;
        color: #f97316;
    }
    .metric-card p {
        margin: 0;
        color: #9ca3af;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Class legend chip */
    .chip {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.3rem 0.7rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.06);
        margin: 0.2rem;
        font-size: 0.85rem;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        display: inline-block;
    }

    section[data-testid="stSidebar"] {
        background: #0b1220;
        border-right: 1px solid rgba(255,255,255,0.06);
    }

    .stButton>button {
        background: linear-gradient(90deg, #f97316, #f59e0b);
        color: #0b1220;
        font-weight: 700;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.4rem;
    }
    .stButton>button:hover {
        opacity: 0.9;
        color: #0b1220;
    }

    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# HERO HEADER
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🛣️ Road Image Detection</h1>
        <p>AI-powered road damage detection &mdash; spot cracks, potholes and surface defects instantly with YOLO.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# LOAD RESOURCES
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading detection model...")
def load_model(path: str):
    return YOLO(path)


@st.cache_data
def load_classes(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_model_info(path: str):
    if Path(path).exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


CLASS_COLORS = {
    "alligator": "#ef4444",
    "block": "#f97316",
    "crack": "#eab308",
    "edge": "#22c55e",
    "longitudinal": "#14b8a6",
    "pothole": "#3b82f6",
    "transverse": "#a855f7",
}


def hex_to_bgr(hex_color: str):
    hex_color = hex_color.lstrip("#")
    r, g, b = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


def download_model(url: str, dest: Path):
    """Download the model weights from a direct URL (e.g. a GitHub Release asset)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        progress = st.progress(0, text="Downloading model weights...")
        downloaded = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    progress.progress(min(downloaded / total, 1.0),
                                       text=f"Downloading model weights... {downloaded // (1024*1024)}MB / {total // (1024*1024)}MB")
        progress.empty()


if not MODEL_PATH.exists():
    if MODEL_URL:
        with st.spinner("First run: downloading model weights, please wait..."):
            try:
                download_model(MODEL_URL, MODEL_PATH)
            except Exception as e:
                st.error(f"⚠️ Failed to download the model from MODEL_URL: {e}")
                st.stop()
    else:
        st.error(
            "⚠️ Model file `best_model.pt` was not found next to `app.py`, and no "
            "`MODEL_URL` was set to download it. Either commit `best_model.pt` to the "
            "repo, or set `MODEL_URL` at the top of this file to a direct download link "
            "(e.g. a GitHub Release asset URL)."
        )
        st.stop()

if not CLASSES_PATH.exists():
    st.error("⚠️ `classes.json` was not found. Please add it next to `app.py`.")
    st.stop()

model = load_model(str(MODEL_PATH))
class_names = load_classes(str(CLASSES_PATH))
model_info = load_model_info(str(INFO_PATH))

# ----------------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Detection Settings")

    conf_threshold = st.slider("Confidence threshold", 0.05, 1.0, 0.25, 0.05)
    iou_threshold = st.slider("IoU threshold (NMS)", 0.05, 1.0, 0.45, 0.05)

    selected_classes = st.multiselect(
        "Classes to detect",
        options=class_names,
        default=class_names,
        help="Only show detections for the selected damage types.",
    )

    st.markdown("---")
    st.markdown("### 🎨 Class Legend")
    legend_html = ""
    for cls in class_names:
        color = CLASS_COLORS.get(cls, "#94a3b8")
        legend_html += (
            f'<span class="chip"><span class="dot" style="background:{color};"></span>'
            f"{cls}</span>"
        )
    st.markdown(legend_html, unsafe_allow_html=True)

    if model_info:
        st.markdown("---")
        st.markdown("### 📊 Model Performance")
        st.markdown(f"**Architecture:** {model_info.get('base_architecture', '—')}")
        st.markdown(f"**mAP50:** {model_info.get('mAP50', 0):.3f}")
        st.markdown(f"**mAP50-95:** {model_info.get('mAP50-95', 0):.3f}")
        st.markdown(f"**Precision:** {model_info.get('precision', 0):.3f}")
        st.markdown(f"**Recall:** {model_info.get('recall', 0):.3f}")
        st.markdown(f"**Size:** {model_info.get('size_mb', 0):.1f} MB")

    st.markdown("---")
    st.caption("Built with YOLO + Streamlit • Road Image Detection")

# ----------------------------------------------------------------------------
# MAIN TABS
# ----------------------------------------------------------------------------
tab_detect, tab_about = st.tabs(["🔍 Detect Damage", "ℹ️ About"])

with tab_detect:
    upload_col, info_col = st.columns([2, 1])

    with upload_col:
        st.markdown("#### Upload road image(s)")
        uploaded_files = st.file_uploader(
            "Drag and drop or browse",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

    with info_col:
        st.markdown(
            f"""
            <div class="card">
            <b>Model classes ({len(class_names)}):</b><br>
            {", ".join(class_names)}
            </div>
            """,
            unsafe_allow_html=True,
        )

    if uploaded_files:
        run = st.button(f"🚀 Run detection on {len(uploaded_files)} image(s)")

        if run:
            all_counts = {}
            total_time = 0.0

            for uploaded_file in uploaded_files:
                st.markdown("---")
                image = Image.open(uploaded_file).convert("RGB")
                img_array = np.array(image)

                start = time.time()
                results = model.predict(
                    img_array,
                    conf=conf_threshold,
                    iou=iou_threshold,
                    verbose=False,
                )[0]
                elapsed = time.time() - start
                total_time += elapsed

                boxes = results.boxes
                annotated = img_array.copy()
                rows = []

                for box in boxes:
                    cls_id = int(box.cls[0])
                    cls_name = model.names.get(cls_id, str(cls_id))
                    if cls_name not in selected_classes:
                        continue

                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    color = hex_to_bgr(CLASS_COLORS.get(cls_name, "#94a3b8"))

                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
                    label = f"{cls_name} {conf:.2f}"
                    (tw, th), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )
                    cv2.rectangle(
                        annotated, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1
                    )
                    cv2.putText(
                        annotated,
                        label,
                        (x1 + 3, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                    rows.append({"class": cls_name, "confidence": round(conf, 3),
                                 "x1": x1, "y1": y1, "x2": x2, "y2": y2})
                    all_counts[cls_name] = all_counts.get(cls_name, 0) + 1

                st.markdown(f"**{uploaded_file.name}** &nbsp;·&nbsp; {elapsed:.2f}s &nbsp;·&nbsp; {len(rows)} detection(s)")

                c1, c2 = st.columns(2)
                with c1:
                    st.image(image, caption="Original", use_container_width=True)
                with c2:
                    st.image(annotated, caption="Detected", use_container_width=True)

                if rows:
                    st.dataframe(rows, use_container_width=True, hide_index=True)

                    annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
                    ok, buf = cv2.imencode(".jpg", annotated_bgr)
                    if ok:
                        st.download_button(
                            "⬇️ Download annotated image",
                            data=buf.tobytes(),
                            file_name=f"detected_{uploaded_file.name}",
                            mime="image/jpeg",
                            key=f"dl_{uploaded_file.name}",
                        )
                else:
                    st.info("No damage detected above the current confidence threshold.")

            # Summary section across all uploaded images
            if all_counts:
                st.markdown("---")
                st.markdown("### 📈 Summary across all images")

                m1, m2, m3 = st.columns(3)
                with m1:
                    st.markdown(
                        f'<div class="metric-card"><h2>{sum(all_counts.values())}</h2><p>Total detections</p></div>',
                        unsafe_allow_html=True,
                    )
                with m2:
                    st.markdown(
                        f'<div class="metric-card"><h2>{len(uploaded_files)}</h2><p>Images processed</p></div>',
                        unsafe_allow_html=True,
                    )
                with m3:
                    st.markdown(
                        f'<div class="metric-card"><h2>{total_time / len(uploaded_files):.2f}s</h2><p>Avg. time / image</p></div>',
                        unsafe_allow_html=True,
                    )

                st.markdown("#### Detections by damage type")
                st.bar_chart(all_counts)
    else:
        st.info("👆 Upload one or more road images to get started.")

with tab_about:
    st.markdown(
        """
        <div class="card">
        <h3>About Road Image Detection</h3>
        <p>
        This app uses a fine-tuned <b>YOLO</b> object-detection model to automatically
        identify common road surface defects from photos, helping road maintenance
        teams triage inspections faster.
        </p>
        <p><b>Detected damage types:</b></p>
        <ul>
            <li><b>Alligator</b> — interconnected cracking resembling alligator skin</li>
            <li><b>Block</b> — rectangular cracking patterns</li>
            <li><b>Crack</b> — general surface cracking</li>
            <li><b>Edge</b> — cracking near the pavement edge</li>
            <li><b>Longitudinal</b> — cracks running parallel to the road direction</li>
            <li><b>Pothole</b> — localized surface cavities</li>
            <li><b>Transverse</b> — cracks running across the road direction</li>
        </ul>
        <p>
        Adjust the confidence and IoU thresholds in the sidebar to fine-tune detections,
        and filter which damage types are shown.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
