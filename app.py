"""
=============================================================
AI-Powered Vehicle Accident Anticipation — Streamlit UI
=============================================================
Run:  streamlit run app.py
"""

import os, csv, io, time, tempfile, warnings
from collections import deque
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import streamlit as st
import pandas as pd
import altair as alt

warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Vehicle Accident Anticipation System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #1a1a2e, #16213e);
    color: #e0e0e0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.05);
    border-right: 1px solid rgba(255,255,255,0.08);
}

/* Metric cards */
.metric-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 14px;
    padding: 18px 22px;
    text-align: center;
    backdrop-filter: blur(10px);
    transition: transform 0.2s;
}
.metric-card:hover { transform: translateY(-3px); }
.metric-value { font-size: 2rem; font-weight: 700; }
.metric-label { font-size: 0.78rem; color: #a0a0c0; margin-top: 4px; letter-spacing: 0.05em; }

/* Alert banner */
.alert-banner {
    background: linear-gradient(90deg, rgba(220,50,50,0.25), rgba(220,50,50,0.08));
    border-left: 4px solid #e74c3c;
    border-radius: 10px;
    padding: 14px 18px;
    margin: 8px 0;
    font-size: 0.93rem;
}
.safe-banner {
    background: linear-gradient(90deg, rgba(46,213,115,0.18), rgba(46,213,115,0.05));
    border-left: 4px solid #2ed573;
    border-radius: 10px;
    padding: 14px 18px;
    margin: 8px 0;
    font-size: 0.93rem;
}

/* Section header */
.section-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #7b8fff;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin: 18px 0 10px 0;
    border-bottom: 1px solid rgba(123,143,255,0.25);
    padding-bottom: 6px;
}

/* Upload zone */
[data-testid="stFileUploader"] {
    border: 2px dashed rgba(123,143,255,0.4) !important;
    border-radius: 14px !important;
    background: rgba(123,143,255,0.05) !important;
}

/* Progress bar color */
.stProgress > div > div > div { background: linear-gradient(90deg, #7b8fff, #a78bfa); }

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #7b8fff, #a78bfa);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    padding: 0.55rem 1.6rem;
    letter-spacing: 0.03em;
    transition: opacity 0.2s, transform 0.15s;
}
.stButton > button:hover { opacity: 0.88; transform: translateY(-2px); }

/* Dataframe */
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# MODEL LOADING  (cached so they load only once per session)
# ════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="⚙️  Loading MobileNetV2 (INT8) …")
def load_feature_extractor():
    import torchvision.models as models

    class FeatNet(nn.Module):
        def __init__(self):
            super().__init__()
            b = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
            self.features = b.features
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        def forward(self, x):
            return self.pool(self.features(x)).flatten(1)

    model = FeatNet().eval()
    model = torch.quantization.quantize_dynamic(model, {nn.Conv2d, nn.Linear}, dtype=torch.qint8)
    return model


@st.cache_resource(show_spinner="⚙️  Loading YOLOv8-nano …")
def load_yolo():
    from ultralytics import YOLO
    return YOLO("yolov8n.pt")


@st.cache_resource(show_spinner="⚙️  Loading BiGRU forecaster …")
def load_bigru():
    class BiGRU(nn.Module):
        def __init__(self, D=1280, H=256, L=2):
            super().__init__()
            self.gru = nn.GRU(D, H, L, batch_first=True, bidirectional=True,
                              dropout=0.2 if L > 1 else 0.0)
            d = H * 2
            self.risk = nn.Sequential(nn.Linear(d,64), nn.ReLU(), nn.Linear(64,1), nn.Sigmoid())
            self.traj = nn.Sequential(nn.Linear(d,64), nn.ReLU(), nn.Linear(64,2))
        def forward(self, x):
            o, _ = self.gru(x)
            last = o[:, -1, :]
            return self.risk(last), self.traj(last)

    model = BiGRU().eval()
    model = torch.quantization.quantize_dynamic(model, {nn.GRU, nn.Linear}, dtype=torch.qint8)
    return model


@st.cache_resource(show_spinner="⚙️  Loading DistilBERT …")
def load_bert():
    from transformers import AutoTokenizer, AutoModel
    tok   = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModel.from_pretrained("distilbert-base-uncased").eval()
    model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    return tok, model


@st.cache_resource(show_spinner="⚙️  Loading MiDaS depth estimator …")
def load_midas():
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True).eval()
    xf    = torch.hub.load("intel-isl/MiDaS", "transforms",  trust_repo=True)
    return midas, xf.small_transform


# ════════════════════════════════════════════════════════════════
# INFERENCE HELPERS
# ════════════════════════════════════════════════════════════════

IMG_TF = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

def extract_features(feat_model, pil_img):
    t = IMG_TF(pil_img).unsqueeze(0)
    with torch.inference_mode():
        return feat_model(t).numpy()          # (1, 1280)

def detect(yolo, bgr):
    res = yolo(bgr, verbose=False)[0]
    return [{"label": res.names[int(b.cls)],
             "conf":  float(b.conf),
             "bbox":  b.xyxy[0].tolist()} for b in res.boxes]

def forecast(gru, buf):
    x = torch.tensor(np.stack(list(buf)), dtype=torch.float32).unsqueeze(0)
    with torch.inference_mode():
        r, t = gru(x)
    return r.item(), t.squeeze().tolist()

def depth_infer(midas, xf, bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    inp = xf(rgb)
    with torch.inference_mode():
        d = midas(inp)
        d = nn.functional.interpolate(
            d.unsqueeze(1) if d.dim()==3 else d,
            size=rgb.shape[:2], mode="bicubic", align_corners=False
        ).squeeze().numpy()
    return d

def compute_threshold(complexity: float) -> float:
    return float(np.clip(0.50 - 0.30*(complexity - 0.5), 0.25, 0.75))

def make_alert(dets, risk, depth_map, threshold):
    if risk < threshold:
        return "safe", "Scene within safe parameters."
    h, w = depth_map.shape
    nearest, dist = "object", "?"
    best = -1
    for d in dets:
        b  = [int(v) for v in d["bbox"]]
        cx = min(max((b[0]+b[2])//2, 0), w-1)
        cy = min(max((b[1]+b[3])//2, 0), h-1)
        dv = depth_map[cy, cx]
        if dv > best:
            best, nearest, dist = dv, d["label"], round(dv/50.0, 1)
    sev = "CRITICAL" if risk >= 0.80 else ("WARNING" if risk >= 0.65 else "CAUTION")
    return "alert", (f"[{sev}] Risk {risk:.2f} — {nearest} detected ~{dist} m ahead. "
                     "Recommend braking / evasive action.")

def annotate_frame(frame, dets, risk, threshold, alert_msg, status):
    ann   = frame.copy()
    H, W  = ann.shape[:2]
    color = (0,0,220) if status=="alert" else (0,210,0)

    for d in dets:
        x1,y1,x2,y2 = [int(v) for v in d["bbox"]]
        cv2.rectangle(ann, (x1,y1),(x2,y2), color, 2)
        cv2.putText(ann, f"{d['label']} {d['conf']:.0%}",
                    (x1, max(y1-8,12)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    cv2.rectangle(ann, (0,0),(W,52),(20,20,20),-1)
    lbl = "⚠ ALERT" if status=="alert" else "SAFE"
    cv2.putText(ann, f"Risk: {risk:.3f}  |  Threshold: {threshold:.2f}  |  {lbl}",
                (10,34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2)

    if status == "alert":
        cv2.rectangle(ann, (0,H-48),(W,H),(0,0,160),-1)
        cv2.putText(ann, alert_msg[:95], (10,H-14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255,255,255), 2)

    return cv2.cvtColor(ann, cv2.COLOR_BGR2RGB)


# ════════════════════════════════════════════════════════════════
# SIDEBAR — Configuration
# ════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🚗 Accident Anticipation")
    st.markdown("---")

    st.markdown('<div class="section-header">📤 Upload Video</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Drag & drop a dashcam video",
        type=["mp4","avi","mov","mkv"],
        label_visibility="collapsed",
    )

    st.markdown('<div class="section-header">🌍 Road Context</div>', unsafe_allow_html=True)
    road_preset = st.selectbox("Environment preset", [
        "Urban city roads",
        "Highway / motorway",
        "Suburban area",
        "Rain / night driving",
        "Custom …",
    ])
    preset_map = {
        "Urban city roads":    "Traffic density: Dense. Weather: Clear. Road: Urban city.",
        "Highway / motorway":  "Traffic density: Light. Weather: Clear. Road: Highway.",
        "Suburban area":       "Traffic density: Medium. Weather: Clear. Road: Suburban.",
        "Rain / night driving":"Traffic density: Dense. Weather: Rainy and dark. Road: Urban.",
    }
    if road_preset == "Custom …":
        road_annotation = st.text_area("Describe the road environment",
            "Traffic density: Dense. Weather: Clear. Road: Urban city roads.")
    else:
        road_annotation = preset_map[road_preset]
        st.caption(f"*{road_annotation}*")

    st.markdown('<div class="section-header">⚙️ Parameters</div>', unsafe_allow_html=True)
    scene_complexity = st.slider(
        "Scene complexity",
        min_value=0.1, max_value=1.0, value=0.65, step=0.05,
        help="0.3 = open highway · 0.7 = dense urban · 0.9 = rain/snow",
    )
    frame_skip = st.slider(
        "Frame skip (process every Nth frame)",
        min_value=1, max_value=30, value=10,
        help="Higher = faster but less temporal resolution",
    )
    seq_len = st.slider(
        "BiGRU temporal window",
        min_value=2, max_value=20, value=8,
        help="Number of past processed frames fed into the GRU",
    )

    st.markdown("---")
    run_btn = st.button("▶  Run Analysis", use_container_width=True,
                        disabled=(uploaded is None))


# ════════════════════════════════════════════════════════════════
# MAIN AREA
# ════════════════════════════════════════════════════════════════

st.markdown("""
<h1 style='text-align:center; font-size:2rem; font-weight:700;
           background: linear-gradient(90deg,#7b8fff,#a78bfa,#f39c12);
           -webkit-background-clip:text; -webkit-text-fill-color:transparent;
           margin-bottom:4px;'>
    🚗 AI Vehicle Accident Anticipation System
</h1>
<p style='text-align:center; color:#8888aa; font-size:0.9rem; margin-bottom:30px;'>
    CPU-optimised · MobileNetV2 + YOLOv8 + BiGRU + DistilBERT + MiDaS
</p>
""", unsafe_allow_html=True)

if uploaded is None:
    st.markdown("""
    <div style='border:2px dashed rgba(123,143,255,0.3); border-radius:18px;
                padding:60px; text-align:center; margin-top:20px;'>
        <div style='font-size:3.5rem'>📹</div>
        <h3 style='color:#7b8fff; margin:10px 0 6px'>Upload a dashcam video</h3>
        <p style='color:#8888aa; font-size:0.9rem'>
            Supported: MP4, AVI, MOV, MKV<br>
            Configure parameters in the sidebar, then hit <b>Run Analysis</b>
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Load models once ─────────────────────────────────────────────────────────
feat_model        = load_feature_extractor()
yolo_model        = load_yolo()
gru_model         = load_bigru()
bert_tok, bert_m  = load_bert()
midas, midas_xf   = load_midas()

threshold = compute_threshold(scene_complexity)

# ── Show video info ───────────────────────────────────────────────────────────
with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
    tmp.write(uploaded.read())
    tmp_path = tmp.name

cap   = cv2.VideoCapture(tmp_path)
fps   = cap.get(cv2.CAP_PROP_FPS)   or 25.0
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

c1, c2, c3, c4 = st.columns(4)
for col, val, lbl in zip(
    [c1, c2, c3, c4],
    [f"{W}×{H}", f"{fps:.0f} fps", str(total), f"~{total//frame_skip}"],
    ["Resolution", "Frame Rate", "Total Frames", "Frames to Process"],
):
    col.markdown(f"""
    <div class='metric-card'>
        <div class='metric-value' style='color:#a78bfa'>{val}</div>
        <div class='metric-label'>{lbl}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Only run when button pressed ──────────────────────────────────────────────
if not run_btn:
    st.info("👈  Configure settings in the sidebar and click **Run Analysis** to start.")
    st.stop()

# ════════════════════════════════════════════════════════════════
# PROCESSING LOOP
# ════════════════════════════════════════════════════════════════

col_left, col_right = st.columns([3, 2])
with col_left:
    st.markdown('<div class="section-header">📺 Live Frame Preview</div>', unsafe_allow_html=True)
    frame_display = st.empty()

with col_right:
    st.markdown('<div class="section-header">📊 Live Risk Score</div>', unsafe_allow_html=True)
    chart_placeholder = st.empty()
    st.markdown('<div class="section-header">🔔 Last Alert</div>', unsafe_allow_html=True)
    alert_placeholder = st.empty()

st.markdown('<div class="section-header">🔢 Progress</div>', unsafe_allow_html=True)
progress_bar  = st.progress(0.0)
status_text   = st.empty()

st.markdown('<div class="section-header">📋 Frame-by-Frame Results</div>', unsafe_allow_html=True)
table_placeholder = st.empty()

# ── Output video writer ───────────────────────────────────────────────────────
out_mp4_path = tmp_path.replace(Path(tmp_path).suffix, "_annotated.mp4")
fourcc       = cv2.VideoWriter_fourcc(*"mp4v")
out_writer   = cv2.VideoWriter(out_mp4_path, fourcc, max(fps/frame_skip, 1), (W, H))

feat_buf  = deque(maxlen=seq_len)
csv_rows  = []
frame_idx = 0
proc_idx  = 0
t0_total  = time.time()

cap = cv2.VideoCapture(tmp_path)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % frame_skip != 0:
            continue
        proc_idx += 1

        t0         = time.time()
        pil_img    = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        feats      = extract_features(feat_model, pil_img)
        dets       = detect(yolo_model, frame)
        feat_buf.append(feats[0])

        risk, traj = forecast(gru_model, feat_buf)
        depth_map  = depth_infer(midas, midas_xf, frame)
        status, alert_msg = make_alert(dets, risk, depth_map, threshold)

        ann_rgb = annotate_frame(frame, dets, risk, threshold, alert_msg, status)
        out_writer.write(cv2.cvtColor(ann_rgb, cv2.COLOR_RGB2BGR))

        elapsed = round(time.time() - t0, 2)
        pct     = frame_idx / total

        # ── Update UI ─────────────────────────────────────────────
        frame_display.image(ann_rgb, channels="RGB", use_container_width=True)

        csv_rows.append({
            "Frame":       frame_idx,
            "Risk Score":  round(risk, 4),
            "Threshold":   round(threshold, 4),
            "Alert":       "⚠ YES" if status=="alert" else "✅ NO",
            "Detections":  ", ".join(d["label"] for d in dets) or "—",
            "Depth Min":   round(float(depth_map.min()), 1),
            "Depth Max":   round(float(depth_map.max()), 1),
            "Alert Msg":   alert_msg,
            "Time (s)":    elapsed,
        })

        # Risk chart
        df_chart = pd.DataFrame(csv_rows)[["Frame","Risk Score","Threshold"]]
        chart = (
            alt.Chart(df_chart.melt("Frame", var_name="Series", value_name="Value"))
            .mark_line(strokeWidth=2)
            .encode(
                x=alt.X("Frame:Q", title="Frame"),
                y=alt.Y("Value:Q", scale=alt.Scale(domain=[0,1])),
                color=alt.Color("Series:N", scale=alt.Scale(
                    domain=["Risk Score","Threshold"],
                    range=["#a78bfa","#f39c12"]
                )),
                tooltip=["Frame:Q","Series:N","Value:Q"],
            )
            .properties(height=220)
            .configure_view(stroke=None)
            .configure_axis(
                gridColor="rgba(255,255,255,0.07)",
                labelColor="#a0a0c0",
                titleColor="#a0a0c0",
            )
            .configure_legend(labelColor="#e0e0e0", titleColor="#e0e0e0")
        )
        chart_placeholder.altair_chart(chart, use_container_width=True)

        # Alert banner
        if status == "alert":
            alert_placeholder.markdown(
                f'<div class="alert-banner">⚠️ {alert_msg}</div>',
                unsafe_allow_html=True)
        else:
            alert_placeholder.markdown(
                '<div class="safe-banner">✅ Scene within safe parameters.</div>',
                unsafe_allow_html=True)

        progress_bar.progress(min(pct, 1.0))
        status_text.markdown(
            f"Processing frame **{frame_idx}/{total}** &nbsp;|&nbsp; "
            f"Risk: **{risk:.3f}** &nbsp;|&nbsp; "
            f"{'⚠️ ALERT' if status=='alert' else '✅ Safe'} &nbsp;|&nbsp; "
            f"{elapsed}s/frame"
        )

        # Live table (last 20 rows)
        df_show = pd.DataFrame(csv_rows[-20:])
        table_placeholder.dataframe(df_show, use_container_width=True, hide_index=True)

finally:
    cap.release()
    out_writer.release()

# ════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════════════════════════

progress_bar.progress(1.0)
elapsed_total = round(time.time() - t0_total, 1)
n_alerts      = sum(1 for r in csv_rows if "YES" in r["Alert"])
avg_risk      = round(np.mean([r["Risk Score"] for r in csv_rows]), 4) if csv_rows else 0
max_risk      = round(max((r["Risk Score"] for r in csv_rows), default=0), 4)

st.markdown("---")
st.markdown("## ✅ Analysis Complete")

m1, m2, m3, m4, m5 = st.columns(5)
for col, val, lbl, color in zip(
    [m1, m2, m3, m4, m5],
    [proc_idx, n_alerts, f"{avg_risk}", f"{max_risk}", f"{elapsed_total}s"],
    ["Frames Processed","Alerts Raised","Avg Risk Score","Peak Risk Score","Total Time"],
    ["#7b8fff","#e74c3c","#a78bfa","#f39c12","#2ed573"],
):
    col.markdown(f"""
    <div class='metric-card'>
        <div class='metric-value' style='color:{color}'>{val}</div>
        <div class='metric-label'>{lbl}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Full results table
st.markdown('<div class="section-header">📋 Full Results Table</div>', unsafe_allow_html=True)
df_full = pd.DataFrame(csv_rows)
st.dataframe(df_full, use_container_width=True, hide_index=True)

# ── Download buttons ──────────────────────────────────────────────────────────
st.markdown('<div class="section-header">💾 Downloads</div>', unsafe_allow_html=True)
dl1, dl2 = st.columns(2)

# CSV download
csv_buf = io.StringIO()
df_full.to_csv(csv_buf, index=False)
dl1.download_button(
    label="⬇️  Download Results CSV",
    data=csv_buf.getvalue().encode(),
    file_name="accident_anticipation_results.csv",
    mime="text/csv",
    use_container_width=True,
)

# Annotated video download
if os.path.exists(out_mp4_path):
    with open(out_mp4_path, "rb") as f:
        dl2.download_button(
            label="⬇️  Download Annotated Video",
            data=f.read(),
            file_name="annotated_dashcam.mp4",
            mime="video/mp4",
            use_container_width=True,
        )

# Cleanup temp files
try:
    os.remove(tmp_path)
    os.remove(out_mp4_path)
except Exception:
    pass
