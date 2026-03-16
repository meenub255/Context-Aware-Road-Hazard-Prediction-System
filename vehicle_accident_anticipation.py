
import os, csv, time, warnings, textwrap
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import cv2
from PIL import Image

warnings.filterwarnings("ignore")

DEVICE = "cpu"   # Force CPU — safe on Colab free tier and local machines
print(f"[INFO] Running on: {DEVICE.upper()}")


# ════════════════════════════════════════════════════════════════
# MODULE 1 — Visual Scene Understanding
#   • MobileNetV2 (INT8 dynamic quantization — works on CPU)
#   • YOLOv8-nano  (loaded ONCE at start, not per frame)
# ════════════════════════════════════════════════════════════════

class FeatureExtractor(nn.Module):
    """MobileNetV2 feature backbone → 1280-dim vector per frame."""
    def __init__(self):
        super().__init__()
        import torchvision.models as models
        backbone     = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        self.features = backbone.features
        self.pool     = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return x.flatten(1)   # (B, 1280)


def build_feature_extractor():
    """INT8 dynamic quantisation — no calibration data needed, pure CPU."""
    model = FeatureExtractor().eval()
    model = torch.quantization.quantize_dynamic(
        model, {nn.Conv2d, nn.Linear}, dtype=torch.qint8
    )
    print("[Module 1] MobileNetV2 feature extractor ready (INT8 dynamic).")
    return model


IMG_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
])


def extract_features(model, pil_img: Image.Image) -> np.ndarray:
    tensor = IMG_TRANSFORM(pil_img).unsqueeze(0)
    with torch.inference_mode():
        feats = model(tensor)   # (1, 1280)
    return feats.numpy()


def build_yolo():
    """Load YOLOv8-nano exactly ONCE. Reuse across all frames."""
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")    # ~6 MB, auto-downloads
    print("[Module 1] YOLOv8-nano loaded.")
    return model


def detect_objects(yolo_model, frame_bgr: np.ndarray) -> list:
    """
    Runs detection on a BGR numpy array (no file I/O needed).
    Returns list of {label, confidence, bbox}.
    """
    results = yolo_model(frame_bgr, verbose=False)[0]
    dets = []
    for box in results.boxes:
        dets.append({
            "label":      results.names[int(box.cls)],
            "confidence": float(box.conf),
            "bbox":       box.xyxy[0].tolist(),
        })
    return dets


# ════════════════════════════════════════════════════════════════
# MODULE 2 — Object Motion Patterns & Vehicle Dynamics
#   Bidirectional GRU (INT8 dynamic) for risk + trajectory
# ════════════════════════════════════════════════════════════════

class BiGRUForecaster(nn.Module):
    """Bi-GRU that maps a feature sequence → risk score + (dx,dy) offset."""
    def __init__(self, input_dim=1280, hidden=256, layers=2):
        super().__init__()
        self.bigru = nn.GRU(input_dim, hidden, layers,
                            batch_first=True, bidirectional=True,
                            dropout=0.2 if layers > 1 else 0.0)
        d = hidden * 2
        self.risk_head = nn.Sequential(nn.Linear(d, 64), nn.ReLU(),
                                       nn.Linear(64, 1), nn.Sigmoid())
        self.traj_head = nn.Sequential(nn.Linear(d, 64), nn.ReLU(),
                                       nn.Linear(64, 2))

    def forward(self, x):           # x: (1, T, D)
        out, _ = self.bigru(x)      # (1, T, 2H)
        last   = out[:, -1, :]
        return self.risk_head(last), self.traj_head(last)


def build_bigru():
    model = BiGRUForecaster().eval()
    model = torch.quantization.quantize_dynamic(
        model, {nn.GRU, nn.Linear}, dtype=torch.qint8
    )
    print("[Module 2] BiGRU forecaster ready (INT8 dynamic).")
    return model


def forecast(model, feat_buffer: deque):
    """feat_buffer: deque of (1280,) numpy arrays."""
    seq = torch.tensor(np.stack(list(feat_buffer)), dtype=torch.float32).unsqueeze(0)
    with torch.inference_mode():
        risk, traj = model(seq)
    return risk.item(), traj.squeeze().tolist()


# ════════════════════════════════════════════════════════════════
# MODULE 3 — Road Environment Context
#   DistilBERT (tiny, ~66 MB) — works great on CPU
#   + Dynamic Risk Threshold
# ════════════════════════════════════════════════════════════════

def build_context_encoder():
    """
    DistilBERT instead of BERT-base — 40% smaller, same quality for
    short road-context phrases, and no GPU required.
    """
    from transformers import AutoTokenizer, AutoModel
    MODEL = "distilbert-base-uncased"
    tok   = AutoTokenizer.from_pretrained(MODEL)
    # FP32 on CPU — safe and stable
    model = AutoModel.from_pretrained(MODEL).eval()
    # Dynamic INT8 on linear layers
    model = torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )
    print("[Module 3] DistilBERT context encoder ready (INT8 dynamic).")
    return tok, model


def encode_context(tok, model, text: str) -> np.ndarray:
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=64)
    with torch.inference_mode():
        out = model(**inputs)
    return out.last_hidden_state[:, 0, :].numpy()   # (1, 768)


class DynamicThreshold:
    """Lower threshold = more sensitive. Rises on open roads, falls in dense scenes."""
    def compute(self, complexity: float) -> float:
        t = float(np.clip(0.50 - 0.30 * (complexity - 0.5), 0.25, 0.75))
        print(f"[Module 3] Complexity {complexity:.2f} → Threshold {t:.2f}")
        return t


# ════════════════════════════════════════════════════════════════
# MODULE 4 — Depth Estimation  (MiDaS small, ~100 MB)
#   VLM REMOVED on CPU — replaced by rule-based alert generator
#   (BLIP-2 2.7B can't run on CPU without crashing)
# ════════════════════════════════════════════════════════════════

def build_depth_model():
    midas     = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True).eval()
    xforms    = torch.hub.load("intel-isl/MiDaS", "transforms",  trust_repo=True)
    transform = xforms.small_transform
    print("[Module 4] MiDaS depth estimator ready.")
    return midas, transform


def estimate_depth(midas, transform, frame_bgr: np.ndarray) -> np.ndarray:
    rgb     = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    inp     = transform(rgb)                          # (1,3,H,W) or (3,H,W)
    with torch.inference_mode():
        depth = midas(inp)
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1) if depth.dim() == 3 else depth,
            size=rgb.shape[:2], mode="bicubic", align_corners=False
        ).squeeze().numpy()
    return depth


def rule_based_alert(detections: list, risk: float,
                     depth_map: np.ndarray, threshold: float) -> str:
    """
    Lightweight rule engine that replaces the VLM on CPU.
    Produces a specific, spatially-aware alert string.
    """
    if risk < threshold:
        return "Scene within safe parameters."

    # Find the closest (highest depth value = nearest in MiDaS convention)
    nearest_label, nearest_dist_m = "object", "?"
    if detections:
        best_score = -1
        h, w = depth_map.shape
        for det in detections:
            b  = [int(v) for v in det["bbox"]]
            cx = min(max((b[0] + b[2]) // 2, 0), w - 1)
            cy = min(max((b[1] + b[3]) // 2, 0), h - 1)
            d  = depth_map[cy, cx]
            if d > best_score:
                best_score     = d
                nearest_label  = det["label"]
                nearest_dist_m = round(d / 50.0, 1)   # rough calibration

    # Severity bucket
    if   risk >= 0.80: severity = "CRITICAL"
    elif risk >= 0.65: severity = "WARNING"
    else:              severity = "CAUTION"

    return (f"[{severity}] Risk {risk:.2f} — {nearest_label} detected "
            f"~{nearest_dist_m} m ahead. Recommend braking / evasive action.")


# ════════════════════════════════════════════════════════════════
# SYSTEM CLASS — Orchestrates all four modules
# ════════════════════════════════════════════════════════════════

class AccidentAnticipationSystem:
    def __init__(self):
        print("=" * 60)
        print("  Accident Anticipation System — CPU Mode")
        print("=" * 60)
        self.feat_model  = build_feature_extractor()
        self.yolo        = build_yolo()
        self.gru         = build_bigru()
        self.tok, self.bert = build_context_encoder()
        self.midas, self.depth_xform = build_depth_model()
        self.threshold_engine = DynamicThreshold()
        print("\n[System] All modules ready.\n")


# ════════════════════════════════════════════════════════════════
# VIDEO PIPELINE
# ════════════════════════════════════════════════════════════════

def process_video(
    sys: AccidentAnticipationSystem,
    video_path:       str,
    road_annotation:  str   = "Traffic density: Dense. Weather: Clear. Road: Urban.",
    scene_complexity: float = 0.65,
    frame_skip:       int   = 5,
    seq_len:          int   = 8,
    save_output:      bool  = True,
):
    """
    Process a dashcam video frame by frame.

    Args
    ----
    video_path       : full path to .mp4 / .avi / .mov
    road_annotation  : text describing the road environment
    scene_complexity : 0.0 = open highway, 1.0 = dense urban/rain
    frame_skip       : analyse every Nth frame  (5 recommended on CPU)
    seq_len          : BiGRU temporal window (number of processed frames)
    save_output      : write annotated video + CSV; set False to skip
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS)   or 25.0
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[Video] {os.path.basename(video_path)}  {W}×{H} @ {fps:.1f} fps  ({total} frames)")
    print(f"        Analysing every {frame_skip}th frame | seq_len={seq_len}\n")

    base    = os.path.splitext(video_path)[0]
    out_mp4 = base + "_annotated.mp4"
    out_csv = base + "_results.csv"

    out_writer = None
    if save_output:
        fourcc     = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(out_mp4, fourcc, fps / frame_skip, (W, H))

    # Pre-compute BERT embedding and threshold — same for whole video
    print("[Video] Encoding road context …")
    encode_context(sys.tok, sys.bert, road_annotation)   # warm-up
    threshold = sys.threshold_engine.compute(scene_complexity)

    feat_buf  = deque(maxlen=seq_len)
    csv_rows  = []
    frame_idx = 0
    proc_idx  = 0
    t0_total  = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % frame_skip != 0:
                continue

            proc_idx += 1
            t0 = time.time()

            # ── Module 1 ──────────────────────────────────────────
            pil_img    = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            feats      = extract_features(sys.feat_model, pil_img)   # (1, 1280)
            detections = detect_objects(sys.yolo, frame)
            feat_buf.append(feats[0])

            # ── Module 2 ──────────────────────────────────────────
            risk_score, traj = forecast(sys.gru, feat_buf)

            # ── Module 3 ──────────────────────────────────────────
            alert_flag = risk_score >= threshold

            # ── Module 4 ──────────────────────────────────────────
            depth_map = estimate_depth(sys.midas, sys.depth_xform, frame)
            alert_msg = rule_based_alert(detections, risk_score, depth_map, threshold)

            elapsed = round(time.time() - t0, 2)
            status  = "⚠ ALERT" if alert_flag else "SAFE"
            print(f"  Frame {frame_idx:>5}/{total}  risk={risk_score:.3f}  "
                  f"thresh={threshold:.2f}  [{status}]  {elapsed}s  "
                  f"dets={[d['label'] for d in detections]}")

            # ── Annotate frame ────────────────────────────────────
            ann   = frame.copy()
            color = (0, 0, 220) if alert_flag else (0, 210, 0)

            for det in detections:
                x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
                cv2.rectangle(ann, (x1, y1), (x2, y2), color, 2)
                cv2.putText(ann,
                            f"{det['label']} {det['confidence']:.0%}",
                            (x1, max(y1 - 8, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            # Top HUD bar
            cv2.rectangle(ann, (0, 0), (W, 52), (20, 20, 20), -1)
            cv2.putText(ann,
                        f"Risk: {risk_score:.3f}  |  Threshold: {threshold:.2f}  |  {status}",
                        (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2)

            # Bottom alert bar (only when alert)
            if alert_flag:
                cv2.rectangle(ann, (0, H - 48), (W, H), (0, 0, 160), -1)
                cv2.putText(ann,
                            alert_msg[:95],
                            (10, H - 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)

            if out_writer:
                out_writer.write(ann)

            # ── CSV row ───────────────────────────────────────────
            csv_rows.append({
                "frame":        frame_idx,
                "risk_score":   round(risk_score, 4),
                "threshold":    round(threshold, 4),
                "alert":        int(alert_flag),
                "status":       status,
                "detections":   "|".join(d["label"] for d in detections),
                "depth_min":    round(float(depth_map.min()), 2),
                "depth_max":    round(float(depth_map.max()), 2),
                "alert_msg":    alert_msg,
                "frame_time_s": elapsed,
            })

    finally:
        cap.release()
        if out_writer:
            out_writer.release()

    # Write CSV
    if save_output and csv_rows:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            w.writeheader()
            w.writerows(csv_rows)

    n_alerts = sum(r["alert"] for r in csv_rows)
    elapsed_total = round(time.time() - t0_total, 1)
    print(f"\n[Done] {proc_idx} frames processed in {elapsed_total}s")
    print(f"       Alerts raised: {n_alerts}/{proc_idx}")
    if save_output:
        print(f"       Annotated video → {out_mp4}")
        print(f"       Results CSV    → {out_csv}")
    return csv_rows


# ════════════════════════════════════════════════════════════════
# ENTRY POINT  ← SET YOUR VIDEO PATH HERE
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ┌──────────────────────────────────────────────────────────────┐
    # │  ✏️  PUT YOUR VIDEO PATH HERE                               │
    # │                                                              │
    # │  Windows : r"D:\Videos\my_dashcam.mp4"                     │
    # │  Colab   : "/content/my_dashcam.mp4"                        │
    # │  Or use  : "sample.mp4"  if the file is in the same folder  │
    # └──────────────────────────────────────────────────────────────┘
    VIDEO_PATH = r"D:\Videos\dashcam.mp4"    # ← CHANGE THIS

    # Describe your video's road environment
    ROAD_ANNOTATION = (
        "Traffic density: Dense. Weather: Clear. "
        "Road type: Urban city roads with pedestrians and bikes."
    )

    # Scene complexity  [0.0 – 1.0]
    #   0.3 → open highway (high threshold, fewer alerts)
    #   0.5 → suburban / mixed
    #   0.7 → dense city
    #   0.9 → rain / night / heavy traffic
    SCENE_COMPLEXITY = 0.70

    # Process every Nth frame
    #   CPU suggestion: 10  (fast)  or  5  (more detail)
    #   Colab T4 GPU : 3   (real-time-ish)
    FRAME_SKIP = 10

    # BiGRU look-back window (processed frames kept in memory)
    SEQ_LEN = 8

    # ── Initialise & run ─────────────────────────────────────────
    system = AccidentAnticipationSystem()

    results = process_video(
        sys              = system,
        video_path       = VIDEO_PATH,
        road_annotation  = ROAD_ANNOTATION,
        scene_complexity = SCENE_COMPLEXITY,
        frame_skip       = FRAME_SKIP,
        seq_len          = SEQ_LEN,
        save_output      = True,
    )
