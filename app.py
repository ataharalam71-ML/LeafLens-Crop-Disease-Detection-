"""
app.py  —  Flask web application for Tomato Disease Detection.

Features:
  • Upload a leaf image  → get prediction + confidence bar + Grad-CAM
  • Live camera stream   → real-time prediction overlay via MJPEG endpoint
  • Model switcher       → choose between all trained models
  • REST API endpoints   → /predict  /models  /stream  /gradcam

Usage:
    python app.py
    python app.py --host 0.0.0.0 --port 5000
"""

import argparse
import base64
import io
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from flask import (Flask, Response, jsonify, render_template,
                   request, send_from_directory)
from PIL import Image

import config
from models.model_builder import get_model
from utils.dataset import inference_transform
from utils.gradcam import GradCAM, get_target_layer, run_gradcam

# ─── App setup ───────────────────────────────────────────────────────────────

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024   # 16 MB upload limit

UPLOAD_DIR = Path("uploads"); UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR = config.RESULTS_DIR

# ─── Model registry (lazy-loaded) ────────────────────────────────────────────

_models: dict = {}
_current_model_name: str = config.INFERENCE_MODEL
_lock = threading.Lock()


def get_available_models() -> list:
    """Models with a trained checkpoint — the ones that can actually be selected."""
    return [
        m for m in config.MODELS_TO_TRAIN
        if (config.MODELS_DIR / f"{m}_best.pth").exists()
    ]


def get_all_models() -> list:
    """
    Every registered architecture plus whether it has been trained yet.
    The UI dropdown lists all of them, greying out the ones with no checkpoint,
    so you can see the full roster instead of only what happens to be trained.
    """
    return [
        {"name": m, "trained": (config.MODELS_DIR / f"{m}_best.pth").exists()}
        for m in config.MODELS_TO_TRAIN
    ]


def load_model_cached(model_name: str):
    """Load and cache model; thread-safe."""
    global _models
    if model_name not in _models:
        ckpt_path = config.MODELS_DIR / f"{model_name}_best.pth"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"No checkpoint for '{model_name}'.")
        ckpt  = torch.load(ckpt_path, map_location=config.DEVICE, weights_only=False)
        model = get_model(model_name, config.NUM_CLASSES, pretrained=False)
        model.load_state_dict(ckpt["model_state"])
        model.to(config.DEVICE).eval()
        _models[model_name] = model
        print(f"  ✅  Loaded: {model_name}")
    return _models[model_name]


# ─── Inference helper ─────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(pil_img: Image.Image, model_name: str, topk: int = 3) -> dict:
    model  = load_model_cached(model_name)
    tensor = inference_transform(pil_img).unsqueeze(0).to(config.DEVICE)
    logits = model(tensor)
    probs  = F.softmax(logits, dim=1)[0]

    top_probs, top_idxs = probs.topk(topk)
    predictions = []
    for prob, idx in zip(top_probs.cpu(), top_idxs.cpu()):
        predictions.append({
            "class":      config.CLASS_NAMES[int(idx)],
            "display":    config.DISPLAY_NAMES[int(idx)],
            "confidence": round(float(prob), 4),
            "is_healthy": "healthy" in config.CLASS_NAMES[int(idx)].lower(),
        })
    return {
        "model":       model_name,
        "top":         predictions[0],
        "topk":        predictions,
        "device":      str(config.DEVICE),
    }


# ─── Camera streaming ────────────────────────────────────────────────────────

class CameraStream:
    """Background thread that grabs frames and annotates them."""

    def __init__(self):
        self.cap          = None
        self.frame        = None
        self.running      = False
        self.model_name   = _current_model_name
        self.label        = "No model loaded"
        self.confidence   = 0.0
        self.fps          = 0.0
        self._thread      = None
        self._lock        = threading.Lock()

    def start(self, camera_idx: int = config.CAMERA_INDEX):
        if self.running:
            return
        self.cap = cv2.VideoCapture(camera_idx)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_idx}")
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.cap = None

    def set_model(self, model_name: str):
        with self._lock:
            self.model_name = model_name

    def _loop(self):
        fps_times = []
        model = None
        while self.running:
            t0  = time.time()
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.05); continue

            # Inference
            try:
                with self._lock:
                    mn = self.model_name
                model  = load_model_cached(mn)
                rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil    = Image.fromarray(rgb)
                tensor = inference_transform(pil).unsqueeze(0).to(config.DEVICE)
                with torch.no_grad():
                    logits = model(tensor)
                    probs  = F.softmax(logits, dim=1)[0]
                    top_p, top_i = probs.max(dim=0)
                    self.label      = config.DISPLAY_NAMES[int(top_i)]
                    self.confidence = float(top_p)
                    is_healthy      = "healthy" in config.CLASS_NAMES[int(top_i)].lower()
            except Exception as e:
                self.label = f"Error: {e}"
                self.confidence = 0.0
                is_healthy = False

            # Annotate frame
            annotated = frame.copy()
            h, w = annotated.shape[:2]

            # Colour scheme
            if self.confidence >= config.CONFIDENCE_THRESHOLD:
                color = (50, 200, 50) if is_healthy else (30, 50, 220)
            else:
                color = (30, 160, 230)

            # Border
            cv2.rectangle(annotated, (0, 0), (w-1, h-1), color, 5)

            # Top bar
            overlay = annotated.copy()
            cv2.rectangle(overlay, (0, 0), (w, 70), (15, 15, 15), -1)
            cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)

            disp = self.label if self.confidence >= config.CONFIDENCE_THRESHOLD else "Low confidence …"
            cv2.putText(annotated, disp, (10, 32),
                        cv2.FONT_HERSHEY_DUPLEX, 0.85, color, 2, cv2.LINE_AA)
            cv2.putText(annotated, f"{self.confidence*100:.1f}%   [{mn}]",
                        (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (200, 200, 200), 1, cv2.LINE_AA)

            # Confidence bar
            bar_w = int(w * min(self.confidence, 1.0))
            cv2.rectangle(annotated, (0, h-5), (bar_w, h), color, -1)

            # FPS bottom-right
            elapsed = time.time() - t0
            fps_times.append(elapsed)
            if len(fps_times) > 30: fps_times.pop(0)
            self.fps = 1.0 / (sum(fps_times) / len(fps_times) + 1e-9)
            cv2.putText(annotated, f"FPS: {self.fps:.1f}",
                        (w - 100, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (150, 150, 150), 1, cv2.LINE_AA)

            with self._lock:
                self.frame = annotated.copy()

    def get_jpeg(self) -> bytes:
        with self._lock:
            if self.frame is None:
                return b""
            _, buf = cv2.imencode(".jpg", self.frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            return buf.tobytes()


camera_stream = CameraStream()


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    available = get_available_models()
    return render_template(
        "index.html",
        models=available,
        all_models=get_all_models(),
        current_model=_current_model_name,
        class_names=config.DISPLAY_NAMES,
        num_classes=config.NUM_CLASSES,
    )


# ── REST: list models ──
@app.route("/models", methods=["GET"])
def list_models():
    return jsonify({
        "available": get_available_models(),
        "all":       get_all_models(),
        "current":   _current_model_name,
    })


# ── REST: switch model ──
@app.route("/models/select", methods=["POST"])
def select_model():
    global _current_model_name
    data = request.get_json(silent=True) or {}
    name = data.get("model", "").strip()
    if name not in get_available_models():
        return jsonify({"error": f"Model '{name}' not available."}), 400
    _current_model_name = name
    camera_stream.set_model(name)
    return jsonify({"selected": name})


# ── REST: predict from uploaded image ──
@app.route("/predict", methods=["POST"])
def predict():
    global _current_model_name
    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    model_name = request.form.get("model", _current_model_name)

    try:
        pil_img = Image.open(file.stream).convert("RGB")
    except Exception:
        return jsonify({"error": "Cannot decode image."}), 400

    try:
        result = run_inference(pil_img, model_name, topk=3)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404

    # Encode thumbnail for response
    thumb = pil_img.copy()
    thumb.thumbnail((300, 300))
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=80)
    result["thumbnail"] = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

    return jsonify(result)


# ── REST: Grad-CAM for uploaded image ──
@app.route("/gradcam", methods=["POST"])
def gradcam_endpoint():
    global _current_model_name
    if "image" not in request.files:
        return jsonify({"error": "No image file."}), 400

    file       = request.files["image"]
    model_name = request.form.get("model", _current_model_name)
    class_idx  = request.form.get("class_idx", None)
    if class_idx is not None:
        class_idx = int(class_idx)

    # Save temp file
    tmp_path = UPLOAD_DIR / f"tmp_{int(time.time()*1000)}.jpg"
    try:
        file.save(str(tmp_path))
        out_path = str(RESULTS_DIR / f"gradcam_{int(time.time()*1000)}.png")
        run_gradcam(str(tmp_path), model_name, out_path, class_idx)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    # Return as base64
    with open(out_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    return jsonify({"gradcam": "data:image/png;base64," + encoded})


# ── Camera: start / stop ──
@app.route("/camera/start", methods=["POST"])
def camera_start():
    data     = request.get_json(silent=True) or {}
    cam_idx  = int(data.get("camera", config.CAMERA_INDEX))
    try:
        camera_stream.start(cam_idx)
        return jsonify({"status": "started"})
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500


@app.route("/camera/stop", methods=["POST"])
def camera_stop():
    camera_stream.stop()
    return jsonify({"status": "stopped"})


# ── Camera: MJPEG stream ──
@app.route("/stream")
def video_stream():
    def generate():
        while True:
            jpeg = camera_stream.get_jpeg()
            if jpeg:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
            time.sleep(0.033)   # ~30 fps cap

    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# ── Camera: current prediction JSON ──
@app.route("/camera/prediction")
def camera_prediction():
    return jsonify({
        "label":      camera_stream.label,
        "confidence": round(camera_stream.confidence, 4),
        "fps":        round(camera_stream.fps, 1),
        "model":      camera_stream.model_name,
    })


# ── Serve result images ──
@app.route("/results/<path:filename>")
def serve_result(filename):
    return send_from_directory(str(RESULTS_DIR), filename)


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host",   default=config.FLASK_HOST)
    parser.add_argument("--port",   type=int, default=config.FLASK_PORT)
    parser.add_argument("--debug",  action="store_true", default=config.FLASK_DEBUG)
    parser.add_argument("--camera", type=int, default=-1,
                        help="Camera index to auto-start (-1 = don't auto-start)")
    args = parser.parse_args()

    # Pre-load default model
    available = get_available_models()
    if available:
        _current_model_name = (
            config.INFERENCE_MODEL
            if config.INFERENCE_MODEL in available
            else available[0]
        )
        try:
            load_model_cached(_current_model_name)
        except Exception as e:
            print(f"  ⚠️  Could not pre-load model: {e}")
    else:
        print("  ⚠️  No trained models found. Run  python train.py  first.")

    if args.camera >= 0:
        try:
            camera_stream.start(args.camera)
            print(f"  🎥  Camera {args.camera} started.")
        except RuntimeError as e:
            print(f"  ⚠️  {e}")

    print(f"\n  🌐  Starting Flask app  →  http://localhost:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug,
            threaded=True, use_reloader=False)
