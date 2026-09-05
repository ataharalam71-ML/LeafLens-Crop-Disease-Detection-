"""
app.py  —  Flask web application for Tomato Disease Detection.

Features:
  • Upload a leaf image  → prediction + confidence + Grad-CAM + treatment plan
  • Abstention layer     → refuses to diagnose non-leaf / unusable images
  • Severity estimate    → how much of the leaf is symptomatic
  • Live camera          → captured in the VISITOR's browser and POSTed to
                           /predict. A webcam opened server-side would be the
                           HOST's camera, not theirs, so it cannot work over
                           the internet; camera_app.py still does native
                           OpenCV capture for local desktop use.
  • Model switcher       → choose between all trained models
  • REST API             → /predict /treatment /severity /models /gradcam /ood/status

Usage:
    python app.py
    python app.py --host 0.0.0.0 --port 5000
"""

import argparse
import base64
import io
import json
import os
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
import treatments
from models.model_builder import get_model
from utils.dataset import inference_transform
from utils.gradcam import GradCAM, get_target_layer, run_gradcam
from utils.ood import FeatureTap, get_detector
from utils.severity import estimate_severity

# ─── App setup ───────────────────────────────────────────────────────────────

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024   # 16 MB upload limit

UPLOAD_DIR = Path("uploads"); UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR = config.RESULTS_DIR

# ─── Model registry (lazy-loaded) ────────────────────────────────────────────

_models: dict = {}
_taps: dict = {}
_current_model_name: str = config.INFERENCE_MODEL
_lock = threading.Lock()


def get_available_models() -> list:
    """Models with a trained checkpoint — the ones that can actually be selected."""
    return [
        m for m in config.MODELS_TO_TRAIN
        if (config.MODELS_DIR / f"{m}_best.pth").exists()
    ]


def load_test_accuracies() -> dict:
    """
    Per-model test accuracy from evaluate.py, re-read each call so the UI picks
    up a fresh evaluation without a restart. Missing file -> empty dict, and the
    UI shows "n/a" rather than inventing a number.
    """
    path = config.RESULTS_DIR / "test_results.json"
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:                                       # noqa: BLE001
        return {}


def get_all_models() -> list:
    """
    Every registered architecture, whether it has been trained, and what it
    actually scored on the test split.

    The accuracy travels with the model because it differs enormously between
    them - VGG16 and AlexNet collapsed to ~20% while the rest sit near 99.8%.
    Showing one figure for all of them would be a straightforward misreport.
    """
    acc = load_test_accuracies()
    return [
        {
            "name": m,
            "trained": (config.MODELS_DIR / f"{m}_best.pth").exists(),
            "test_accuracy": acc.get(m),
        }
        for m in config.MODELS_TO_TRAIN
    ]


def load_model_cached(model_name: str):
    """
    Load and cache a model plus its penultimate-feature tap; thread-safe.

    The tap is attached once and kept for the model's lifetime — re-registering
    a forward hook on every request would leak handles.
    """
    with _lock:
        if model_name not in _models:
            ckpt_path = config.MODELS_DIR / f"{model_name}_best.pth"
            if not ckpt_path.exists():
                raise FileNotFoundError(f"No checkpoint for '{model_name}'.")
            ckpt = torch.load(ckpt_path, map_location=config.DEVICE, weights_only=False)
            model = get_model(model_name, config.NUM_CLASSES, pretrained=False)
            model.load_state_dict(ckpt["model_state"])
            model.to(config.DEVICE).eval()
            _models[model_name] = model
            _taps[model_name] = FeatureTap(model)
            print(f"  ✅  Loaded: {model_name}")
        return _models[model_name], _taps[model_name]


# ─── Inference helper ─────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(pil_img: Image.Image, model_name: str, topk: int = 3,
                  with_advice: bool = True) -> dict:
    """
    Classify one image, decide whether the answer is trustworthy, and — if it
    is — attach the severity estimate and the treatment plan.
    """
    model, tap = load_model_cached(model_name)

    tensor = inference_transform(pil_img).unsqueeze(0).to(config.DEVICE)
    # The tap is shared across threads, so grab logits and features together
    # under the lock rather than letting a second request overwrite them.
    with _lock:
        logits = model(tensor)
        features = tap.get().clone()

    probs = F.softmax(logits, dim=1)[0]

    top_probs, top_idxs = probs.topk(min(topk, config.NUM_CLASSES))
    predictions = []
    for prob, idx in zip(top_probs.cpu(), top_idxs.cpu()):
        predictions.append({
            "class":      config.CLASS_NAMES[int(idx)],
            "display":    config.DISPLAY_NAMES[int(idx)],
            "confidence": round(float(prob), 4),
            "is_healthy": "healthy" in config.CLASS_NAMES[int(idx)].lower(),
        })

    result = {
        "model":     model_name,
        "top":       predictions[0],
        "topk":      predictions,
        "device":    str(config.DEVICE),
        "verdict":   None,
        "severity":  None,
        "treatment": None,
    }

    verdict = None
    if config.OOD_ENABLED:
        verdict = get_detector(model_name).evaluate(pil_img, logits[0], features)
        result["verdict"] = verdict.to_dict()

    # Never hand out a spray programme for an image we just refused to diagnose.
    if with_advice and (verdict is None or verdict.answerable):
        top_class = predictions[0]["class"]
        sev = estimate_severity(pil_img, top_class)
        result["severity"] = sev
        result["treatment"] = treatments.get_treatment(
            top_class,
            severity=sev["fraction"] if sev["reliable"] else None,
            confidence=predictions[0]["confidence"],
        )

    return result


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
    return jsonify({"selected": name})


# ── REST: how the abstention layer is configured ──
@app.route("/ood/status", methods=["GET"])
def ood_status():
    """
    Lets the UI say honestly whether thresholds are calibrated or fallback.
    An uncalibrated demo should never look like a tuned one.
    """
    name = request.args.get("model", _current_model_name)
    status = get_detector(name).status()
    status["enabled"] = config.OOD_ENABLED
    return jsonify(status)


# ── REST: predict from uploaded image ──
@app.route("/predict", methods=["POST"])
def predict():
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


# ── REST: treatment plan for a class, without needing an image ──
@app.route("/treatment/<path:class_name>", methods=["GET"])
def treatment_lookup(class_name):
    """
    Browse the advisory for any class directly — used by the disease-class
    legend in the UI, and useful as a standalone reference endpoint.
    """
    severity = request.args.get("severity", type=float)
    try:
        return jsonify(treatments.get_treatment(class_name, severity=severity))
    except KeyError:
        return jsonify({"error": f"Unknown class '{class_name}'."}), 404


# ── REST: severity mask overlay for an uploaded image ──
@app.route("/severity", methods=["POST"])
def severity_endpoint():
    """Returns the segmentation the severity number was computed from."""
    if "image" not in request.files:
        return jsonify({"error": "No image file."}), 400
    try:
        pil_img = Image.open(request.files["image"].stream).convert("RGB")
    except Exception:
        return jsonify({"error": "Cannot decode image."}), 400

    class_name = request.form.get("class_name") or None
    res = estimate_severity(pil_img, class_name, return_mask=True)
    overlay = res.pop("overlay")

    ok, buf = cv2.imencode(".png", overlay)
    if ok:
        res["overlay"] = ("data:image/png;base64,"
                          + base64.b64encode(buf.tobytes()).decode())
    res["band"] = treatments.severity_band(res["fraction"])
    return jsonify(res)


# ── REST: Grad-CAM for uploaded image ──
@app.route("/gradcam", methods=["POST"])
def gradcam_endpoint():
    if "image" not in request.files:
        return jsonify({"error": "No image file."}), 400

    file       = request.files["image"]
    model_name = request.form.get("model", _current_model_name)
    class_idx  = request.form.get("class_idx", None)
    if class_idx is not None:
        class_idx = int(class_idx)

    tmp_path = UPLOAD_DIR / f"tmp_{int(time.time()*1000)}.jpg"
    out_path = None
    try:
        file.save(str(tmp_path))
        out_path = str(RESULTS_DIR / f"gradcam_{int(time.time()*1000)}.png")
        run_gradcam(str(tmp_path), model_name, out_path, class_idx)
    except Exception as e:                                  # noqa: BLE001
        return jsonify({"error": str(e)}), 500
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    with open(out_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    return jsonify({"gradcam": "data:image/png;base64," + encoded})


# ── Serve result images ──
@app.route("/results/<path:filename>")
def serve_result(filename):
    return send_from_directory(str(RESULTS_DIR), filename)


# --- Startup ------------------------------------------------------------------

def warmup() -> None:
    """
    Pick the model to serve and load it before the first request.

    Under gunicorn nothing in `__main__` runs, so without this the first
    visitor pays the full PyTorch load (tens of seconds on a free CPU tier)
    and may time out. Guarded by TOMATO_PRELOAD so importing app.py in tests
    stays instant.
    """
    global _current_model_name

    available = get_available_models()
    if not available:
        print("  [!]  No trained models found. Run  python train.py  first.")
        return

    _current_model_name = (
        config.INFERENCE_MODEL
        if config.INFERENCE_MODEL in available
        else available[0]
    )
    try:
        load_model_cached(_current_model_name)
    except Exception as e:                                  # noqa: BLE001
        print(f"  [!]  Could not pre-load model: {e}")

    det = get_detector(_current_model_name)
    if det.calibrated:
        print(f"  [ok] OOD detector calibrated (T={det.temperature:.3f})")
    else:
        print("  [!]  OOD detector on fallback thresholds - "
              "run  python calibrate_ood.py  to tune them.")


if os.environ.get("TOMATO_PRELOAD") == "1":
    warmup()


# --- Entry point --------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host",   default=config.FLASK_HOST)
    parser.add_argument("--port",   type=int, default=config.FLASK_PORT)
    parser.add_argument("--debug",  action="store_true", default=config.FLASK_DEBUG)
    args = parser.parse_args()

    warmup()

    print(f"\n  Starting Flask app  ->  http://localhost:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug,
            threaded=True, use_reloader=False)
