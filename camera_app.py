"""
camera_app.py  —  Real-time tomato disease detection via webcam.

Controls:
    Q / ESC  → quit
    S        → save current frame as PNG
    M        → toggle model (cycles through available models)
    SPACE    → pause / resume

Usage:
    python camera_app.py
    python camera_app.py --model MobileNetV3   (fastest)
    python camera_app.py --camera 1            (external webcam)
"""

import argparse
import time
from pathlib import Path
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import config
from models.model_builder import get_model
from utils.dataset import inference_transform


# ─── Disease severity → border colour ────────────────────────────────────────
HEALTHY_CLASS  = config.CLASS_NAMES.index("Tomato_healthy")
COLOR_HEALTHY  = (50, 205, 50)    # green
COLOR_DISEASE  = (0, 60, 255)     # red (BGR)
COLOR_LOW_CONF = (0, 180, 255)    # orange


# ─── Model loader ─────────────────────────────────────────────────────────────
def load_inference_model(model_name: str):
    ckpt_path = config.MODELS_DIR / f"{model_name}_best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No checkpoint for '{model_name}'. Run  python train.py  first."
        )
    ckpt  = torch.load(ckpt_path, map_location=config.DEVICE, weights_only=False)
    model = get_model(model_name, config.NUM_CLASSES, pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(config.DEVICE).eval()
    print(f"  ✅  Loaded model: {model_name}")
    return model


# ─── Preprocessing ────────────────────────────────────────────────────────────
def preprocess_frame(frame_bgr: np.ndarray) -> torch.Tensor:
    """Convert OpenCV BGR frame → normalised tensor."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    return inference_transform(pil).unsqueeze(0).to(config.DEVICE)


# ─── Overlay helpers ──────────────────────────────────────────────────────────
def draw_overlay(frame, label, confidence, fps, model_name, paused):
    h, w = frame.shape[:2]
    if confidence >= config.CONFIDENCE_THRESHOLD:
        color = COLOR_HEALTHY if "healthy" in label.lower() else COLOR_DISEASE
    else:
        color = COLOR_LOW_CONF

    # Outer border
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 6)

    # Semi-transparent top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # Main prediction text
    conf_pct = f"{confidence * 100:.1f}%"
    if confidence < config.CONFIDENCE_THRESHOLD:
        label = "Low confidence …"
    cv2.putText(frame, label, (14, 40),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 2, cv2.LINE_AA)
    cv2.putText(frame, conf_pct, (14, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 1, cv2.LINE_AA)

    # Bottom bar: FPS + model + controls
    cv2.rectangle(frame, (0, h - 40), (w, h), (20, 20, 20), -1)
    status = "⏸ PAUSED" if paused else f"FPS: {fps:.1f}"
    cv2.putText(frame, status, (14, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Model: {model_name}   [Q]uit [S]ave [M]odel [SPACE]pause",
                (160, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1, cv2.LINE_AA)

    # Confidence bar
    bar_w = int(w * min(confidence, 1.0))
    cv2.rectangle(frame, (0, h - 44), (bar_w, h - 41), color, -1)


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default=config.INFERENCE_MODEL)
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX)
    args = parser.parse_args()

    # Available models (only those with saved checkpoints)
    available_models = [
        m for m in config.MODELS_TO_TRAIN
        if (config.MODELS_DIR / f"{m}_best.pth").exists()
    ]
    if not available_models:
        print("  ❌  No trained models found. Run  python train.py  first.")
        return

    current_model_idx = (
        available_models.index(args.model)
        if args.model in available_models else 0
    )

    model      = load_inference_model(available_models[current_model_idx])
    model_name = available_models[current_model_idx]

    # Open webcam
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.CAMERA_FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          config.CAMERA_FPS)

    if not cap.isOpened():
        print(f"  ❌  Cannot open camera {args.camera}")
        return

    print("\n  🎥  Camera started.")
    print("  Controls: Q/ESC = quit | S = save frame | M = switch model | SPACE = pause\n")

    fps_queue = deque(maxlen=30)
    paused    = False
    label     = "Initialising …"
    confidence = 0.0
    frame_count = 0
    save_dir    = Path("captures"); save_dir.mkdir(exist_ok=True)

    while True:
        t_start = time.time()

        ret, frame = cap.read()
        if not ret:
            print("  ⚠️  Frame grab failed."); break

        if not paused:
            frame_count += 1
            # Run inference every frame (or every 2nd for slower GPUs)
            if frame_count % 1 == 0:
                with torch.no_grad():
                    tensor = preprocess_frame(frame)
                    logits = model(tensor)
                    probs  = F.softmax(logits, dim=1)[0]
                    top_prob, top_idx = probs.max(dim=0)
                    confidence = float(top_prob)
                    label      = config.DISPLAY_NAMES[int(top_idx)]

        elapsed = time.time() - t_start
        fps_queue.append(1.0 / max(elapsed, 1e-6))
        fps = sum(fps_queue) / len(fps_queue)

        display = frame.copy()
        draw_overlay(display, label, confidence, fps, model_name, paused)
        cv2.imshow("🍅 Tomato Disease Detector", display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):   # Q or ESC
            break
        elif key == ord("s"):
            fname = save_dir / f"frame_{int(time.time())}.png"
            cv2.imwrite(str(fname), frame)
            print(f"  💾  Saved: {fname}")
        elif key == ord("m"):
            current_model_idx = (current_model_idx + 1) % len(available_models)
            model_name = available_models[current_model_idx]
            model      = load_inference_model(model_name)
        elif key == ord(" "):
            paused = not paused
            print("  ⏸ Paused" if paused else "  ▶️ Resumed")

    cap.release()
    cv2.destroyAllWindows()
    print("\n  👋  Camera closed.")


if __name__ == "__main__":
    main()
