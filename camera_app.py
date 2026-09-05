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
import treatments
from models.model_builder import get_model
from utils.dataset import inference_transform
from utils.ood import FeatureTap, get_detector, ACCEPT, REVIEW, REJECT


# ─── Disease severity → border colour ────────────────────────────────────────
HEALTHY_CLASS  = config.CLASS_NAMES.index("Tomato_healthy")
COLOR_HEALTHY  = (50, 205, 50)    # green
COLOR_DISEASE  = (0, 60, 255)     # red (BGR)
COLOR_LOW_CONF = (0, 180, 255)    # orange
COLOR_REJECT   = (128, 128, 128)  # grey - refusing to answer

# Running the abstention layer every frame would halve the frame rate for no
# benefit; the verdict barely changes between adjacent frames.
OOD_EVERY_N_FRAMES = 5


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
def draw_overlay(frame, label, confidence, fps, model_name, paused,
                 decision=ACCEPT, note="", advice=""):
    h, w = frame.shape[:2]

    # Colour carries the verdict, not just the class - a refused frame must not
    # look like a diagnosis just because it is moving.
    if decision == REJECT:
        color = COLOR_REJECT
    elif decision == REVIEW or confidence < config.CONFIDENCE_THRESHOLD:
        color = COLOR_LOW_CONF
    else:
        color = COLOR_HEALTHY if "healthy" in label.lower() else COLOR_DISEASE

    # Outer border
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 6)

    # Semi-transparent top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 108), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    if decision == REJECT:
        headline = "No tomato leaf detected"
    elif decision == REVIEW:
        headline = f"{label}?"
    else:
        headline = label

    cv2.putText(frame, headline, (14, 40),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 2, cv2.LINE_AA)

    if decision == REJECT:
        sub = note[:70] if note else "Point the camera at a single tomato leaf"
    else:
        sub = f"{confidence * 100:.1f}%   [{decision.upper()}]"
    cv2.putText(frame, sub, (14, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)

    # What to do about it, on the frame itself.
    if decision != REJECT and advice:
        cv2.putText(frame, advice[:78], (14, 96),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (170, 190, 175), 1, cv2.LINE_AA)

    # Bottom bar: FPS + model + controls
    cv2.rectangle(frame, (0, h - 40), (w, h), (20, 20, 20), -1)
    status = "PAUSED" if paused else f"FPS: {fps:.1f}"
    cv2.putText(frame, status, (14, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Model: {model_name}   [Q]uit [S]ave [M]odel [SPACE]pause",
                (160, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1, cv2.LINE_AA)

    # Confidence bar
    if decision != REJECT:
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
    detector   = get_detector(model_name)
    tap        = FeatureTap(model)
    if config.OOD_ENABLED and not detector.calibrated:
        print("  [!]  OOD detector on fallback thresholds - "
              "run  python calibrate_ood.py  to tune them.")

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
    decision   = ACCEPT
    note       = ""
    advice     = ""
    frame_count = 0
    save_dir    = Path("captures"); save_dir.mkdir(exist_ok=True)

    while True:
        t_start = time.time()

        ret, frame = cap.read()
        if not ret:
            print("  ⚠️  Frame grab failed."); break

        if not paused:
            frame_count += 1
            with torch.no_grad():
                tensor = preprocess_frame(frame)
                logits = model(tensor)
                probs  = F.softmax(logits, dim=1)[0]
                top_prob, top_idx = probs.max(dim=0)
                confidence = float(top_prob)
                top_class  = config.CLASS_NAMES[int(top_idx)]
                label      = config.DISPLAY_NAMES[int(top_idx)]
                advice     = treatments.summary_line(top_class)

            if config.OOD_ENABLED and frame_count % OOD_EVERY_N_FRAMES == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                verdict = detector.evaluate(Image.fromarray(rgb),
                                            logits[0], tap.get())
                decision = verdict.decision
                note = verdict.reasons[0] if verdict.reasons else ""

        elapsed = time.time() - t_start
        fps_queue.append(1.0 / max(elapsed, 1e-6))
        fps = sum(fps_queue) / len(fps_queue)

        display = frame.copy()
        draw_overlay(display, label, confidence, fps, model_name, paused,
                     decision, note, advice)
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
            tap.close()
            model      = load_inference_model(model_name)
            tap        = FeatureTap(model)
            detector   = get_detector(model_name)
        elif key == ord(" "):
            paused = not paused
            print("  ⏸ Paused" if paused else "  ▶️ Resumed")

    tap.close()
    cap.release()
    cv2.destroyAllWindows()
    print("\n  👋  Camera closed.")


if __name__ == "__main__":
    main()
