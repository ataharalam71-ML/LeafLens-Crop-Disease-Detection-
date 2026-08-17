"""
config.py — Central configuration for Tomato Disease CNN Project
"""

import os
from pathlib import Path

import console_setup  # noqa: F401  — makes emoji/box-drawing output safe on cp1252 terminals

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent


def _resolve_data_dir() -> Path:
    """
    Find the PlantVillage dataset without hardcoding one machine's layout.
    Priority:
        1. TOMATO_DATA_DIR environment variable   (use this on a new device)
        2. <project>/PlantVillage                 (dataset shipped next to code)
        3. ~/Desktop/PlantVillage                 (original dev machine layout)
    Falls back to (2) so error messages point somewhere sensible.
    """
    env = os.environ.get("TOMATO_DATA_DIR")
    if env:
        return Path(env).expanduser()

    candidates = [
        BASE_DIR / "PlantVillage",
        Path.home() / "Desktop" / "PlantVillage",
    ]
    for c in candidates:
        if c.is_dir():
            return c
    return candidates[0]


DATA_DIR      = _resolve_data_dir()
MODELS_DIR    = BASE_DIR / "saved_models"
LOGS_DIR      = BASE_DIR / "logs"
RESULTS_DIR   = BASE_DIR / "results"

for d in [MODELS_DIR, LOGS_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Classes ───────────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "Tomato__Target_Spot",
    "Tomato__Tomato_mosaic_virus",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_healthy",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
]

DISPLAY_NAMES = [
    "Target Spot",
    "Mosaic Virus",
    "Yellow Leaf Curl Virus",
    "Bacterial Spot",
    "Early Blight",
    "Healthy ✓",
    "Late Blight",
    "Leaf Mold",
    "Septoria Leaf Spot",
    "Spider Mites",
]

NUM_CLASSES = len(CLASS_NAMES)

# ─── Training ──────────────────────────────────────────────────────────────────
IMAGE_SIZE   = 224          # Input image size (224×224)

# Raise on a big GPU, lower if you hit CUDA out-of-memory — no code edit needed:
#   set TOMATO_BATCH_SIZE=64        (Windows)
#   export TOMATO_BATCH_SIZE=8      (Linux/macOS)
BATCH_SIZE   = int(os.environ.get("TOMATO_BATCH_SIZE", 32))
NUM_EPOCHS   = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-4
PATIENCE      = 7           # Early stopping patience

# Train / Val / Test split
TRAIN_SPLIT = 0.70
VAL_SPLIT   = 0.15
TEST_SPLIT  = 0.15

# ─── Models to train ──────────────────────────────────────────────────────────
# Set to True/False to enable/disable each model.  All 8 are enabled.
MODELS_TO_TRAIN = {
    "CustomCNN":        True,   # Built from scratch (fast baseline)
    "ResNet50":         True,   # Transfer learning
    "EfficientNetB0":   True,   # Transfer learning (lightweight)
    "MobileNetV3":      True,   # Transfer learning (edge-friendly)
    "VGG16":            True,   # Transfer learning (heavy — needs a GPU)
    "AlexNet":          True,   # Transfer learning (classic, very fast)
    "UNet":             True,   # Built from scratch (encoder/decoder + skips)
    "DenseNet121":      True,   # Transfer learning (strong on leaf textures)
}

# ─── Augmentation ─────────────────────────────────────────────────────────────
USE_AUGMENTATION = True

# ─── Camera ───────────────────────────────────────────────────────────────────
CAMERA_INDEX         = 0       # 0 = default webcam
CAMERA_FRAME_WIDTH   = 1280
CAMERA_FRAME_HEIGHT  = 720
CAMERA_FPS           = 30
CONFIDENCE_THRESHOLD = 0.60    # Minimum confidence to display prediction

# Best model to use for inference (auto-selected after training)
INFERENCE_MODEL = "EfficientNetB0"   # Change if you prefer another

# ─── Flask Web App ────────────────────────────────────────────────────────────
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
FLASK_DEBUG = False

# ─── Device ───────────────────────────────────────────────────────────────────
import torch


def _resolve_device() -> "torch.device":
    """CUDA if present, else Apple-Silicon MPS, else CPU. Override with TOMATO_DEVICE."""
    forced = os.environ.get("TOMATO_DEVICE")
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = _resolve_device()

# cuDNN autotuner — benchmarks conv algorithms once per input shape and caches
# the fastest. Every image is resized to IMAGE_SIZE×IMAGE_SIZE so the shape never
# varies, which makes this close to free speed on GPU. CUDA-only.
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True

# Mixed precision — big speed/VRAM win on CUDA, unsupported elsewhere.
USE_AMP = DEVICE.type == "cuda"

# pin_memory only helps for host→CUDA copies; on CPU it just warns.
PIN_MEMORY = DEVICE.type == "cuda"

# Per-model batch-size caps (VGG16 is VRAM-hungry; U-Net keeps full-res feature
# maps through its decoder). Models not listed here use BATCH_SIZE as-is.
# These are ceilings, not fixed values — lowering BATCH_SIZE still lowers these,
# so TOMATO_BATCH_SIZE=8 on a small GPU applies to every model.
MODEL_BATCH_SIZE = {
    "VGG16": min(16, BATCH_SIZE),
    "UNet":  min(16, BATCH_SIZE),
}

# ─── Misc ─────────────────────────────────────────────────────────────────────
RANDOM_SEED  = 42

# Windows spawns worker processes, which is costly; override with TOMATO_NUM_WORKERS.
_cpu = os.cpu_count() or 2
NUM_WORKERS  = int(os.environ.get("TOMATO_NUM_WORKERS", 4 if _cpu > 4 else 2))

# Cap images per class — used by the smoke test to run the full pipeline fast.
# 0 / unset = use the entire dataset.
MAX_IMAGES_PER_CLASS = int(os.environ.get("TOMATO_MAX_PER_CLASS", 0))
