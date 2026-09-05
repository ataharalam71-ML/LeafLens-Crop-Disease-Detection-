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

# ─── Out-of-distribution (OOD) rejection ──────────────────────────────────────
# A 10-way softmax has no way to say "that isn't a tomato leaf" — it will assign
# a coffee cup to a disease with high confidence. utils/ood.py adds an
# abstention layer; see that module's docstring for the method and citations.
OOD_ENABLED = True

# Image-quality gates (model-free, run before inference).
OOD_MIN_VEGETATION = 0.18    # min share of the frame that must read as plant
OOD_MIN_SHARPNESS  = 45.0    # min variance-of-Laplacian; below this it's blurry
OOD_MIN_BRIGHTNESS = 35.0    # mean grey level floor
OOD_MAX_BRIGHTNESS = 235.0   # mean grey level ceiling (blown out)

# Fraction of genuine validation images the calibrated thresholds must keep.
# 0.95 = tune each threshold so 95% of real tomato leaves still get answered.
OOD_TARGET_TPR = 0.95

# Signals are grouped into independent families, and a family casts at most ONE
# vote. msp, margin and entropy are three readings of the same softmax vector:
# on a saturated model they fire together, and letting them vote separately
# would turn one piece of evidence into an instant rejection. Energy (logit
# magnitude) and Mahalanobis (feature geometry) are genuinely separate evidence.
OOD_SIGNAL_FAMILIES = {
    "quality":    ["blurry", "too_dark", "too_bright"],
    "confidence": ["msp", "margin", "entropy"],
    "energy":     ["energy"],
    "feature":    ["mahalanobis"],
}

# Votes needed before an answer is downgraded. 1 family unhappy -> REVIEW,
# 2 or more independent families unhappy -> REJECT.
OOD_REVIEW_VOTES = 1
OOD_REJECT_VOTES = 2

# Clamps on the calibrated thresholds, as (min, max).
#
# Why this is needed: PlantVillage validation accuracy is 99.96%, so after
# temperature scaling the confidence distribution is nearly a point mass at 1.0
# and the 95%-TPR quantile of MSP comes out at 0.9999. That is not an operating
# point, it is an artifact of a validation set with almost no hard examples —
# and it would flag a perfectly good leaf photographed at 98% confidence. The
# clamp keeps the calibrated value when it is informative and falls back to a
# defensible ceiling when the validation distribution has collapsed.
# calibrate_ood.py prints a warning whenever a clamp actually binds.
#
# Energy and Mahalanobis have no absolute scale (they depend on the model's
# logit magnitudes and feature dimension), so they are left unclamped.
OOD_THRESHOLD_LIMITS = {
    "msp":     (0.30, 0.90),
    "margin":  (0.05, 0.80),
    "entropy": (0.05, 0.60),
}

# Covariance shrinkage for the Mahalanobis scorer. Feature dimension is the same
# order as the number of calibration images, so the raw covariance is
# ill-conditioned; this pulls it toward a scaled identity.
OOD_MAHALANOBIS_SHRINKAGE = 0.10

# Used when results/ood_calibration_<model>.json is missing. Deliberately
# conservative: they abstain more than a tuned detector would, and every
# response reports calibrated=False so an uncalibrated run is never mistaken
# for a tuned one. Run  python calibrate_ood.py  to replace them.
OOD_FALLBACK_THRESHOLDS = {
    "msp":     0.60,     # reject below this max softmax probability
    "entropy": 0.35,     # reject above this normalised predictive entropy
    "margin":  0.15,     # reject below this top-1 minus top-2 gap
    "energy": -5.0,      # reject above this free energy  (-logsumexp(logits))
}

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
