"""
config.py — Central configuration for Tomato Disease CNN Project
"""

import os
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
DATA_DIR      = Path(r"C:\Users\ataha\Desktop\PlantVillage")          # ← Point to your dataset folder
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
BATCH_SIZE   = 32
NUM_EPOCHS   = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-4
PATIENCE      = 7           # Early stopping patience

# Train / Val / Test split
TRAIN_SPLIT = 0.70
VAL_SPLIT   = 0.15
TEST_SPLIT  = 0.15

# ─── Models to train ──────────────────────────────────────────────────────────
# Set to True/False to enable/disable each model
MODELS_TO_TRAIN = {
    "CustomCNN":        True,   # Built from scratch (fast baseline)
    "ResNet50":         True,   # Transfer learning
    "EfficientNetB0":   True,   # Transfer learning (lightweight)
    "MobileNetV3":      True,   # Transfer learning (edge-friendly)
    "VGG16":            False,  # Heavy, optional
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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Misc ─────────────────────────────────────────────────────────────────────
RANDOM_SEED  = 42
NUM_WORKERS  = 4 if os.cpu_count() > 4 else 2
